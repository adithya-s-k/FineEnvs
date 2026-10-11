"""Build the wind weeks on this machine: this repository doesn't redistribute the readings of Meteocat's station Y7,
so nothing derived from them is committed. The build fetches them, derives the weather weeks with the committed ECMWF
points, solves the 15 dealt pairs, packs them and checks the result against the published pins
(data/meteocat/pins.json).

The steps:

- fetch: the 37 Y7 months into --cache/wind/y7/ (a rerun fetches only the missing ones), each compared with its pin:
  exact, values-match (only the validation state changed, as Meteocat validates), values-changed (warned; the output
  pins decide) or missing (fails).
- weather: tasks/wind/weather.jsonl, the 16 fixed WEEKS from Y7 and data/ecmwf/points.jsonl.gz, and the sources the
  pack's manifest copies (--cache/wind/sources.json). The storm rule is rerun as a diagnostic only.
- solve: the 15 pairs of DEAL with tools/wind_references.py (ortools==9.15.6755; about 9 minutes on 14 processes, set
  by the slowest pair), each row cached in --cache/wind/screen/ by the digest of its inputs and of the solving code.
- pack, references: tasks/dock-v1-wind/{tasks.jsonl, manifest.json} and tasks/wind/references.jsonl, replaying the
  stored plans.
- verify: the pack's tasks and the references against their pins, listing the task ids whose lines differ and exiting
  1 on a mismatch. Drifted weeks stay in place, but rollout.py won't play them until --accept-drift records the drift
  in the pack's manifest; their scores are then not comparable with the published runs. weather.jsonl's pin is
  informational: validation changes its unvalidated shares only.
- all (the default): each in turn, stopping at the first failure.

Outputs go under packs.tasks_root() (PORT_WEEK_TASKS_DIR, else envs/port_week/tasks/), which must also hold the
committed dock-v1-marine/, live/ and marine/; a NOTICE goes beside them, and they are git-ignored. Run from
envs/port_week:

    uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/build_wind.py [COMMAND] [--cache DIR]
        [--workers N] [--accept-drift]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import berth_core
import ortools
import port_week_core
from port_week_core import marine, packs

import live_references
import wind_references
import wind_weather as weather_tools

DEAL = [("dock-24B-w06x1-busy-0", "e15"), ("dock-24B-w07x1-busy-0", "e04"), ("dock-24B-w07x1-busy-0", "e08"),
        ("dock-24B-w07x1-busy-0", "e09"), ("dock-24B-w16x1-busy-0", "e12"), ("dock-24B-w35x1-busy-0", "e00"),
        ("dock-36A-w05x1-busy-0", "e09"), ("dock-36A-w05x1-busy-0", "e14"), ("dock-36A-w06x1-busy-0", "e07"),
        ("dock-36A-w06x1-busy-0", "e12"), ("dock-36A-w17x1-busy-0", "e12"), ("dock-36A-w37x1-busy-0", "e12"),
        ("dock-24B-w37x1-standard-0", "e01"), ("dock-36A-w06x1-standard-0", "e07"),
        ("dock-36A-w35x1-standard-0", "e03")]
"""The published pairs of (marine week, weather week), dealt from the full 240-pair screen, in the pack's order; fixed,
so the task ids and the solve are the same on every machine."""
DEAL_STATS = {"screened": 240, "qualifying": 26, "dealt": 15, "gate": 12}
STATES = ("exact", "values-match", "values-changed", "missing")
OUTPUTS = {"tasks": f"{packs.WIND_PACK}/tasks.jsonl", "references": "wind/references.jsonl",
           "weather": "wind/weather.jsonl"}
INFORMATIONAL = ("weather",)
NOTICE = """Built on this machine by tools/build_wind.py on {built}. Not part of the repository.

The wind env's observed windows and forecast calibration are built on your machine by build_wind.py from the readings
of the Servei Meteorològic de Catalunya's (Meteocat) XEMA station Y7, Port de Barcelona – Bocana Sud ("Dades
meteorològiques de la XEMA", dataset nzvn-apee, Generalitat de Catalunya open data portal:
https://analisi.transparenciacatalunya.cat/d/nzvn-apee). The repository holds no reading and no data derived from
them, only the identifiers the build fixes (the 16 ISO weeks with their kind, the 15 pairs) and digests (each month's
row count and sha256). The files the build writes keep the terms published there
(https://www.meteo.cat/wpweb/avis-legal/; https://web.gencat.cat/ca/generalitat/dades-indicadors/dades-obertes/llicencies),
which the repository doesn't relicense: check them before you share these files, the results and transcripts of wind
runs, or an image or dataset made with them.
  Font: Servei Meteorològic de Catalunya (Meteocat), estació XEMA Y7; dades extretes el {extracted}; darrera actualització {last_modified}; finestres derivades

The forecasts are modified ECMWF open data.
  Copyright: © 2023-2026 European Centre for Medium-Range Weather Forecasts (ECMWF).
  Source: www.ecmwf.int
  Licence Statement: This data is published under a Creative Commons Attribution 4.0 International (CC BY 4.0). https://creativecommons.org/licenses/by/4.0/
  Disclaimer: ECMWF does not accept any liability whatsoever for any error or omission in the data, their availability, or for any loss or damage arising from their use.
  This data is based on data and products of the European Centre for Medium-Range Weather Forecasts (ECMWF).
  Modified: each run's 10 m wind at one grid point, every 3 hours to 72 hours, mapped to the Y7 anemometer by quantiles and rounded to whole knots.
  Terms of use: https://apps.ecmwf.int/datasets/licences/general/
"""  # noqa: E501


def pins() -> dict:
    return json.loads(weather_tools.PINS.read_text())


def y7_dir(cache: Path) -> Path:
    return cache / "wind" / "y7"


def classify(directory: Path, months: dict[str, dict]) -> dict[str, str]:
    """Each pinned month as fetched: exact, values-match, values-changed or missing."""
    out = {}
    for month, pin in months.items():
        meta_path = directory / f"{month}.meta.json"
        if not meta_path.is_file():
            out[month] = "missing"
            continue
        meta = json.loads(meta_path.read_text())
        if (meta["rows"], meta["sha256"]) == (pin["rows"], pin["sha256"]):
            out[month] = "exact"
            continue
        rows = json.loads((directory / f"{month}.json").read_bytes())
        out[month] = "values-match" if weather_tools.values_digest(rows) == pin["values_sha256"] else "values-changed"
    return out


def report(states: dict[str, str]) -> None:
    print("Y7 months against their pins: " + ", ".join(f"{sum(s == state for s in states.values())} {state}"
                                                       for state in STATES))
    for state, note in (("values-match", "validation state changed since the pins, readings the same"),
                        ("values-changed", "readings changed since the pins; the output pins decide"),
                        ("missing", "not fetched; rerun fetch")):
        if months := [m for m, s in states.items() if s == state]:
            print(f"  {state}: {', '.join(months)} ({note})")


def fetch(args: argparse.Namespace) -> int:
    _, failures = weather_tools.fetch_y7(y7_dir(args.cache))
    states = classify(y7_dir(args.cache), pins()["y7"]["months"])
    report(states)
    return 1 if failures or "missing" in states.values() else 0


def notice(directory: Path, y7: dict) -> None:
    weather_tools.write_atomic(directory / "NOTICE", NOTICE.format(
        built=f"{datetime.now(UTC):%Y-%m-%d}", extracted=y7["extracted"], last_modified=y7["last_modified"]).encode())


def weather(args: argparse.Namespace) -> int:
    directory, pinned = y7_dir(args.cache), pins()
    states = classify(directory, pinned["y7"]["months"])
    if "missing" in states.values():
        report(states)
        return 1
    points = weather_tools.POINTS.read_bytes()
    if hashlib.sha256(points).hexdigest() != pinned["ecmwf"]["sha256"]:
        print(f"{weather_tools.POINTS} does not match its pin in {weather_tools.PINS}")
        return 1
    metas = {m: json.loads((directory / f"{m}.meta.json").read_text()) for m in pinned["y7"]["months"]}
    obs = weather_tools.readings(directory)
    rule = weather_tools.candidate_weeks(obs)
    if set(rule["weeks"]) != set(weather_tools.WEEKS):
        print(f"Note: the storm rule now gives {sorted(rule['weeks'])} (dropped {rule['dropped']}); the build keeps "
              "the 16 fixed weeks")
    records, fits = weather_tools.weather(obs, weather_tools.read_points(weather_tools.POINTS))
    root = packs.tasks_root()
    weather_tools.write_atomic(root / OUTPUTS["weather"], weather_tools.weather_body(records))
    sources = {"ecmwf": weather_tools.ecmwf_sources(records, pinned["ecmwf"]["sha256"]),
               "y7": weather_tools.y7_sources(metas, states),
               "calibration": weather_tools.calibration_sources(fits)}
    weather_tools.write_json(args.cache / "wind" / "sources.json", sources)
    notice(root / "wind", sources["y7"])
    print(f"Wrote {root / OUTPUTS['weather']}: {len(records)} weeks, {sum(len(r['runs']) for r in records)} runs")
    return 0


def code_digest() -> str:
    """The code a row comes from: the reference tools, port_week_core and berth_core."""
    files = sorted(Path(__file__).parent.glob("*_references.py"))
    for package in (port_week_core, berth_core):
        files += sorted(Path(package.__file__).parent.rglob("*.py"))
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()


def inputs_digest(task: dict, record: dict) -> str:
    """What a pair's row depends on: the marine week, the weather week, the solver, its code and the Python minor
    version."""
    inputs = [task, record, ortools.__version__, f"{sys.version_info.major}.{sys.version_info.minor}",
              live_references.DETERMINISTIC_SECONDS, live_references.CONFIGS, code_digest()]
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def screen_paths(cache: Path) -> dict[tuple[str, str], tuple[Path, dict]]:
    """Each dealt pair's cached row path for the current inputs, with its weather record."""
    weathers = {w["id"]: w for w in map(json.loads, (packs.tasks_root() / OUTPUTS["weather"]).read_text().splitlines())}
    v3 = marine.pack()
    return {(s, w): (cache / "wind" / "screen" / f"{s}-{w}.{inputs_digest(v3.get(s).to_dict(), weathers[w])[:16]}.json",
                     weathers[w]) for s, w in DEAL}


def solve(args: argparse.Namespace) -> int:
    todo = {pair: (path, record) for pair, (path, record) in screen_paths(args.cache).items() if not path.is_file()}
    print(f"Solving {len(todo)} of {len(DEAL)} pairs on {args.workers} processes; {len(DEAL) - len(todo)} cached")
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(wind_references.pair, s, record): (s, w) for (s, w), (_, record) in todo.items()}
        for future in as_completed(futures):
            row, seconds = future.result()
            weather_tools.write_json(todo[futures[future]][0], row)
            print(f"{row['task_id']}: hindsight {row.get('hindsight_cost')}, anchor {row.get('optimal_cost')}, "
                  f"qualifies {row['qualifies']}, {seconds:.0f} s, at {time.monotonic() - started:.0f} s", flush=True)
    return 0


def solved(cache: Path) -> tuple[list[dict], dict[tuple[str, str], dict]] | None:
    paths = screen_paths(cache)
    if lacking := [f"{s}-{w}" for (s, w), (path, _) in paths.items() if not path.is_file()]:
        print(f"Not solved for the current inputs: {', '.join(lacking)}; run solve")
        return None
    picks = [{"schedule": s, "weather": w} for s, w in DEAL]
    return picks, {pair: json.loads(path.read_text()) for pair, (path, _) in paths.items()}


def pack(args: argparse.Namespace) -> int:
    if not (got := solved(args.cache)):
        return 1
    sources = json.loads((args.cache / "wind" / "sources.json").read_text())
    root = packs.tasks_root()
    out = wind_references.write_pack(root, root / OUTPUTS["weather"], *got, sources, sources["ecmwf"]["messages"],
                                     DEAL_STATS)
    notice(out, sources["y7"])
    return 0


def references(args: argparse.Namespace) -> int:
    if not (got := solved(args.cache)):
        return 1
    wind_references.write_references(packs.tasks_root(), *got)
    return 0


def line_digests(body: bytes) -> dict[str, str]:
    out = {}
    for line in body.splitlines():
        record = json.loads(line)
        out[record.get("task_id") or record["id"]] = hashlib.sha256(line).hexdigest()
    return out


def compare(root: Path, outputs: dict[str, dict]) -> dict[str, list[str] | None]:
    """Each output's differing line ids against its pin, [] when identical, None when it is missing."""
    out: dict[str, list[str] | None] = {}
    for name, rel in OUTPUTS.items():
        path, pin = root / rel, outputs[rel]
        if not path.is_file():
            out[name] = None
            continue
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() == pin["sha256"]:
            out[name] = []
            continue
        got = line_digests(body)
        out[name] = sorted(k for k in got.keys() | pin["lines"].keys() if got.get(k) != pin["lines"].get(k))
    return out


def verify(args: argparse.Namespace) -> int:
    root = packs.tasks_root()
    diffs = compare(root, pins()["outputs"])
    for name, ids in diffs.items():
        state = "missing" if ids is None else "identical" if not ids else f"differs in {', '.join(ids)}"
        print(f"{OUTPUTS[name]}: {state}" + (" (informational)" if name in INFORMATIONAL else ""))
    drift = {name: ids for name, ids in diffs.items() if name not in INFORMATIONAL and ids != []}
    if not drift:
        print("The wind tasks and references are identical to agentenv-portsim v0.4.5's")
        return 0
    if args.accept_drift and all(ids is not None for ids in drift.values()):
        path = root / packs.WIND_PACK / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["drift"] = {"accepted": f"{datetime.now(UTC):%Y-%m-%d}", **drift,
                             "note": "the scores are not comparable with the published wind runs"}
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"Drift accepted and recorded in {path}")
        return 0
    if any(ids is None for ids in drift.values()):
        print("The wind weeks are not built; run python tools/build_wind.py")
    else:
        print("The built wind weeks differ from the published ones and stay in place; rollout.py plays them once "
              "--accept-drift records the drift in the pack's manifest")
    return 1


COMMANDS = {"fetch": fetch, "weather": weather, "solve": solve, "pack": pack, "references": references,
            "verify": verify}


def build_all(args: argparse.Namespace) -> int:
    timings = {}
    for name, command in COMMANDS.items():
        started = time.monotonic()
        print(f"== {name}", flush=True)
        code = command(args)
        timings[name] = time.monotonic() - started
        if code:
            print(f"{name} failed after {timings[name]:.0f} s")
            return code
    print("Done: " + ", ".join(f"{name} {seconds:.0f} s" for name, seconds in timings.items()))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", nargs="?", default="all", choices=[*COMMANDS, "all"])
    parser.add_argument("--cache", type=Path, default=weather_tools.cache_dir(),
                        help="the Y7 readings, sources and solved rows (default ~/.cache/portsim-live)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="solve processes (default: every core)")
    parser.add_argument("--accept-drift", action="store_true", help="keep outputs that differ from the pins")
    args = parser.parse_args()
    raise SystemExit(build_all(args) if args.command == "all" else COMMANDS[args.command](args))


if __name__ == "__main__":
    main()
