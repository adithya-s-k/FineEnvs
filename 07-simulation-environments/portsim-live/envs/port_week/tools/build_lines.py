"""Build the lines weeks on this machine, after the wind weeks: they are the wind weeks with the ships' agents, so they
derive from Meteocat's readings too and nothing of them is committed. The build adds each ship's agent, its previous
port's distance and its ETA error from the port's pinned 2024 calls, plays every week four ways, sets each week's
anchor and checks the result against the published pins (data/meteocat/pins.json).

The steps:

- pack: the wind weeks (built by tools/build_wind.py and identical to the published ones, or with their drift
  accepted) with the agents, as tools/lines_references.py pack writes them, into --cache/lines/stage/; the port's 2024
  calls are downloaded once into ~/.cache/portsim-live/raw/ and checked against their pin.
- screen: the hindsight optimum, the follower under its 4 solver configurations, the silent follower and the naive
  policy on each staged week (ortools==9.15.6755; about 27 minutes on 14 processes, set by the slowest week), each row
  cached in --cache/lines/screen/ by the digest of its week and of the solving code.
- references: tasks/dock-v1-lines/{tasks.jsonl, manifest.json}, each week with its anchor, and
  tasks/lines/references.jsonl, replaying the follower's stored plans and requests.
- verify: the pack's tasks, its manifest and the references against their pins, listing the task ids whose lines
  differ and exiting 1 on a mismatch. The references are held to their pins but for the summaries of the follower's
  and the silent follower's 8-worker configuration without the tie-break, which CP-SAT doesn't reproduce from run to
  run (two runs of the plugin's own script differ there): a week that differs only there is listed, and passes.
  Drifted weeks stay in place, but rollout.py won't play them until --accept-drift records the drift in the pack's
  manifest; their scores are then not comparable with the published runs.
- all (the default): each in turn, stopping at the first failure.

Outputs go under packs.tasks_root(), beside the wind weeks, with a NOTICE in each directory; they are git-ignored.
Run from envs/port_week:

    uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/build_lines.py [COMMAND] [--cache DIR]
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

import ortools
from berth_core import TaskPack
from port_week_core import packs, wind

import build_wind
import lines_references
import live_references
import marine_references
import wind_weather as weather_tools

OUTPUTS = {"tasks": f"{packs.LINES_PACK}/tasks.jsonl", "manifest": f"{packs.LINES_PACK}/manifest.json",
           "references": "lines/references.jsonl"}
UNREPRODUCIBLE = {"workers": 8, "tiebreak": False}
"""The solver configuration whose plays CP-SAT doesn't repeat from run to run, though every other one it does."""
NOTICE = """Built on this machine by tools/build_lines.py on {built}, from the wind weeks tools/build_wind.py built
here. Not part of the repository.

The ships' agents, their ships' calls and the other traffic are derived from the Port de Barcelona's 2024 calls of
every ship type ("Contains data from the Port de Barcelona open data portal", CC BY-SA 4.0;
data/barcelona/SOURCE.md); the previous ports' distances are NGA Pub. 151's, Distances Between Ports (no copyright
claimed).

"""


def stage_dir(cache: Path) -> Path:
    return cache / "lines" / "stage"


def notice(directory: Path, cache: Path) -> None:
    """The lines NOTICE, then the wind one: the weeks keep the terms of the readings and forecasts they come from."""
    y7 = json.loads((cache / "wind" / "sources.json").read_text())["y7"]
    wind_notice = build_wind.NOTICE.split("\n\n", 1)[1].format(extracted=y7["extracted"],
                                                                last_modified=y7["last_modified"])
    weather_tools.write_atomic(directory / "NOTICE", (NOTICE.format(built=f"{datetime.now(UTC):%Y-%m-%d}")
                                                      + wind_notice).encode())


def wind_ready() -> bool:
    """The wind weeks are built and are the published ones, or their drift was accepted."""
    root = packs.tasks_root()
    if build_wind.compare(root, build_wind.pins()["outputs"])["tasks"] == []:
        return True
    manifest = wind.pack_dir() / "manifest.json"
    if manifest.is_file() and "drift" in json.loads(manifest.read_text()):
        print("The wind weeks drifted from the published ones (accepted); the lines weeks built on them will too")
        return True
    print("The wind weeks are not built, or differ from the published ones: run python tools/build_wind.py first")
    return False


def pack(args: argparse.Namespace) -> int:
    if not wind_ready():
        return 1
    lines_references.write_pack(stage_dir(args.cache), lines_references.lines_tasks(marine_references.raw(None)))
    return 0


def staged(cache: Path) -> TaskPack | None:
    if not (stage_dir(cache) / packs.LINES_PACK / "tasks.jsonl").is_file():
        print("The lines weeks are not staged; run pack")
        return None
    return TaskPack(stage_dir(cache) / packs.LINES_PACK)


def inputs_digest(task: dict) -> str:
    """What a week's row depends on: the staged week, the solver, its code and the Python minor version."""
    inputs = [task, ortools.__version__, f"{sys.version_info.major}.{sys.version_info.minor}",
              live_references.DETERMINISTIC_SECONDS, live_references.CONFIGS, build_wind.code_digest()]
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def screen_paths(cache: Path, tasks: TaskPack) -> dict[str, tuple[Path, dict]]:
    """Each staged week's cached row path for the current inputs, with the week."""
    out = {}
    for t in tasks.tasks:
        task = t.to_dict()
        out[t.task_id] = (cache / "lines" / "screen" / f"{t.task_id}.{inputs_digest(task)[:16]}.json", task)
    return out


def screen(args: argparse.Namespace) -> int:
    if not (tasks := staged(args.cache)):
        return 1
    paths = screen_paths(args.cache, tasks)
    todo = {task_id: (path, task) for task_id, (path, task) in paths.items() if not path.is_file()}
    print(f"Screening {len(todo)} of {len(paths)} weeks on {args.workers} processes; {len(paths) - len(todo)} cached")
    started = time.monotonic()
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(lines_references.week, task): task_id for task_id, (_, task) in todo.items()}
        for future in as_completed(futures):
            row, seconds = future.result()
            weather_tools.write_json(todo[futures[future]][0], row)
            print(f"{row['task_id']}: " + (f"error {row['error']}" if "error" in row else (
                f"hindsight {row['hindsight_cost']}, anchor {row['optimal_cost']}, follower "
                f"{[c['cost'] for c in row['configs']]}, silent {[c['reward'] for c in row['silent']['configs']]}, "
                f"naive {row['naive']['reward']}, qualifies {row['qualifies']}"))
                  + f", {seconds:.0f} s, at {time.monotonic() - started:.0f} s", flush=True)
    return 0


def references(args: argparse.Namespace) -> int:
    if not (tasks := staged(args.cache)):
        return 1
    paths = screen_paths(args.cache, tasks)
    if lacking := [task_id for task_id, (path, _) in paths.items() if not path.is_file()]:
        print(f"Not screened for the current inputs: {', '.join(lacking)}; run screen")
        return 1
    root = packs.tasks_root()
    rows = {task_id: json.loads(path.read_text()) for task_id, (path, _) in paths.items()}
    lines_references.write_references(root, tasks.tasks, rows)
    for directory in (root / packs.LINES_PACK, root / "lines"):
        notice(directory, args.cache)
    return 0


def reproducible(row: dict) -> dict:
    """A references row without the summaries of the configuration CP-SAT doesn't reproduce."""
    def kept(configs: list[dict]) -> list[dict]:
        return [c for c in configs if {k: c.get(k) for k in UNREPRODUCIBLE} != UNREPRODUCIBLE]
    silent = row["silent"] | {"configs": kept(row["silent"]["configs"])}
    return row | {"configs": kept(row["configs"]), "silent": silent}


def reproducible_digests(body: bytes) -> dict[str, str]:
    return {row["task_id"]: hashlib.sha256(json.dumps(reproducible(row)).encode()).hexdigest()
            for row in map(json.loads, body.splitlines())}


def compare(root: Path, outputs: dict[str, dict]) -> tuple[dict[str, list[str] | None], list[str]]:
    """Each output's differing line ids against its pin (the manifest's ["manifest"] when it differs), [] when
    identical, None when it is missing; and the references' weeks that differ only in what CP-SAT doesn't reproduce,
    which count as identical."""
    out: dict[str, list[str] | None] = {}
    varying: list[str] = []
    for name, rel in OUTPUTS.items():
        path, pin = root / rel, outputs[rel]
        if not path.is_file():
            out[name] = None
            continue
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() == pin["sha256"]:
            out[name] = []
        elif "lines" not in pin:
            out[name] = [name]
        else:
            got = build_wind.line_digests(body)
            ids = sorted(k for k in got.keys() | pin["lines"].keys() if got.get(k) != pin["lines"].get(k))
            if "reproducible_lines" in pin:
                stable = reproducible_digests(body)
                varying = [k for k in ids if stable.get(k) == pin["reproducible_lines"].get(k)]
                ids = [k for k in ids if k not in varying]
            out[name] = ids
    return out, varying


def verify(args: argparse.Namespace) -> int:
    root = packs.tasks_root()
    diffs, varying = compare(root, build_wind.pins()["outputs"])
    for name, ids in diffs.items():
        state = "missing" if ids is None else "identical" if not ids else f"differs in {', '.join(ids)}"
        if name == "references" and ids == [] and varying:
            state += (f", but for the 8-worker untied follower's summaries in {', '.join(varying)}, which CP-SAT "
                      "doesn't reproduce from run to run")
        print(f"{OUTPUTS[name]}: {state}")
    drift = {name: ids for name, ids in diffs.items() if ids != []}
    if not drift:
        print("The lines tasks, manifest and references are identical to the agentenv-portsim plugin's")
        return 0
    if args.accept_drift and all(ids is not None for ids in drift.values()):
        path = root / packs.LINES_PACK / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["drift"] = {"accepted": f"{datetime.now(UTC):%Y-%m-%d}", **drift,
                             "note": "the scores are not comparable with the published lines runs"}
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"Drift accepted and recorded in {path}")
        return 0
    if any(ids is None for ids in drift.values()):
        print("The lines weeks are not built; run python tools/build_lines.py")
    else:
        print("The built lines weeks differ from the published ones and stay in place; rollout.py plays them once "
              "--accept-drift records the drift in the pack's manifest")
    return 1


COMMANDS = {"pack": pack, "screen": screen, "references": references, "verify": verify}


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
                        help="the wind build's cache, the staged weeks and the screened rows (default "
                             "~/.cache/portsim-live)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="screen processes (default: every core)")
    parser.add_argument("--accept-drift", action="store_true", help="keep outputs that differ from the pins")
    args = parser.parse_args()
    raise SystemExit(build_all(args) if args.command == "all" else COMMANDS[args.command](args))


if __name__ == "__main__":
    main()
