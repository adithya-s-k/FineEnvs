"""Compute the wind pack, tasks/dock-v1-wind/, and tasks/wind/references.jsonl: the qualified marine weeks, each
transplanted into weather weeks of tasks/wind/weather.jsonl by port_week_core.wind.transplant and played live through
port_week_core.wind.WindWeek, as tools/marine_references.py does for portsim-marine.

Three rolling CP-SAT re-planners solve the week at every watch with live_references.solve under forecast relief and
marine_references' pilot and tug pools for the ships still open: the follower on the week as known, the blind one with
the wind taken out of it, and the hold-every-warning one on every window any delivered forecast showed. The hindsight
optimum is solved on the wind that blew, to a proven optimum; the task's reference cost is the anchor, the lower of
that optimum and the follower's best net cost. A pair qualifies when (a) the follower reaches the hindsight optimum in
at least 3 of its 4 solver configurations, (b) the blind re-planner (in a bust week the hold-every-warning one) scores
at most 0.9 in at least 3 of 4, (c) the week has at most 10 watches and (d) no ship alongside at hour 0 leaves inside a
window above 30 kn.

tools/build_wind.py plays only the 15 pairs of its fixed deal and packs them. This command plays every pair (screen),
picks the pairs (deal), writes the pack from the screen and the deal (pack, with weather-sources.json beside the
weather), and replays the packed pairs' stored plans without solving and writes their records (references): how the
deal was chosen, and how the test fixtures are built from synthetic weather. Run from envs/port_week; the screen and
the deal default to ~/.cache/portsim-live/wind/, the rest to packs.tasks_root().

    uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/wind_references.py screen
        [--weather PATH] [--schedules id,...] [--out PATH]
    uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/wind_references.py deal [--screen PATH]
        [--out PATH]
    uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/wind_references.py pack|references
        [--weather PATH] [--screen PATH] [--deal PATH] [--out DIR]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import berth_core
import ortools
from berth_core import Plan, Task, TaskPack, plan_from_list, plan_to_list
from berth_core.check import unavoidable_cost
from berth_core.solve import naive_replan
from ortools.sat.python import cp_model
from port_week_core import marine, packs, schedule, wind, world
from port_week_core.marine import CONTAINER
from port_week_core.wind import WindWeek

from live_references import CONFIGS, DETERMINISTIC_SECONDS, outcome, solve
from marine_references import ORDINANCE, constraints
from wind_weather import cache_dir

WEATHER = packs.tasks_root() / "wind/weather.jsonl"
SCREEN = cache_dir() / "wind/screen.jsonl"
DEAL = cache_dir() / "wind/deal.json"
REACHED = 3
BLIND_BAR = 0.9
MAX_WATCHES = (schedule.WIND_MAX_TURNS - 2) // schedule.TURNS_PER_WATCH
GATE = 12
MOST = 15
ECMWF = "https://www.ecmwf.int/en/forecasts/datasets/open-data"
XEMA = "https://analisi.transparenciacatalunya.cat/d/nzvn-apee"
BUCKET = "https://registry.opendata.aws/ecmwf-forecasts/"
GROUNDING = {
    "rule": {"value": "above 25 kn no ship of 300 m or more berths or leaves, above 30 kn no ship does", "basis": (
        f"The port's traffic ordinance, BOE-A-2023-6719 ({ORDINANCE}), 4.1 and 4.1.2.1: the thresholds are 25 kn for "
        "container ships of 300 m or more and 30 kn for the others, reached or forecast by the official forecasts. "
        "They start a review with the pilots; PortSimEnv simplifies them to no-movement windows, as v1 does.")},
    "anemometer": {"value": "Meteocat XEMA station Y7, Port de Barcelona - Bocana Sud", "basis": (
        f"BOE-A-2023-6719 ({ORDINANCE}), annex III, measures the wind at the anemometer of the Dique Sur's red light; "
        "taking Y7 for it is an assumption. Its readings come from the Generalitat de Catalunya's open data portal, "
        f"dataset nzvn-apee ({XEMA}).")},
    "reading": {"value": "an hour is above T when max(1.045 x the 30-minute mean, 0.664 x the 3-second gust) > T",
                "basis": (
        f"BOE-A-2023-6719 ({ORDINANCE}), annex III, takes the 10-minute mean, or the largest 1-minute mean above "
        "1.25 T. Y7 publishes a 30-minute mean and a 3-second gust; the factors 1.045 (10- over 30-minute mean) and "
        "0.83 (1-minute mean over 3-second gust, so 0.83 / 1.25 = 0.664) are assumptions.")},
    "windows": {"value": "the hours above each threshold, gaps of 2 hours or less merged", "basis": "Assumption."},
    "forecast": {"value": "the four daily HRES runs of ECMWF open data, to 72 hours, at the nearest sea grid point",
                 "basis": (f"ECMWF open data ({ECMWF}), CC BY 4.0; that the port's official forecasts are like it is "
                           "an assumption.")},
    "published": {"value": "a run is delivered 9 hours after its start time; a run that never came leaves the one "
                           "before it in force", "basis": (
        f"Assumption, after the times the runs appeared in the public bucket ({BUCKET}).")},
    "calibration": {"value": "each run's wind mapped to the anemometer by quantiles fitted on another year, in whole "
                             "knots every 3 hours", "basis": "Assumption."},
    "watches": {"value": "v3's watches, plus each 12-hourly hour whose delivered run restricts an hour of the next 36 "
                         "past the freeze line that the last watch's run didn't", "basis": "Assumption."},
    "wind_tug": {"value": "one more tug per movement inside a window above 25 kn, as in dock-v1-marine", "basis": (
        f"BOE-A-2023-6719 ({ORDINANCE}), 4.1.2.2, as dock-v1-marine's manifest sets out under marine.wind.")},
}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def blind(week: world.Week) -> Task:
    view = week.view
    return replace(view, rules=view.rules | {"no_moves": []})


VIEWS = {"follow": lambda week: week.view, "blind": blind, "hold": wind.held}


def pools(view: Task, fixed: Plan) -> Callable[[cp_model.CpModel, dict], None]:
    """marine_references' pools for the open ships, the frozen windows' movements counted with the port's other
    traffic: wind that holds several frozen ships to one hour may need more than the whole pool, and none can move."""
    marine_rules = view.rules["marine"]
    moves = marine.own_moves(view, berth_core.evaluate(view, fixed))
    frozen = [[hour, CONTAINER, view.ships[s].length_m] for s, hour, _ in moves if hour < marine_rules["hours"]]
    extra = constraints(replace(view, rules=view.rules | {"marine": marine_rules | {
        "movements": sorted(marine_rules["movements"] + frozen)}}), {})
    return lambda md, ships: extra(md, {s: hours for s, hours in ships.items() if s not in fixed})


def rolling(mode: str, workers: int, tiebreak: bool, plans: list[list[dict]]) -> world.Policy:
    def policy(week: world.Week) -> Plan:
        view = VIEWS[mode](week)
        plan = solve(view, week.fixed, week.before, week.plan if week.watch else naive_replan(view), workers, tiebreak,
                     extra=pools(view, week.fixed), forecast_relief=True)
        plans.append(plan_to_list(plan))
        return plan
    return policy


def played(task: Task, policy: world.Policy) -> WindWeek:
    week = world.play(task, policy, week=WindWeek)
    assert week.refusals == []
    return week


def configs(task: Task, mode: str) -> list[tuple[dict, WindWeek | None, list[list[dict]]]]:
    """Each configuration's row, its finished week and its plans; a re-solve that fails ends the play."""
    out = []
    for workers, tiebreak in CONFIGS:
        row, plans = {"workers": workers, "tiebreak": tiebreak}, []
        try:
            out.append((row, played(task, rolling(mode, workers, tiebreak, plans)), plans))
        except RuntimeError as e:
            out.append((row | {"cost": None, "reward": None, "error": str(e)}, None, plans))
    return out


def regraded(week: WindWeek, task: Task) -> dict:
    return outcome(world.grade_week(task, week.plan, *week.excuses(), evaluate=week.evaluate))


def summary(runs: list[tuple[dict, WindWeek | None, list[list[dict]]]], task: Task) -> tuple[dict | None, list[dict]]:
    """The first completed configuration's outcome and plans, and every configuration's cost and reward."""
    first, rows = None, []
    for k, (row, week, plans) in enumerate(runs):
        if week is not None:
            graded = regraded(week, task)
            row = row | {"cost": graded["cost"], "reward": graded["reward"]}
            if first is None:
                first = {"config": k, **graded, "plans": plans}
        rows.append(row)
    return first, rows


def hindsight(task: Task, v3: Task) -> tuple[int, Plan]:
    plan = solve(task, {}, 0, plan_from_list(v3.reference["optimal_plan"]), 1, False, extra=constraints(task, {}),
                 optimal=True)
    res = marine.evaluate(task, plan)
    assert res.feasible, task.task_id
    return res.cost, plan


def blocked(task: Task) -> bool:
    return any(w["start"] <= b.end < w["end"] for b in task.blocks if b.kind == "alongside"
               for w in task.rules["no_moves"] if w["min_length"] == 0)


def screened(v3: Task, weather: dict) -> dict:
    task = wind.transplant(v3, weather)
    v3_hours = [w.hour for w in schedule.schedule(v3)]
    hours = [w.hour for w in schedule.schedule(task)]
    row = {"task_id": task.task_id, "schedule": v3.task_id, "weather": weather["id"], "kind": weather["kind"],
           "watch_hours": hours, "added_watches": [h for h in hours if h not in v3_hours]}
    try:
        cost, plan = hindsight(task, v3)
    except RuntimeError as e:
        return row | {"error": str(e), "qualifies": False}
    follow = configs(task, "follow")
    anchor = min([cost] + [w.grade["cost"] for _, w, _ in follow if w is not None and w.grade["cost"] is not None])
    final = replace(task, reference=v3.reference | {"optimal_cost": anchor, "optimal_plan": plan_to_list(plan),
                                                    "lower_bound": cost, "proven_optimal": True})
    rolled, rows = summary(follow, final)
    other = "blind" if weather["kind"] == "storm" else "hold"
    policies = {mode: summary(configs(task, mode), final) for mode in ("blind", "hold")
                if mode == "blind" or weather["kind"] == "bust"}
    row |= {"optimal_cost": anchor, "hindsight_cost": cost, "v3_optimal_cost": v3.reference["optimal_cost"],
            "unavoidable_cost": unavoidable_cost(task), "rolling": rolled, "configs": rows}
    row |= {mode: (first or {}) | {"configs": runs} for mode, (first, runs) in policies.items()}
    clauses = {"a": sum(c["cost"] is not None and c["cost"] <= cost for c in rows) >= REACHED,
               "b": sum(c["reward"] is not None and c["reward"] <= BLIND_BAR for c in row[other]["configs"]) >= REACHED,
               "c": len(hours) <= MAX_WATCHES, "d": not blocked(task)}
    return row | {"naive": outcome(played(final, world.naive).grade), "clauses": clauses,
                  "qualifies": all(clauses.values()),
                  "solver": {"ortools": ortools.__version__, "max_deterministic_time": DETERMINISTIC_SECONDS},
                  "hindsight_plan": plan_to_list(plan)}


def pair(schedule_id: str, weather: dict) -> tuple[dict, float]:
    started = time.monotonic()
    return screened(marine.pack().get(schedule_id), weather), time.monotonic() - started


def screen(args: argparse.Namespace) -> None:
    weathers = read(args.weather)
    ids = args.schedules.split(",") if args.schedules else marine.task_ids()
    done = {(r["schedule"], r["weather"]) for r in read(args.out)} if args.out.is_file() else set()
    todo = [(s, w) for s in ids for w in weathers if (s, w["id"]) not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ProcessPoolExecutor(os.cpu_count()) as pool, args.out.open("a") as out:
        for future in as_completed([pool.submit(pair, s, w) for s, w in todo]):
            row, seconds = future.result()
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{row['task_id']}: {len(row['watch_hours'])} watches (+{row['added_watches']}), "
                  + (f"error {row['error']}" if "error" in row else
                     f"hindsight {row['hindsight_cost']}, anchor {row['optimal_cost']}, follow "
                     f"{[c['cost'] for c in row['configs']]}, blind {[c['reward'] for c in row['blind']['configs']]}, "
                     + (f"hold {[c['reward'] for c in row['hold']['configs']]}, " if "hold" in row else "")
                     + f"naive {row['naive']['reward']}, clauses {row['clauses']}")
                  + f", qualifies {row['qualifies']}, {seconds:.0f}s", flush=True)
    print(f"Screened {len(todo)} pairs into {args.out} in {time.monotonic() - started:.0f}s, {len(done)} there before")


def dealt(rows: list[dict]) -> list[dict]:
    """Round 1: each schedule in turn takes the first qualifying storm week no other has, from its own index on;
    each bust week goes to the first schedule it qualifies on; below the gate, round 2: each schedule in turn takes
    the first qualifying storm week it doesn't have yet, again from its index, until the most."""
    ok = {(r["schedule"], r["weather"]) for r in rows if r["qualifies"]}
    storms = sorted({r["weather"] for r in rows if r["kind"] == "storm"})
    busts = sorted({r["weather"] for r in rows if r["kind"] == "bust"})
    ids = marine.task_ids()

    def turn(i: int) -> list[str]:
        k = i % len(storms)
        return storms[k:] + storms[:k]

    picks = []
    for i, s in enumerate(ids):
        if w := next((w for w in turn(i) if (s, w) in ok and all(w != p[1] for p in picks)), None):
            picks.append((s, w))
    for w in busts:
        if taker := next((s for s in ids if (s, w) in ok), None):
            picks.append((taker, w))
    if len(picks) < GATE:
        for i, s in enumerate(ids):
            if len(picks) >= MOST:
                break
            if w := next((w for w in turn(i) if (s, w) in ok and (s, w) not in picks), None):
                picks.append((s, w))
    return [{"schedule": s, "weather": w} for s, w in sorted(picks, key=lambda p: (ids.index(p[0]), p[1]))]


def deal(args: argparse.Namespace) -> None:
    picks = dealt(read(args.screen))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(picks, indent=2) + "\n")
    print(json.dumps(picks, indent=2))
    if len(picks) < GATE:
        print(f"{len(picks)} tasks, below the gate of {GATE}", file=sys.stderr)
        raise SystemExit(1)


def chosen(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict[tuple[str, str], dict]]:
    weathers = read(args.weather)
    rows = {(r["schedule"], r["weather"]): r for r in read(args.screen)}
    ids = marine.task_ids()
    picks = sorted(json.loads(args.deal.read_text()), key=lambda p: (ids.index(p["schedule"]), p["weather"]))
    return weathers, picks, rows


def packed(v3: Task, weather: dict, row: dict) -> Task:
    return replace(wind.transplant(v3, weather), reference=v3.reference | {
        "optimal_cost": row["optimal_cost"], "optimal_plan": row["hindsight_plan"],
        "lower_bound": row["hindsight_cost"], "proven_optimal": True})


def data_file(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def file_sources(weather: Path) -> tuple[dict, dict]:
    """The sources file beside the weather, and its message pins referenced by the file's sha256 and their count."""
    path = weather.parent / "weather-sources.json"
    text = path.read_bytes()
    sources = json.loads(text)
    return sources, {"file": data_file(path), "sha256": hashlib.sha256(text).hexdigest(),
                     "count": len(sources["ecmwf"]["messages"])}


def manifest(tasks: list[Task], body: bytes, weather: Path, weathers: list[dict], picks: list[dict],
             rows: dict[tuple[str, str], dict], sources: dict, messages: dict, deal: dict) -> dict:
    """The sources' y7 and calibration blocks verbatim, and their ecmwf block but for its tens of thousands of message
    pins, which ``messages`` references."""
    by_id = {w["id"]: w for w in weathers}
    ecmwf = sources["ecmwf"]
    return {"name": packs.WIND_PACK, "split": "eval", "tasks": len(tasks),
            "tiers": dict(sorted(Counter(t.difficulty for t in tasks).items())),
            "sha256": hashlib.sha256(body).hexdigest(), "weeks": sorted({t.week for t in tasks}),
            "quays": dict(sorted(Counter(t.quay for t in tasks).items())),
            "source": (f"{packs.MARINE_PACK}'s qualified weeks, each moved into a weather week of "
                       f"{weather.name}: the wind observed at the anemometer and the forecasts as delivered, from "
                       "the sources under ecmwf, y7 and calibration"),
            "marine": {"pack": packs.MARINE_PACK, "sha256": marine.pack().manifest["sha256"]},
            "weather": {"file": data_file(weather),
                        "sha256": hashlib.sha256(weather.read_bytes()).hexdigest(), "records": len(weathers)},
            "pairs": [{"task_id": t.task_id, "schedule": p["schedule"], "weather": p["weather"],
                       "kind": by_id[p["weather"]]["kind"], "iso_week": by_id[p["weather"]]["iso_week"],
                       "added_watches": rows[p["schedule"], p["weather"]]["added_watches"],
                       "windows": by_id[p["weather"]]["windows"]} for t, p in zip(tasks, picks, strict=True)],
            "ecmwf": {key: value for key, value in ecmwf.items() if key not in ("messages", "fields")} | {
                "messages": messages},
            "y7": sources["y7"], "calibration": sources["calibration"],
            "wind": {"grounding": GROUNDING},
            "deal": deal,
            "note": (f"{packs.MARINE_PACK}'s weeks in the wind of another week: rules.no_moves holds the windows "
                     "that blew, which the grade uses, and the gale is gone; a forecast disruption at every watch "
                     "holds the run delivered by then (its 3-hourly knots and windows from the watch on) and the wind "
                     "observed before it, which the week as known shows instead. The reference cost is the lower of "
                     "the hindsight optimum, proven by CP-SAT, and the forecast-following re-planner's best net cost; "
                     "the reference plan is the hindsight optimum's. Each task id ends in its weather week's id.")}


def write_pack(out: Path, weather: Path, picks: list[dict], rows: dict[tuple[str, str], dict], sources: dict,
               messages: dict, deal: dict) -> Path:
    weathers = read(weather)
    by_id, v3 = {w["id"]: w for w in weathers}, marine.pack()
    tasks = [packed(v3.get(p["schedule"]), by_id[p["weather"]], rows[p["schedule"], p["weather"]]) for p in picks]
    body = "".join(json.dumps(t.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n" for t in tasks).encode()
    out = out / packs.WIND_PACK
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_bytes(body)
    (out / "manifest.json").write_text(json.dumps(
        manifest(tasks, body, weather, weathers, picks, rows, sources, messages, deal), indent=2,
        ensure_ascii=False) + "\n")
    print(f"Wrote {len(tasks)} tasks to {out}")
    return out


def pack(args: argparse.Namespace) -> None:
    _, picks, rows = chosen(args)
    deal = {"screened": len(rows), "qualifying": sum(r["qualifies"] for r in rows.values()), "dealt": len(picks),
            "gate": GATE}
    write_pack(args.out, args.weather, picks, rows, *file_sources(args.weather), deal)


def replayed(task: Task, plans: list[list[dict]]) -> dict:
    return outcome(played(task, lambda w: plan_from_list(plans[w.watch])).grade)


def write_references(out: Path, picks: list[dict], rows: dict[tuple[str, str], dict]) -> Path:
    tasks = TaskPack(out / packs.WIND_PACK).tasks
    lines = []
    for task, p in zip(tasks, picks, strict=True):
        row = rows[p["schedule"], p["weather"]]
        assert task.task_id == row["task_id"]
        res = marine.evaluate(task, plan_from_list(task.reference["optimal_plan"]))
        assert res.feasible and res.cost == row["hindsight_cost"], task.task_id
        for mode in ("rolling", "blind", "hold"):
            if "plans" in (stored := row.get(mode) or {}):
                assert replayed(task, stored["plans"]) == outcome(stored), (task.task_id, mode)
        assert outcome(played(task, world.naive).grade) == row["naive"], task.task_id
        lines.append(json.dumps({key: value for key, value in row.items() if key != "hindsight_plan"}) + "\n")
        print(f"{task.task_id}: replayed, qualifies {row['qualifies']}")
    path = out / "wind/references.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    print(f"Wrote {len(lines)} references to {path}")
    return path


def references(args: argparse.Namespace) -> None:
    _, picks, rows = chosen(args)
    write_references(args.out, picks, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("screen", help="play every pair of schedule and weather week")
    one.add_argument("--weather", type=Path, default=WEATHER)
    one.add_argument("--schedules", help="comma-separated marine task ids; default: every qualified one")
    one.add_argument("--out", type=Path, default=SCREEN)
    one.set_defaults(run=screen)
    one = commands.add_parser("deal", help="pick the pairs from the screen")
    one.add_argument("--screen", type=Path, default=SCREEN)
    one.add_argument("--out", type=Path, default=DEAL)
    one.set_defaults(run=deal)
    for name, run, text in (("pack", pack, "write the pack"), ("references", references, "replay and write the "
                                                                                           "references")):
        one = commands.add_parser(name, help=text)
        one.add_argument("--weather", type=Path, default=WEATHER)
        one.add_argument("--screen", type=Path, default=SCREEN)
        one.add_argument("--deal", type=Path, default=DEAL)
        one.add_argument("--out", type=Path, default=packs.tasks_root())
        one.set_defaults(run=run)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
