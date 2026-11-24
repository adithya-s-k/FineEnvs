"""Compute the marine pack, tasks/dock-v1-marine/, and tasks/marine/references.jsonl: the one-week dock-v1-eval weeks
with the port's pilots and tugs, each played live through port_week_core.marine by the rolling CP-SAT re-planner and the
naive online policy, as tools/live_references.py does for portsim-live.

The other traffic comes from the port's 2024 calls of every ship type (RAW, CC BY-SA 4.0), downloaded once at a pinned
commit and checked by its sha256. Each week's tug outage and pilot shortage start 24 hours after the bulletin whose
window holds the most of that pool's demand in the week's dock-v1-eval optimum, so they arrive on that bulletin. The
hindsight optimum is solved with everything known, to a proven optimum. The re-planner lets the pools cover the frozen
windows' own demand, since the grade excuses what news does to them. Both use live_references.solve, berth_core.solve's
optimal_plan model (Apache-2.0), with one cumulative constraint per pool.

Run from envs/port_week; --out defaults to packs.tasks_root(), and RAW is cached in ~/.cache/portsim-live/raw/.

    uv run -p 3.12 --with ortools==9.15.6755 --with-editable core python tools/marine_references.py [--raw PATH]
        [--tasks id,...] [--out DIR]
"""

import argparse
import csv
import hashlib
import io
import json
import os
import time
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import berth_core
import ortools
from berth_core import Plan, Task, plan_from_list, plan_to_list
from berth_core.check import unavoidable_cost
from berth_core.generate import hour
from berth_core.solve import naive_replan
from ortools.sat.python import cp_model
from port_week_core import marine, packs, schedule, world
from port_week_core.marine import CONTAINER, MarineWeek

from live_references import CONFIGS, DETERMINISTIC_SECONDS, outcome, solve

REPOSITORY = "https://github.com/alberto-santini/berth-allocation-problems"
COMMIT = "8e726a47625da99eec72f911b16a0b76a78a8d88"
RAW_FILE = "generator/raw/Barcelona_2024.csv"
RAW = f"https://raw.githubusercontent.com/alberto-santini/berth-allocation-problems/{COMMIT}/{RAW_FILE}"
RAW_SHA256 = "123012c078e66a5978d55bcedd4203ef85c4282ceefc0e3d6b01ebdc94207d94"
PILOTS = 7
TUGS = 8
HOURS = 264
ANCHORAGE = "90A"
PILOT_LENGTH_M = 45
NEWS = [("tug_outage", "tugs", 2, 24), ("pilot_shortage", "pilots", 2, 12)]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FAQ = ("https://contentv5.portdebarcelona.cat/cntmng/gd/d/workspace/SpacesStore/0f925a59-a5c8-4c7f-9b44-28c80f50f88a/"
       "FAQ_ATRACS_es.pdf")
PILOTAGE = "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2020-15568"
TOWAGE = "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2015-10895"
ORDINANCE = "https://www.boe.es/diario_boe/txt.php?id=BOE-A-2023-6719"
INFOPUERTOS = "https://infopuertos.com/la-fuerza-que-mueve-los-puertos-espanoles/"
HOUSTON = "https://www.houston-pilots.com/media/bmxni4u0/tug-matrix-july-2022.pdf"
GROUNDING = {
    "pilot": {"value": "every berthing and departure of a ship of 45 m or more", "basis": (
        f"Port of Barcelona, maritime operations FAQ, item 15 ({FAQ}): ships over 45 m and 500 GT take a pilot. RAW "
        "has no gross tonnage, so applying the length alone is an assumption.")},
    "service": {"value": "1 hour, in the hour the ship berths or leaves", "basis": (
        f"The pilotage service specification, BOE-A-2020-15568 ({PILOTAGE}), annex III, sizes the service on a mean "
        "of one hour per job. Counting it in the berthing and leaving hours, with the tugs for the same hour, is an "
        "assumption.")},
    "pilots": {"value": PILOTS, "basis": (
        "Assumption, calibrated: the smallest pool the port's 2024 traffic exceeds in under 1% of hours (see "
        f"calibration). Published: at least 18 pilots and never fewer than 3 on duty (BOE-A-2020-15568, prescription "
        f"16, {PILOTAGE}).")},
    "tugs": {"value": TUGS, "basis": (
        f"The port has \"8 or more\" tugs (InfoPuertos, {INFOPUERTOS}); each towage provider must have at least 5 "
        f"(BOE-A-2015-10895, prescription 14, {TOWAGE}). That all 8 are on duty at once is an assumption.")},
    "tugs_by_length": {"value": {"under 120 m": 0, "under 200 m": 1, "under 300 m": 2, "300 m or more": 3,
                                 "ferries, cruise ships and yachts": 0}, "basis": (
        "Assumption: the port publishes no table; towage is on request and the master decides with the pilot "
        f"(BOE-A-2015-10895, prescription 3, {TOWAGE}). The nearest public matrix is Houston Pilots' ({HOUSTON}).")},
    "wind": {"value": "one more tug per movement inside an announced window of wind above 25 kn, except Ro-Ro ships "
                      "and ferries under 200 m and yachts", "basis": (
        f"The port's traffic ordinance, BOE-A-2023-6719 ({ORDINANCE}), 4.1.2.2: at least one tug for merchant ships "
        "above 25 kn, and for Ro/Ro and Ro/Pax ships under 200 m from 30 kn. PortSimEnv applies the 25 kn windows "
        "only, so Ro-Ro ships and ferries under 200 m take no wind tug in the 30 kn hours either, hours in which none "
        "of our ships may move. That, the extra tug for ships that already take tugs, ferries as Ro/Pax, RAW's car "
        "carriers as merchant ships other than Ro/Ro, and yachts as non-merchant ships are assumptions.")},
    "gales": {"value": "the gales' no-movement windows hold our ships only; the other traffic keeps its 2024 hours",
              "basis": (
        f"BOE-A-2023-6719, 4.1.2.1 ({ORDINANCE}): the 25 and 30 kn thresholds start a review with the pilots, which "
        "PortSimEnv simplifies to no-movement windows. The gales are PortSimEnv's, not 2024's weather, so leaving "
        "the other traffic where it sailed is an assumption.")},
    "other_traffic": {"value": (
        "every 2024 call in RAW of a ship of 45 m or more that doesn't stop at the task's quay: a movement each time "
        "it berths, and one when it leaves a berth for the anchorage (90A) or the sea, so a shift between berths "
        "counts once; plus the departures of the ships alongside at hour 0, at their RAW length"), "basis": (
        "RAW. Its ETA and ETD are berthing and leaving times for container calls (PortSimEnv's data/barcelona/"
        "SOURCE.md); for other ship types that, and 90A as the anchorage rather than a berth, are assumptions. RAW "
        "lists a stop served by two terminal operators once per operator; it counts once. Every call that stops at "
        "the task's quay is left out over all 264 hours, with its stops at other berths, since the quay holds only "
        "the week's own ships; leaving out the quay's other calls, 10 to 39 movements a week, most of them the next "
        "week's after hour 168, is an assumption.")},
    "hours": {"value": HOURS, "basis": (
        "The other traffic is counted from hour 0 to hour 264 (11 days), past every ship's planned departure; after "
        "it every pilot and tug is free.")},
    "tug_outage": {"value": {"tugs": 2, "hours": 24, "notice_hours": 24}, "basis": (
        f"A tug may leave service for repairs or a call-away (BOE-A-2015-10895, prescription 2, {TOWAGE}), and a "
        "provider may be one tug short for up to 240 hours a year (annex 1). Its size, length and timing are "
        "assumptions.")},
    "pilot_shortage": {"value": {"pilots": 2, "hours": 12, "notice_hours": 24}, "basis": "Assumption."},
}


def cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "portsim-live" / "raw" / COMMIT[:7]


def raw(path: Path | None) -> bytes:
    """RAW from ``path``, else from the cache, downloading it there the first time."""
    cached = cache_dir() / Path(RAW_FILE).name
    path = path or (cached if cached.is_file() else None)
    if path:
        body = path.read_bytes()
    else:
        with urllib.request.urlopen(RAW, timeout=120) as response:
            body = response.read()
    if hashlib.sha256(body).hexdigest() != RAW_SHA256:
        raise SystemExit(f"{path or RAW} is not the file pinned at {RAW} (sha256 mismatch)")
    if not path:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(body)
    return body


def utc(text: str, form: str = TIME_FORMAT) -> datetime:
    return datetime.strptime(text, form).replace(tzinfo=UTC)


def length(row: dict) -> float:
    return float(row["ESLORA_METRES"].replace(",", "."))


def calls(body: bytes) -> list[list[dict]]:
    """RAW's stops by call. A stop served by two terminal operators has a row for each; it is one berthing."""
    by: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for row in csv.DictReader(io.StringIO(body.decode())):
        stop = (row["ESCALANUM"], row["MOLLCODI"], row["ETAUTC"], row["ETDUTC"])
        by[row["ESCALANUM"].split("-")[0]].setdefault(stop, row)
    return [sorted(stops.values(), key=lambda r: utc(r["ETAUTC"])) for stops in by.values()]


def moves(stops: list[dict], start: datetime) -> list[list]:
    """[hour, type, length_m] for each time the call's ship berths, and each time it leaves a berth for the anchorage
    or the sea."""
    out = []
    for i, r in enumerate(stops):
        if r["MOLLCODI"] != ANCHORAGE:
            out.append([hour(start, utc(r["ETAUTC"])), r["VAIXELLTIPUS"], length(r)])
            if i + 1 == len(stops) or stops[i + 1]["MOLLCODI"] == ANCHORAGE:
                out.append([hour(start, utc(r["ETDUTC"])), r["VAIXELLTIPUS"], length(r)])
    return out


def movements(task: Task, raw_calls: list[list[dict]]) -> list[list]:
    """The other traffic: every call that doesn't stop at the task's quay, and the ships alongside at hour 0."""
    start = utc(task.week_start_utc, schedule.TIME_FORMAT)
    out = [m for stops in raw_calls if all(r["MOLLCODI"] != task.quay for r in stops) for m in moves(stops, start)]
    for b in task.blocks:
        if b.kind == "alongside":
            [row] = [r for stops in raw_calls for r in stops
                     if r["MOLLCODI"] == task.quay and r["VAIXELLNOM"] == b.label
                     and utc(r["ETAUTC"]) < start <= utc(r["ETDUTC"])]
            out.append([b.end, CONTAINER, length(row)])
    return sorted(m for m in out if 0 <= m[0] < HOURS and m[2] >= PILOT_LENGTH_M)


def news(task: Task) -> list[dict]:
    """The tug outage and the pilot shortage, each over the most demand for its pool in the dock-v1-eval optimum."""
    demand = {"pilots": Counter(), "tugs": Counter()}
    optimum = berth_core.evaluate(task, plan_from_list(task.reference["optimal_plan"]))
    for _, hour, tugs in marine.own_moves(task, optimum):
        demand["pilots"][hour] += 1
        demand["tugs"][hour] += tugs
    bulletins = [w.hour for w in schedule.schedule(task)][1:]
    out = []
    for kind, pool, count, hours in NEWS:
        lead = schedule.LEADS[kind]
        b = max(bulletins, key=lambda b: (sum(demand[pool][h] for h in range(b + lead, b + lead + hours)), -b))
        out.append({"type": kind, "start": b + lead, "end": b + lead + hours, pool: count})
    return out


def notice(e: dict) -> str:
    if e["type"] == "tug_outage":
        return (f"{e['tugs']} of the port's {TUGS} tugs are out of service from hour {e['start']} to hour {e['end']}: "
                f"{TUGS - e['tugs']} tugs in that window.")
    return (f"{e['pilots']} of the port's {PILOTS} pilots on duty are unavailable from hour {e['start']} to hour "
            f"{e['end']}: {PILOTS - e['pilots']} pilots in that window.")


def marine_task(task: Task, raw_calls: list[list[dict]]) -> Task:
    cuts = news(task)
    notices = [n + f" Ships that berth or leave from hour {e['start']} to hour {e['end']} take one more tug."
               if e["type"] == "gale" else n for n, e in zip(task.notices, task.disruptions, strict=True)]
    return replace(task, rules=task.rules | {"marine": {"pilots": PILOTS, "tugs": TUGS, "hours": HOURS,
                                                        "movements": movements(task, raw_calls)}},
                   disruptions=task.disruptions + cuts, notices=notices + [notice(e) for e in cuts])


def _inside(md: cp_model.CpModel, x, a: int, b: int):
    z, below, above = md.NewBoolVar(""), md.NewBoolVar(""), md.NewBoolVar("")
    md.Add(x >= a).OnlyEnforceIf(z)
    md.Add(x <= b - 1).OnlyEnforceIf(z)
    md.Add(x <= a - 1).OnlyEnforceIf(below)
    md.Add(x >= b).OnlyEnforceIf(above)
    md.AddBoolOr([z, below, above])
    md.AddImplication(z, below.Not())
    md.AddImplication(z, above.Not())
    return z


def constraints(view: Task, fixed: Plan) -> Callable[[cp_model.CpModel, dict], None]:
    """Each pool as a cumulative over our ships' movements, with the hours it isn't free blocked; in an hour where the
    frozen windows' own movements need more than is free, they get what they need."""
    pools = view.rules["marine"]
    room, windows = marine.free(view), marine.wind(view)
    frozen = {"pilots": Counter(), "tugs": Counter()}
    for _, hour, tugs in marine.own_moves(view, berth_core.evaluate(view, fixed)):
        frozen["pilots"][hour] += 1
        frozen["tugs"][hour] += tugs

    def extra(md: cp_model.CpModel, ships: dict) -> None:
        for pool in ("pilots", "tugs"):
            intervals, demands = [], []
            for sid, hours in ships.items():
                need = 1 if pool == "pilots" else marine.tugs_needed(CONTAINER, view.ships[sid].length_m, False)
                for x in hours:
                    if need:
                        intervals.append(md.NewFixedSizeIntervalVar(x, 1, ""))
                        demands.append(need)
                    if pool == "tugs":
                        for a, b in windows:
                            intervals.append(md.NewOptionalFixedSizeIntervalVar(x, 1, _inside(md, x, a, b), ""))
                            demands.append(1)
            for h in range(pools["hours"]):
                if (blocked := pools[pool] - max(room[pool][h], frozen[pool][h])) > 0:
                    intervals.append(md.NewFixedSizeIntervalVar(h, 1, ""))
                    demands.append(blocked)
            md.AddCumulative(intervals, demands, pools[pool])
    return extra


def rolling(workers: int, tiebreak: bool, plans: list[list[dict]], seconds: list[float]) -> world.Policy:
    def policy(week: world.Week) -> Plan:
        view = week.view
        started = time.monotonic()
        plan = solve(view, week.fixed, week.before, week.plan if week.watch else naive_replan(view), workers, tiebreak,
                     extra=constraints(view, week.fixed))
        seconds.append(time.monotonic() - started)
        plans.append(plan_to_list(plan))
        return plan
    return policy


def played(task: Task, policy: world.Policy) -> dict:
    week = world.play(task, policy, week=MarineWeek)
    assert week.refusals == []
    return week.grade


def hindsight(task: Task) -> tuple[int, Plan]:
    """On one worker: proven optima on eight differ from run to run where the week has more than one."""
    plan = solve(task, {}, 0, plan_from_list(task.reference["optimal_plan"]), 1, False, extra=constraints(task, {}),
                 optimal=True)
    res = marine.evaluate(task, plan)
    assert res.feasible and res.cost >= task.reference["optimal_cost"], (task.task_id, res.cost)
    return res.cost, plan


def reference(task: Task, v2_optimal_cost: int, seconds: list[float]) -> dict:
    runs = []
    for workers, tiebreak in CONFIGS:
        plans = []
        runs.append((played(task, rolling(workers, tiebreak, plans, seconds)), plans))
    (grade, plans), costs = runs[0], [g["cost"] for g, _ in runs]
    return {"task_id": task.task_id, "watch_hours": [w.hour for w in schedule.schedule(task)],
            "optimal_cost": grade["optimal_cost"], "v2_optimal_cost": v2_optimal_cost,
            "unavoidable_cost": unavoidable_cost(task), "rolling": {**outcome(grade), "plans": plans},
            "configs": [{"workers": w, "tiebreak": tb, "cost": c} for (w, tb), c in zip(CONFIGS, costs, strict=True)],
            "qualifies": sum(c is not None and c <= grade["optimal_cost"] for c in costs) >= 3,
            "naive": outcome(played(task, world.naive)),
            "solver": {"ortools": ortools.__version__, "max_deterministic_time": DETERMINISTIC_SECONDS}}


def calibration(raw_calls: list[list[dict]]) -> dict:
    """The whole port's 2024 traffic, no quay left out and no wind, against the pools."""
    start, hours = datetime(2024, 1, 1, tzinfo=UTC), 366 * 24
    use = {"pilots": Counter(), "tugs": Counter()}
    for stops in raw_calls:
        for hour, kind, length_m in moves(stops, start):
            if 0 <= hour < hours and length_m >= PILOT_LENGTH_M:
                use["pilots"][hour] += 1
                use["tugs"][hour] += marine.tugs_needed(kind, length_m, False)
    return {"hours": hours, "movements": sum(use["pilots"].values()),
            "over_pool_percent": {pool: round(100 * sum(n > size for n in use[pool].values()) / hours, 2)
                                  for pool, size in (("pilots", PILOTS), ("tugs", TUGS))}}


def manifest(weeks: list[Task], body: bytes, raw_calls: list[list[dict]]) -> dict:
    return {"name": packs.MARINE_PACK, "split": "eval", "tasks": len(weeks),
            "tiers": dict(sorted(Counter(t.difficulty for t in weeks).items())),
            "sha256": hashlib.sha256(body).hexdigest(), "weeks": sorted({t.week for t in weeks}),
            "source": ("PortSimEnv's dock-v1-eval one-week weeks, with the port's other 2024 traffic from RAW; both "
                       "from the Port de Barcelona open data portal (CC BY-SA 4.0)"),
            "quays": dict(sorted(Counter(t.quay for t in weeks).items())),
            "raw": {"repository": REPOSITORY, "file": RAW_FILE, "commit": COMMIT, "sha256": RAW_SHA256,
                    "licence": "CC BY-SA 4.0"},
            "marine": GROUNDING, "calibration": calibration(raw_calls),
            "note": ("dock-v1-eval's one-week weeks with the port's pilots and tugs: rules.marine holds the pools and "
                     "the other traffic's movements as [hour, type, length_m]; a tug outage and a pilot shortage "
                     "follow the week's disruptions and notices, and each gale notice says that wind takes one more "
                     "tug; the reference is the hindsight optimum under those rules, proven optimal by CP-SAT. The "
                     "task ids are dock-v1-eval's, so never load the two packs together.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw", type=Path, help="RAW, checked by its sha256; default: downloaded once into the cache")
    parser.add_argument("--tasks", help="comma-separated task ids; default: every one-week dock-v1-eval task")
    parser.add_argument("--out", type=Path, default=packs.tasks_root())
    args = parser.parse_args()
    raw_calls = calls(raw(args.raw))
    chosen = [t for t in packs.live_pack().tasks if t.split == "eval" and schedule.ONE_WEEK.match(t.task_id)
              and (args.tasks is None or t.task_id in args.tasks.split(","))]
    weeks, lines = [], []
    for task in chosen:
        started, seconds = time.monotonic(), []
        week = marine_task(task, raw_calls)
        cost, plan = hindsight(week)
        solved = time.monotonic() - started
        week = replace(week, reference=task.reference | {"optimal_cost": cost, "optimal_plan": plan_to_list(plan),
                                                         "lower_bound": cost, "proven_optimal": True})
        line = reference(week, task.reference["optimal_cost"], seconds)
        weeks.append(week)
        lines.append(json.dumps(line) + "\n")
        print(f"{task.task_id}: {len(line['watch_hours'])} watches, news from hours "
              f"{[e['start'] for e in week.disruptions[-2:]]}, optimum {cost} (v2 {line['v2_optimal_cost']}, "
              f"{solved:.1f}s), rolling "
              f"{[c['cost'] for c in line['configs']]}, naive {line['naive']['cost']}, qualifies {line['qualifies']}, "
              f"slowest re-solve {max(seconds):.1f}s, {time.monotonic() - started:.1f}s", flush=True)
    body = "".join(json.dumps(t.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n" for t in weeks).encode()
    pack = args.out / packs.MARINE_PACK
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "tasks.jsonl").write_bytes(body)
    (pack / "manifest.json").write_text(json.dumps(manifest(weeks, body, raw_calls), indent=2) + "\n")
    (args.out / "marine").mkdir(parents=True, exist_ok=True)
    (args.out / "marine/references.jsonl").write_text("".join(lines))
    print(f"Wrote {len(weeks)} weeks to {pack} and {args.out / 'marine/references.jsonl'}")


if __name__ == "__main__":
    main()
