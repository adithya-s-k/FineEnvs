"""The storm day with crews: can they fly JetBlue's real day, what do they add to the calm day, and how close is the
best plan over aircraft and crews to a proven bound? Its guards were set before the run. Only the crew and aircraft
solvers run (HiGHS, one thread each); no LLM is called.

    python tools/crew_check.py --out cache/<run> --tasks cache/<world_check run> [--only DATE|TASK_ID,...]
                                 [--bands middle,low,high] [--workers 4] [--resume] [--scale 1.0]
    python tools/crew_check.py --desk-replay cache/<crew_check run> --out cache/<new run> --tasks ... [--workers 4]
    python tools/crew_check.py --ends-over cache/<crew_check run> --out cache/<new run> --tasks ... [--workers 4]
                                 [--resume]

Per task of a world_check run (its storm dates and its controls, at its fitted scale), in its own process:
1. the trips over the published schedule (pairings.py), once;
2. per reserve band (the share of crews on reserve: pilots 13/16/20%, flight attendants 16/19/22%), the middle first:
   - the units at the start, from a crew replay of the record before it (crewday.storm_units);
   - the crew replay of JetBlue's real day: its flown flights at their recorded departures and blocks, crewed by those
     units, their reserves and every move, the fewest flights without a legal crew first (crewday.replay_record), the
     plan audited move by move against the crew sim's rules;
   - the crew-aware calm day (crewplan.calm_day) against the world_check run's calm optimum without crews, at the
     middle band with a lower bound on it (the least rise any crew-aware calm day could have);
3. at the middle band (crewplan.joint), notice-priced, and flat-priced on eval dates and controls (elsewhere the flat
   best is the best plan's flat cost): the aircraft optimum A*, the restricted A*, the best plan found by the search
   within its budget, replayed in the crew sim, and the lower bound (crewbound.py); the standard crew desk, the
   baseline (naive's empty plan for the aircraft, its crews by the crew sim's desk: crewsim.py), among the search's
   candidates; naive without crews; the airline replay with crews by the crew sim's trip rule; JetBlue's recorded day
   with the crew replay's crew cost.
Eval dates and controls get the larger budgets (BUDGETS). Writes dates/<date>.json and plans/<date>.json as each task
finishes (`--resume` keeps those written), then guards.json and report.md.

With `--desk-replay RUN` only the standard crew desk runs again, over RUN's dates: per date the trips and the units at
the start as RUN computed them (each date says whether the morning replay reproduced RUN's), then the desk's day; every
other result is RUN's, best plans and bounds included. It writes each date with the desk's numbers in place, then
guards.json and report.md, the controls and leverage read against the new desk, both marked as the desk replay over RUN
at this commit.

With `--ends-over RUN` only the reserve-band ends (low and high) run, over RUN's dates (RUN having run the middle band):
per date the trips and the calm optimum without crews, then for each end what the full run does for it (the units at
the start, the record replay and the calm day, on the end budgets), measured against the calm base RUN used. Each date
is RUN's row with the two ends added, its plans copied; guards.json and report.md then read all three bands, marked as
the band ends over RUN at this commit, the middle band and the crew desk being RUN's.

The guards, as registered (a storm date being a task that is not a control; the middle band):
- record: flights without a legal crew in the crew replay <= 1% of the date's flown flights and <= 2, on >= 90% of
  storm dates;
- calm: the crew-aware calm day rises over the calm optimum without crews by <= 2% (from 0, stays 0), on >= 90% of
  storm dates;
- best: on every kept date the best plan replays exactly with a legal crew on every flight it flies, and the bound is
  valid (no day played on the date costs less); gap = (best - bound) / best per date, counted within 0.01%, 1% and 5%,
  no threshold; a date whose search or bound fails to run is left out and counted;
- controls: on both control dates, the recorded day (with crews) costs at least the best plan, and the standard crew
  desk is within 3% of the flat best plan.
Reported, not gated: the band ends, leverage with crews, the standard crew desk over the best plan (a lower bound on the
true leverage; flat and notice-priced), the desk's cancellations and auto-cancellations and its cost over naive without
crews, the crew-caused cancellations and delays (the best plan against A*), the crew cost by kind, the gap distribution,
the trips (legs left without one, the builder's rounds and gap), the cabin duty lengths of the best plan, the calm plan
and the crew replay (121.467(b)(14) lets a team fly its own trip's late legs past 14 h), solve times and peak memory per
task.
"""

import argparse
import json
import math
import resource
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from airops_core.core.sim import AUTO_CANCELLED, CANCELLED, FLOWN
from airops_core.programs import crewbound, crewday, crewplan, crews, crewsim, references, sim
from airops_core.programs.model import Task

ROOT = Path(__file__).resolve().parents[1]
INPUTS, DERIVED = ROOT / "cache" / "inputs", ROOT / "cache" / "derived"
SHARE, MOST = 0.01, 2
RISE = 2.0
GATE_SHARE = 0.9
CONTROL_MARGIN = 0.03
CONTROL_DATES = 2
LEVERAGE = 0.10
GAPS = (0.01, 1.0, 5.0)
RSS_UNIT = 1 << 30 if sys.platform == "darwin" else 1 << 20
BUDGETS = {
    "regular": {"morning": 240, "record": 360, "calm": 240, "calm_cover": 180, "search": 1200, "cover": 600,
                "end_morning": 120, "end_record": 180, "end_calm": 120, "polish": 60},
    "extra": {"morning": 480, "record": 720, "calm": 600, "calm_cover": 600, "search": 2000, "flat": 900,
              "cover": 1200, "end_morning": 240, "end_record": 360, "end_calm": 240, "polish": 180},
}
WIDTH = {"regular": 60, "extra": 30}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, required=True, help="a world_check run's out directory")
    ap.add_argument("--only", default="", help="comma-separated dates or task ids (default: all)")
    ap.add_argument("--bands", default="middle,low,high")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every time budget")
    ap.add_argument("--desk-replay", type=Path, help="an earlier run's out directory: only the crew desk runs again")
    ap.add_argument("--ends-over", type=Path, help="an earlier middle-band run's out directory: only the band ends run")
    a = ap.parse_args()
    bands = tuple(a.bands.split(","))
    if "middle" not in bands or set(bands) - set(crews.BANDS):
        sys.exit(f"ERROR: --bands takes {', '.join(crews.BANDS)}, the middle among them")
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    for sub in ("dates", "plans"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    lines = {}
    for line in (a.tasks / "tasks.jsonl").open():
        head = json.loads(line[:line.index(',"tz"')] + "}")
        if not only or head["date"] in only or head["task_id"] in only:
            lines[head["date"]] = line
    if a.desk_replay:
        desk_replay(a.desk_replay, a.out, lines, a.workers)
        return
    if a.ends_over:
        ends_over(a.ends_over, a.out, lines, a.workers, a.resume)
        return
    refs = {}
    for p in (a.tasks / "refs").glob("*.json"):
        r = json.loads(p.read_text())
        if r["date"] in lines:
            refs[r["date"]] = {k: r.get(k) for k in ("anchor", "flat", "calm", "recorded", "naive")}
    todo = [d for d in sorted(lines) if not (a.resume and (a.out / "dates" / f"{d}.json").exists())]
    todo.sort(key=lambda d: -len(lines[d]))
    t0 = time.monotonic()
    if todo:
        with ProcessPoolExecutor(min(a.workers, len(todo)), max_tasks_per_child=1) as pool:
            futures = {pool.submit(job, lines[d], bands, refs.get(d, {}), a.out, a.scale): d for d in todo}
            for n, f in enumerate(as_completed(futures), 1):
                print(f"[{time.monotonic() - t0:8.0f}s] {n}/{len(todo)} {futures[f]} {f.result()}", flush=True)
    rows = [json.loads((a.out / "dates" / f"{d}.json").read_text()) for d in sorted(lines)
            if (a.out / "dates" / f"{d}.json").exists()]
    report = guards(rows, bands)
    (a.out / "guards.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    (a.out / "report.md").write_text(markdown(report, rows, bands))
    print(json.dumps(report["guards"], indent=1, default=str), flush=True)


def job(line: str, bands: tuple[str, ...], refs: dict, out: Path, scale: float) -> str:
    t0 = time.monotonic()
    task = Task.from_dict(json.loads(line))
    plans: dict = {}
    try:
        row = check(task, bands, refs, scale, plans)
    except Exception:
        row = {"date": task.date, "task_id": task.task_id, "split": task.split, "error": traceback.format_exc()}
    row["seconds"] = round(time.monotonic() - t0, 1)
    row["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_UNIT, 2)
    (out / "plans" / f"{task.date}.json").write_text(json.dumps(plans, default=str))
    (out / "dates" / f"{task.date}.json").write_text(json.dumps(row, indent=1, default=str))
    return "error" if "error" in row else f"{row['seconds']}s {row['peak_rss_gb']}GB"


def check(task: Task, bands: tuple[str, ...], refs: dict, scale: float, plans: dict) -> dict:
    kind = "extra" if task.split in ("eval", "control") else "regular"
    budget = {k: v * scale for k, v in BUDGETS[kind].items()} | {"width": WIDTH[kind]}
    laps: dict[str, float] = {}
    clock = Laps(laps)
    w = crewday.world(task, INPUTS, DERIVED)
    clock("trips")
    row = {"date": task.date, "task_id": task.task_id, "split": task.split, "start": task.start,
           "flights": len(task.flights), "budget": budget, "trips": trips(w), "laps": laps, "bands": {}}
    optimum = crewplan.calm_optimum(w)
    clock("calm_optimum")
    reference = (refs.get("calm") or {}).get("objective")
    row["calm_optimum"] = {"objective": optimum.objective, "status": optimum.status, "gap_pct": optimum.gap_pct,
                           "reference": reference}
    base = optimum.objective if reference is None else reference
    for band in bands:
        if band != "middle":
            row["bands"][band] = end_band(w, band, optimum, base, budget, clock)
            continue
        units, before = crewday.storm_units(w, band, budget["morning"], budget["polish"])
        clock(f"{band}_morning")
        rec, rec_plan = record(w, units, budget["record"], budget["polish"])
        clock(f"{band}_record")
        rec["before_start"] = before
        b = {"record": rec, "joint": joint(w, units, refs, budget, rec, plans, clock)}
        plans["record"] = rec_plan
        b["calm"] = calm(w, band, optimum, base, budget["calm"], budget["calm_cover"], budget["width"], plans)
        clock(f"{band}_calm")
        row["bands"][band] = b
    return row


def end_band(w: crewday.World, band: str, optimum, base: float, budget: dict, clock) -> dict:
    """A reserve band's end: the units at the start, the record replay and the calm day, on the end budgets."""
    units, before = crewday.storm_units(w, band, budget["end_morning"], budget["polish"])
    clock(f"{band}_morning")
    rec, _ = record(w, units, budget["end_record"], budget["polish"])
    clock(f"{band}_record")
    rec["before_start"] = before
    b = {"record": rec, "calm": calm(w, band, optimum, base, budget["end_calm"], 0.0, budget["width"], None)}
    clock(f"{band}_calm")
    return b


class Laps:
    def __init__(self, laps: dict):
        self.laps, self.last = laps, time.monotonic()

    def __call__(self, name: str) -> None:
        now = time.monotonic()
        self.laps[name] = round(now - self.last, 1)
        self.last = now


def trips(w: crewday.World) -> dict:
    by = {g.key: g for g in w.legs}
    return {f"{k}/{g}": {"duties": len(b.duties), "uncovered": len(b.uncovered),
                         "uncovered_legs": [f"{by[x].flight} {by[x].orig}-{by[x].dest} at {by[x].sd}"
                                            for x in b.uncovered],
                         "gap_pct": round(b.gap_pct, 3), "converged": b.converged, "columns": b.columns,
                         "rounds": b.rounds, "seconds": b.seconds,
                         "legs_per_duty": round(sum(len(d.legs) for d in b.duties) / max(1, len(b.duties)), 2),
                         "outstation_starts": sum(d.start not in crews.bases(k, w.task.date) for d in b.duties)}
            for (k, g), b in w.built.items()}


def record(w: crewday.World, units, limit: float, polish: float) -> tuple[dict, dict]:
    task = w.task
    p, sol = crewday.replay_record(w, units, limit, polish)
    deps, blocks, ends = crewday.recorded(task)
    plan = crewday.plan_of(p, sol)
    audit = crewsim.audit(task, units, plan, deps, blocks, w.local, w.airports, ends)
    flown = len(deps)
    n = len(sol.uncrewed)
    return {"flown": flown, "uncrewed": n, "pass": replays(n, flown), "solve": sol.summary(),
            "causes": crewday.uncrewed_causes(p, sol), "crew_cost": round(sol.crew_cost, 1), "audit": audit,
            "units": units_at(units), "cabin_duty_h": duty_hours(column_duties(p, sol))}, plan


def replays(uncrewed: int, flown: int) -> bool:
    """The record guard's reading of one day: flights without a legal crew at most SHARE of those flown and at most
    MOST."""
    return uncrewed <= SHARE * flown and uncrewed <= MOST


def calm(w: crewday.World, band: str, optimum, base: float, budget: float, cover: float, width: int,
         plans: dict | None) -> dict:
    """The crew-aware calm day against the calm optimum, and with `cover` a lower bound on it: the least rise any
    crew-aware calm day could have."""
    units = crewday.calm_units(w, band)
    c = crewplan.calm_day(w, units, optimum, budget, crew_limit=max(60.0, budget / 3), cover_limit=cover, width=width)
    day, f = c.found.day, c.found
    deps = {i: r.dep for i, r in day.run.flights.items() if r.dep is not None}
    audit = crewsim.audit(c.task, units, f.plan, deps, {x.id: x.sched_block for x in c.task.flights}, w.local,
                          w.airports)
    if plans is not None:
        plans["calm"] = f.plan
    return {"optimum": optimum.objective, "base": base, "crew_aware": round(day.total, 1),
            "rise_pct": rise(day.total, base), "pass": calm_pass(day.total, base), "exact": f.exact,
            "claimed": round(f.claimed, 1), "crew_cost": {k: round(v, 1) for k, v in day.cost.items()},
            "aircraft": day.run.cost.to_dict(), "cancelled": sum(r.dep is None for r in day.run.flights.values()),
            "uncrewed": len(day.uncrewed), "attempts": f.attempts, "audit": audit, "units": units_at(units),
            "cabin_duty_h": duty_hours(day_duties(day)),
            "pending_crews": sum(u.id.endswith("p") for u in units),
            "bound": None if c.bound is None else round(c.bound, 1),
            "rise_lower_pct": None if c.bound is None else rise(max(c.bound, base), base),
            "pieces": {k: None if v is None else round(v, 1) for k, v in c.pieces.items()}}


def rise(value: float, base: float) -> float | None:
    if abs(value - base) < 1e-6:
        return 0.0
    return None if base <= 0 else round(100 * (value - base) / base, 4)


def calm_pass(value: float, base: float) -> bool:
    r = rise(value, base)
    return r is not None and r <= RISE


def joint(w: crewday.World, units, refs: dict, budget: dict, rec: dict, plans: dict, clock) -> dict:
    """The best plan and its bound, and the days played with crews; on eval dates and controls also the flat best plan
    and bound (elsewhere the flat best is the best plan's flat cost)."""
    task = w.task
    world = sim.truth(task)
    air = crewplan.Aircraft(task, world)
    fl = crewbound.Flights.of(task, world.block, air.net.copies)
    reached = crewbound.reach(task, fl, units, w.local)
    least = crewbound.least(task, fl, units, w.local, crews.Ends(units, w.airports, task.date).least, reached)
    clock("aircraft_reach_least")
    out = {"least": {"total": round(float(least.sum()), 1), "units": int(np.count_nonzero(least > 1e-9))}}
    played, desk = plays(task, units, world, w)
    clock("plays")
    extra = task.split in ("eval", "control")
    as_flat = None
    for flat in (False, True) if extra else (False,):
        name = "flat" if flat else "notice"
        j = crewplan.joint(task, world, units, w.local, w.airports, air, fl, reached, least,
                           budget["flat" if flat else "search"], budget["cover"], flat=flat,
                           crew_limit=max(60.0, budget["flat" if flat else "search"] / 4),
                           days=(("standard crew desk", desk), ("airline replay", references.airline_plan(task))),
                           width=budget["width"])
        clock(f"{name}_joint")
        plans[name] = j.found.plan
        replay = crewsim.execute(task, j.found.plan, units, world, w.local, fill=False, airports=w.airports)
        others = {k: v[name] for k, v in played.items() if k != "naive"}
        others |= {f"try {n}: {t['try']}": t["replayed"] for n, t in enumerate(j.found.attempts)}
        out[name] = summary(task, j, crewplan.worth(replay, flat), others, refs.get("flat" if flat else "anchor") or {})
        if not flat:
            as_flat = crewplan.worth(replay, True)
    if "flat" not in out:
        out["flat"] = {"best": round(as_flat, 1), "from": "notice plan", "bound": None, "gap_pct": None}
        plans["flat"] = plans["notice"]
    elif as_flat < out["flat"]["best"]:
        f = out["flat"]
        f |= {"best": round(as_flat, 1), "from": "notice plan", "below_bound": {
            **f["below_bound"], **({"notice plan": as_flat} if as_flat < f["bound"] - 0.5 else {})},
            "gap_pct": round(100 * (as_flat - f["bound"]) / as_flat, 4)}
        plans["flat"] = plans["notice"]
    out["played"] = played
    recorded, ferries = references.recorded(task)
    out["recorded"] = {"aircraft": recorded.to_dict(), "crew_cost": rec["crew_cost"],
                       "total": round(recorded.total + rec["crew_cost"], 1), "uncrewed": rec["uncrewed"],
                       "ferries": ferries}
    del air
    return out


def plays(task: Task, units, world: sim.World, w: crewday.World) -> tuple[dict, dict]:
    """The standard crew desk (the empty plan, its crews by the crew sim's desk) and the airline replay with crews (by
    the trip rule), naive without crews; and the desk's day as a plan naming what it did."""
    desk = crewsim.execute(task, {}, units, world, w.local, airports=w.airports, desk=True)
    airline = crewsim.execute(task, references.airline_plan(task), units, world, w.local, airports=w.airports)
    out = {"crew_desk": outcome(desk), "airline": outcome(airline)}
    naive = sim.execute(task, {}, world)
    out["naive"] = {"notice": naive.cost.total, "flat": naive.cost.flat, "aircraft": naive.cost.to_dict(),
                    "statuses": dict(Counter(r.status for r in naive.flights.values()))}
    return out, crewday.named(desk)


def ends_over(run: Path, out: Path, lines: dict[str, str], workers: int, resume: bool) -> None:
    """The band ends over `run`'s dates (see the module doc), then the guards and the report over all three bands."""
    dates = sorted(d for d in lines if (run / "dates" / f"{d}.json").exists())
    todo = [d for d in dates if not (resume and (out / "dates" / f"{d}.json").exists())]
    t0 = time.monotonic()
    if todo:
        with ProcessPoolExecutor(min(workers, len(todo)), max_tasks_per_child=1) as pool:
            futures = {pool.submit(ends, lines[d], run, out): d for d in sorted(todo, key=lambda d: -len(lines[d]))}
            for n, f in enumerate(as_completed(futures), 1):
                print(f"[{time.monotonic() - t0:8.0f}s] {n}/{len(todo)} {futures[f]} {f.result()}", flush=True)
    rows = [json.loads((out / "dates" / f"{d}.json").read_text()) for d in dates]
    bands = ("middle", "low", "high")
    report = guards(rows, bands)
    was = run / "guards.json"
    if was.exists() and "desk_replay" in (earlier := json.loads(was.read_text())):
        report["desk_replay"] = earlier["desk_replay"]
    report["ends_over"] = {"over": str(run), "commit": commit(), "dates": len(rows),
                           "failed": sorted(r["date"] for r in rows if "ends_error" in r),
                           "seconds": round(time.monotonic() - t0, 1)}
    (out / "guards.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    mark = report["ends_over"]
    (out / "report.md").write_text(f"The reserve-band ends (low, high) run over {mark['over']} at commit "
                                   f"{mark['commit']}; the middle band, the crew desk and every other number are that "
                                   "run's.\n\n" + markdown(report, rows, bands))
    print(json.dumps({"ends_over": mark, **report["guards"]}, indent=1, default=str), flush=True)


def ends(line: str, run: Path, out: Path) -> str:
    """One date of the band ends: its row from `run` with the low and high bands added, its plans copied."""
    t0 = time.monotonic()
    task = Task.from_dict(json.loads(line))
    row = json.loads((run / "dates" / f"{task.date}.json").read_text())
    plans = run / "plans" / f"{task.date}.json"
    if plans.exists():
        (out / "plans").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(plans, out / "plans" / plans.name)
    if "error" not in row:
        laps: dict[str, float] = {}
        try:
            clock = Laps(laps)
            w = crewday.world(task, INPUTS, DERIVED)
            clock("trips")
            optimum = crewplan.calm_optimum(w)
            clock("calm_optimum")
            for band in ("low", "high"):
                row["bands"][band] = end_band(w, band, optimum, row["bands"]["middle"]["calm"]["base"], row["budget"],
                                              clock)
        except Exception:
            row["ends_error"] = traceback.format_exc()
        row["ends_over"] = {"seconds": round(time.monotonic() - t0, 1), "laps": laps,
                            "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_UNIT, 2)}
    (out / "dates").mkdir(parents=True, exist_ok=True)
    (out / "dates" / f"{task.date}.json").write_text(json.dumps(row, indent=1, default=str))
    if "error" in row:
        return "error in the run"
    return "error" if "ends_error" in row else f"{row['ends_over']['seconds']}s {row['ends_over']['peak_rss_gb']}GB"


def outcome(d: crewsim.CrewDay) -> dict:
    return {"notice": round(d.total, 1), "flat": round(crewplan.worth(d, True), 1), "aircraft": d.run.cost.to_dict(),
            "crew_cost": {k: round(v, 1) for k, v in d.cost.items()}, "uncrewed": len(d.uncrewed),
            "statuses": dict(Counter(r.status for r in d.run.flights.values())),
            "riders": sum(len(v) for v in d.riders.values())}


def desk_replay(run: Path, out: Path, lines: dict[str, str], workers: int) -> None:
    """The standard crew desk again over `run`'s dates (see the module doc), then the guards and the report."""
    dates = sorted(d for d in lines if (run / "dates" / f"{d}.json").exists())
    t0 = time.monotonic()
    if dates:
        with ProcessPoolExecutor(min(workers, len(dates)), max_tasks_per_child=1) as pool:
            futures = {pool.submit(redesk, lines[d], run / "dates" / f"{d}.json", out): d
                       for d in sorted(dates, key=lambda d: -len(lines[d]))}
            for n, f in enumerate(as_completed(futures), 1):
                print(f"[{time.monotonic() - t0:8.0f}s] {n}/{len(dates)} {futures[f]} {f.result()}", flush=True)
    rows = [json.loads((out / "dates" / f"{d}.json").read_text()) for d in dates]
    bands = tuple(b for b in crews.BANDS if any(b in r.get("bands", {}) for r in rows))
    bands = ("middle", *(b for b in bands if b != "middle"))
    report = guards(rows, bands)
    report["desk_replay"] = {"over": str(run), "commit": commit(), "dates": len(rows),
                             "units_as_run": sum(r.get("desk_replay", {}).get("units_as_run", False) for r in rows),
                             "seconds": round(time.monotonic() - t0, 1)}
    (out / "guards.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    mark = report["desk_replay"]
    (out / "report.md").write_text(f"The standard crew desk replayed over {mark['over']} at commit {mark['commit']}; "
                                   "every other number is that run's.\n\n" + markdown(report, rows, bands))
    print(json.dumps({"desk_replay": mark, **report["guards"]}, indent=1, default=str), flush=True)


def redesk(line: str, path: Path, out: Path) -> str:
    """One date of the desk replay: its row from the earlier run with the desk's day played again on the units at the
    start computed as that run did, and the desk against each bound."""
    t0 = time.monotonic()
    row = json.loads(path.read_text())
    if "error" not in row:
        task = Task.from_dict(json.loads(line))
        w = crewday.world(task, INPUTS, DERIVED)
        units, before = crewday.storm_units(w, "middle", row["budget"]["morning"], row["budget"]["polish"])
        desk = crewsim.execute(task, {}, units, sim.truth(task), w.local, airports=w.airports, desk=True)
        j = row["bands"]["middle"]["joint"]
        j["played"]["crew_desk"] = outcome(desk)
        for name in ("notice", "flat"):
            n, value = j[name], j["played"]["crew_desk"][name]
            if n.get("bound") is not None and "below_bound" in n:
                n["below_bound"] = {k: v for k, v in n["below_bound"].items() if k != "crew_desk"} | (
                    {"crew_desk": value} if value < n["bound"] - 0.5 else {})
        ran = row["bands"]["middle"]["record"]["before_start"]
        row["desk_replay"] = {"seconds": round(time.monotonic() - t0, 1),
                              "units_as_run": all(before.get(k) == ran.get(k) for k in ("objective", "uncrewed"))}
    (out / "dates" / path.name).write_text(json.dumps(row, indent=1, default=str))
    return "error in the run" if "error" in row else f"{row['desk_replay']['seconds']}s"


def commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def summary(task: Task, j: crewplan.Joint, replayed: float, others: dict, reference: dict) -> dict:
    f, a = j.found, j.optimum
    day = f.day
    lb = j.bound
    below = {k: v for k, v in others.items() if v < lb - 0.5}
    cancelled = [i for i, r in day.run.flights.items() if r.status != FLOWN]
    crew_cancels = [i for i in cancelled if a.deps.get(i) is not None]
    later = [(i, r.dep - a.deps[i]) for i, r in day.run.flights.items()
             if r.status == FLOWN and a.deps.get(i) is not None and r.dep > a.deps[i]]
    cv = j.cover
    return {"best": round(f.value, 1), "claimed": round(f.claimed, 1), "replayed": round(replayed, 1),
            "exact": f.exact and abs(replayed - f.value) <= 0.5, "flown_without_crew": len(day.uncrewed),
            "bound": round(lb, 1), "gap_pct": round(j.gap_pct, 4), "pieces": {k: None if v is None else round(v, 1)
                                                                           for k, v in j.pieces.items()},
            "below_bound": below, "optimum": {**a.to_dict(), "reference": reference.get("objective")},
            "restricted": j.restricted.to_dict(), "reach": j.reach,
            "cover": None if cv is None else {k: round(v, 1) if isinstance(v, float) else v
                                               for k, v in cv.__dict__.items()},
            "aircraft": day.run.cost.to_dict(), "crew_cost": {k: round(v, 1) for k, v in day.cost.items()},
            "cancelled": len(cancelled), "crew_cancelled": len(crew_cancels),
            "crew_delayed": {"flights": len(later), "minutes": sum(m for _, m in later)},
            "optimum_cancelled": sum(d is None for d in a.deps.values()),
            "cabin_duty_h": duty_hours(day_duties(day)), "attempts": f.attempts}


def guards(rows: list[dict], bands: tuple[str, ...]) -> dict:
    ok = [r for r in rows if "error" not in r]
    storm = [r for r in ok if r["split"] != "control"]
    mid = [r["bands"]["middle"] for r in storm]
    rec = [b["record"]["pass"] for b in mid]
    cal = [b["calm"]["pass"] for b in mid]
    kept, left = [], []
    for r in ok:
        n = r["bands"]["middle"]["joint"]["notice"]
        (kept if n.get("bound") is not None and n.get("best") is not None else left).append(r)
    best = [best_ok(r["bands"]["middle"]["joint"]["notice"]) for r in kept]
    gaps = [r["bands"]["middle"]["joint"]["notice"]["gap_pct"] for r in kept]
    controls = {r["task_id"]: control_checks(r["bands"]["middle"]["joint"]) for r in ok if r["split"] == "control"}
    out = {
        "record": gate(share(rec), len(rec), f"flights without a legal crew <= {SHARE:.0%} of those flown and "
                       f"<= {MOST}, per storm date"),
        "calm": gate(share(cal), len(cal), f"crew-aware calm day rises <= {RISE}% over the calm optimum without "
                     "crews, per storm date"),
        "best": {"kept": len(kept), "left_out": len(left) + len(rows) - len(ok),
                 "left_out_tasks": sorted([r["task_id"] for r in left] + [r.get("task_id", r["date"])
                                                                         for r in rows if "error" in r]),
                 "replay_and_bound_ok": sum(best),
                 "failed": sorted(r["task_id"] for r, b in zip(kept, best, strict=True) if not b),
                 "gap_within": gap_counts(gaps), "gap_pct": dist(gaps), "pass": bool(kept) and all(best)},
        "controls": {"controls": controls, "n": len(controls), "pass": len(controls) == CONTROL_DATES
                     and all(c["ok"] for c in controls.values())},
    }
    out["go"] = all(g["pass"] for g in out.values())
    return {"guards": out, "reported": reported(ok, storm, bands), "timing": timing(rows)}


def best_ok(n: dict) -> bool:
    """The best plan replays exactly with a legal crew on every flight it flies, and no day played on the date costs
    less than the bound."""
    return bool(n["exact"]) and n["flown_without_crew"] == 0 and not n["below_bound"] and n["bound"] <= n["best"] + 0.5


def control_checks(j: dict) -> dict:
    best, flat = j["notice"]["best"], j["flat"]["best"]
    recorded, desk = j["recorded"]["total"], j["played"]["crew_desk"]["flat"]
    recorded_ok = recorded >= best
    desk_ok = desk <= (1 + CONTROL_MARGIN) * flat
    return {"ok": recorded_ok and desk_ok, "recorded": recorded, "best": best, "recorded_ge_best": recorded_ok,
            "crew_desk_flat": desk, "flat_best": flat, "crew_desk_flat_over_best": round(desk / flat - 1, 4),
            "crew_desk_within_margin": desk_ok, "naive_flat": j["played"]["naive"]["flat"],
            "bound": j["notice"]["bound"], "flat_bound": j["flat"]["bound"]}


def gate(value: float | None, n: int, rule: str) -> dict:
    return {"value": value, "threshold": GATE_SHARE, "n": n, "rule": rule,
            "pass": value is not None and value >= GATE_SHARE}


def share(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def gap_counts(gaps: list[float]) -> dict[str, int]:
    return {f"{g}%": sum(x <= g for x in gaps) for g in GAPS}


def reported(ok: list[dict], storm: list[dict], bands: tuple[str, ...]) -> dict:
    out: dict = {"bands": {}}
    for band in bands:
        bs = [r["bands"][band] for r in storm if band in r["bands"]]
        out["bands"][band] = {"record": share([b["record"]["pass"] for b in bs]),
                              "calm": share([b["calm"]["pass"] for b in bs]), "n": len(bs),
                              "uncrewed": dist([b["record"]["uncrewed"] for b in bs]),
                              "rise_pct": dist([b["calm"]["rise_pct"] for b in bs
                                                if b["calm"]["rise_pct"] is not None])}
    lows = [b["calm"]["rise_lower_pct"] for b in (r["bands"]["middle"] for r in storm)
            if b["calm"].get("rise_lower_pct") is not None]
    out["calm_rise_lower_pct"] = {**dist(lows), f"over_{RISE}": sum(v > RISE for v in lows)}
    js = [r["bands"]["middle"]["joint"] for r in storm]
    notice = [j["played"]["crew_desk"]["notice"] / j["notice"]["best"] - 1 for j in js]
    flat = [j["played"]["crew_desk"]["flat"] / j["flat"]["best"] - 1 for j in js]
    out["leverage"] = {"notice": {**dist(notice), "share_ge_10pct": share([v >= LEVERAGE for v in notice])},
                       "flat": {**dist(flat), "share_ge_10pct": share([v >= LEVERAGE for v in flat])}}
    ns = [r["bands"]["middle"]["joint"]["notice"] for r in ok]
    out["crew_caused"] = {"cancelled": dist([n["crew_cancelled"] for n in ns]),
                          "delayed_flights": dist([n["crew_delayed"]["flights"] for n in ns]),
                          "delay_minutes": dist([n["crew_delayed"]["minutes"] for n in ns]),
                          "over_optimum_pct": dist([100 * (n["best"] / n["optimum"]["objective"] - 1) for n in ns])}
    out["crew_cost"] = {k: {**dist([n["crew_cost"][k] for n in ns]), "total": round(sum(n["crew_cost"][k]
                                                                                        for n in ns), 1)}
                        for k in ("deadhead", "hotel", "end")}
    out["bound_pieces_best"] = dict(Counter(max((k for k, v in n["pieces"].items() if v is not None),
                                                key=lambda k: n["pieces"][k]) for n in ns))
    desks = [r["bands"]["middle"]["joint"]["played"] for r in ok]
    out["crew_desk"] = {"uncrewed": dist([p["crew_desk"]["uncrewed"] for p in desks]),
                        "cancelled": dist([p["crew_desk"]["statuses"].get(CANCELLED, 0) for p in desks]),
                        "auto_cancelled": dist([p["crew_desk"]["statuses"].get(AUTO_CANCELLED, 0) for p in desks]),
                        "over_naive_flat_pct": dist([100 * (p["crew_desk"]["flat"] / p["naive"]["flat"] - 1)
                                                     for p in desks])}
    trips_ = [v for r in ok for v in r["trips"].values()]
    out["trips"] = {"uncovered_legs": sum(v["uncovered"] for v in trips_),
                    "dates_with_uncovered": sum(any(v["uncovered"] for v in r["trips"].values()) for r in ok),
                    "builds": len(trips_), "converged": sum(v["converged"] for v in trips_),
                    "rounds": dist([v["rounds"] for v in trips_]), "gap_pct": dist([v["gap_pct"] for v in trips_])}
    out["cabin_duty_h"] = {name: {"max": dist([x["max"] for x in hs if "max" in x]),
                                  "over_14h": sum(x["over_14h"] for x in hs)}
                           for name, hs in (("best", [n["cabin_duty_h"] for n in ns]),
                                            ("calm", [r["bands"]["middle"]["calm"]["cabin_duty_h"] for r in ok]),
                                            ("record", [r["bands"]["middle"]["record"]["cabin_duty_h"] for r in ok]))}
    return out


def timing(rows: list[dict]) -> dict:
    laps: dict[str, list[float]] = {}
    for r in rows:
        for k, v in r.get("laps", {}).items():
            laps.setdefault(k, []).append(v)
    return {"per_task_s": dist([r["seconds"] for r in rows if "seconds" in r]),
            "sum_s": round(sum(r.get("seconds", 0) for r in rows), 1),
            "peak_rss_gb": dist([r["peak_rss_gb"] for r in rows if "peak_rss_gb" in r]),
            "laps": {k: dist(v) for k, v in laps.items()}}


def dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    return {"n": len(v), "min": round(v[0], 3), "p10": round(pct(v, 10), 3), "median": round(statistics.median(v), 3),
            "p90": round(pct(v, 90), 3), "max": round(v[-1], 3), "mean": round(statistics.mean(v), 3)}


def pct(v: list[float], p: float) -> float:
    return v[min(len(v) - 1, math.ceil(p / 100 * len(v)) - 1)]


def day_duties(day: crewsim.CrewDay) -> list[float]:
    """A crew sim day's cabin duty periods, report to release, in hours."""
    return [(d["release"] - d["report"]) / 60 for r in day.units.values() if r["kind"] == "cabin" for d in r["duties"]
            if d["report"] is not None and d["release"] is not None]


def column_duties(p, sol) -> list[float]:
    """A crew replay's cabin duty periods, report to release, in hours."""
    out = []
    for c in sol.columns.values():
        u = p.unit[c.unit]
        if u.kind != "cabin":
            continue
        for report, steps in ((c.report if c.report is not None else u.report, c.first),
                              (c.rest[1] if c.rest else None, c.second)):
            if report is not None and steps:
                out.append((p.leg[steps[-1][0]].arr + crews.DEBRIEF - report) / 60)
    return out


def duty_hours(hours: list[float]) -> dict:
    return {**dist(hours), "over_14h": sum(h > crews.CABIN_DUTY / 60 for h in hours)}


def units_at(units) -> dict:
    return dict(sorted(Counter(f"{u.kind}/{u.reserve or 'trip'}" for u in units).items()))


def markdown(report: dict, rows: list[dict], bands: tuple[str, ...]) -> str:
    g = report["guards"]
    out = ["# The storm day with crews: guards", "", "| Guard | Value | Pass |", "|---|---|---|"]
    for name in ("record", "calm"):
        out.append(f"| {name} | {g[name]['value']} of {g[name]['n']} storm dates (needs {GATE_SHARE}) | "
                   f"{g[name]['pass']} |")
    b = g["best"]
    out.append(f"| best | {b['replay_and_bound_ok']} of {b['kept']} kept dates replay exactly with a valid bound; "
               f"{b['left_out']} left out; gap within {b['gap_within']} | {b['pass']} |")
    out.append(f"| controls | {json.dumps(g['controls']['controls'], default=str)} | {g['controls']['pass']} |")
    out += ["", f"All guards pass: {g['go']}", "", "## Per date (middle band)", "",
            "| Date | Split | Flown | No legal crew (LP) | Rec | Calm rise % | Calm | A* | Best | Bound | Gap % | "
            "Exact | Desk / best | Seconds | GB |", "|" + "---|" * 15]
    for r in rows:
        if "error" in r:
            out.append(f"| {r['date']} | {r.get('split')} | error |" + " |" * 12)
            continue
        m = r["bands"]["middle"]
        rec, cal, n = m["record"], m["calm"], m["joint"]["notice"]
        out.append(f"| {r['date']} | {r['split']} | {rec['flown']} | {rec['uncrewed']} "
                   f"({rec['solve']['uncrewed_bound']}) | {rec['pass']} | {cal['rise_pct']} | {cal['pass']} | "
                   f"{n['optimum']['objective']} | {n['best']} | {n['bound']} | {n['gap_pct']} | {n['exact']} | "
                   f"{round(m['joint']['played']['crew_desk']['notice'] / n['best'], 4)} | {r['seconds']} | "
                   f"{r['peak_rss_gb']} |")
    out += ["", "## Reported", "", f"```\n{json.dumps(report['reported'], indent=1, default=str)}\n```", "",
            "## Timing", "", f"```\n{json.dumps(report['timing'], indent=1, default=str)}\n```", ""]
    for band in bands:
        out += [f"## Flights without a legal crew, {band} band", ""]
        for r in rows:
            for c in r.get("bands", {}).get(band, {}).get("record", {}).get("causes", []):
                out.append(f"- {r['date']} {c['flight']} {c['orig']}-{c['dest']} at {c['dep']}: "
                           + "; ".join(f"{k}: {v}" for k, v in c["missing"].items()))
        out.append("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
