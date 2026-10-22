"""Stage 2 of AirOpsEnv v3, as pre-registered: passengers in the crew world, and the reference under cost P that b6-v3
carries. Per date of b6-v2 (24 eval, 44 train, 2 control), each date's passengers booked (pax.book on the survey
extracts in tasks/pax and the following days' BTS schedules and T-100 loads), cost P with crews (crewsim.execute with
the passengers), the 3 h line and the default rebooking. No LLM is called.

    python tools/passenger_references.py --out cache/v3/references --aircraft cache/v3/plans [--workers 70]
        [--split all] [--only DATE,...] [--search-s 1800] [--cover-s 300] [--inputs cache/inputs]
        [--derived cache/derived] [--resume] [--summarize]

One job per date:
- candidates, each played with crews and passengers: v2's joint best plan (its crews named, fill off), B0 (the
  standard crew desk, aircraft doing nothing), and --aircraft's plans (tools/passenger_plans.py --split all, under
  the 3 h line: its incumbent, rounded LP plan and A_v2) through the standard crew desk;
- the repair: v2's crewplan.search from the cheapest of those desk days, the aircraft MIP priced by the passengers'
  proxy and every day played with the passengers, within --search-s seconds;
- the reference: the cheapest candidate or repaired day with a legal crew on every flight it flies, its plan, how it
  replays ("named": the plan's crews, fill off; "desk": the standard crew desk) and every candidate's P;
- the bound (as first registered): crewplan.lower on the aircraft model with every passenger term at its floor
  (pax.Proxy.floor, each term left out at least 0, so a bound under P): A*, the restricted A*, each unit's least crew
  cost and the covering bound (--cover-s); the gap (reference - bound) / reference is published with no threshold, and
  no played day may fall below the bound (b6-v3 publishes tools/passenger_bound.py's capacity-aware bound in its
  place);
- generator consistency checks: P1, the day's booked passengers against T-100's passengers per departure performed
  on its routes that month; P4, the same on the following days' schedules; and the per-date counts P2 (inbound
  transfers and departing passengers per connecting station) and P3 (scheduled connection times) pool;
- reported only: the reference's holds (flights leaving later than its aircraft-free day, crew waits included;
  count and minutes, against about 4 min for United's ConnectionSaver and 10 for American's Connect Assist), the
  controls (doing nothing, the default crew rule, and B0 against the reference), the share of passengers disrupted,
  their mean delay, the same-day, overnight and refund shares of those off cancelled flights, misconnects and refunds,
  and JetBlue's replayed day (--aircraft's).
The checks, as registered: P1 and P4 within 5% on at least 90% of dates; P2 within 1.5 points of the survey's
B6-to-B6 inbound-transfer share per period and station; P3 the median within 20 min of DB1C's per station with at
least 1,000 survey transfers. Every job writes dates/<date>.json; guards.json and report.md are recomputed from those
files alone (--summarize does only that).
"""

import argparse
import csv
import glob
import json
import resource
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from airops_core.build import fleet, nextday
from airops_core.core.sim import FLOWN
from airops_core.occ.pack import Pack
from airops_core.programs import crewbound, crewplan, crews, crewsim, pax, references, sim

ROOT = Path(__file__).resolve().parents[1]
P1_TOL, P1_SHARE, P2_POINTS, P3_MINUTES, P3_MIN_SURVEY = 0.05, 0.90, 1.5, 20.0, 1000
BIN = 15
STEP0 = ("incumbent", "rounded", "A_v2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--aircraft", type=Path, required=True,
                    help="a tools/passenger_plans.py run under the 3 h line with every candidate plan")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--split", default="all", help="eval, train, control or all")
    ap.add_argument("--only", default="", help="comma-separated dates")
    ap.add_argument("--search-s", type=float, default=1800.0)
    ap.add_argument("--cover-s", type=float, default=300.0)
    ap.add_argument("--inputs", type=Path, default=ROOT / "cache" / "inputs")
    ap.add_argument("--derived", type=Path, default=ROOT / "cache" / "derived")
    ap.add_argument("--pack", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    days = a.out / "dates"
    days.mkdir(parents=True, exist_ok=True)
    if not a.summarize:
        tasks = Pack(a.pack).tasks
        only = set(filter(None, a.only.split(",")))
        dates = sorted(v["date"] for v in tasks.values() if (a.split == "all" or v["split"] == a.split)
                       and (not only or v["date"] in only))
        todo = [d for d in dates if not (a.resume and (days / f"{d}.json").exists())]
        print(f"{len(todo)} jobs on {min(a.workers, max(1, len(todo)))} workers", file=sys.stderr)
        with ProcessPoolExecutor(min(a.workers, max(1, len(todo))), max_tasks_per_child=1) as ex:
            futures = {ex.submit(job, d, a): d for d in todo}
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    row = fut.result()
                except Exception:
                    row = {"date": d, "error": traceback.format_exc()}
                (days / f"{d}.json").write_text(json.dumps(row, indent=1))
                print(f"{d}: {brief(row)}", file=sys.stderr)
    summary = guards(a.out)
    (a.out / "guards.json").write_text(json.dumps(summary, indent=1))
    (a.out / "report.md").write_text(report(summary))
    print(json.dumps({k: summary["consistency"][k]["pass"] for k in ("P1", "P2", "P3", "P4")}))


def job(date: str, a: argparse.Namespace) -> dict:
    t0 = time.monotonic()
    pack = Pack(a.pack)
    tid = next(t for t, v in pack.tasks.items() if v["date"] == date)
    w = pack.world(tid)
    task, units = w.task, w.units
    passengers = pax.book(task, pax.Flows.load(date), pax.Dwell.load(), nextday.seats(task, a.inputs, a.derived))
    step0 = json.loads((a.aircraft / "dates" / f"{date}.json").read_text())
    world = sim.truth(task)

    def play(plan: dict, **kw) -> crewsim.CrewDay:
        return crewsim.execute(task, plan, units, world, w.local, airports=w.airports, pax=passengers, **kw)

    days = {"v2_best": (w.best_plan, "named", play(w.best_plan, fill=False)), "B0": ({}, "desk", play({}, desk=True))}
    for n in STEP0:
        if n in step0["plans"]:
            plan = plan_of(step0["plans"][n])
            days[f"{n}_desk"] = (plan, "desk", play(plan, desk=True))
    naive = play({})
    t1 = time.monotonic()
    first = min((n for n in days if n.endswith("_desk")), key=lambda n: crewplan.worth(days[n][2], False))
    air = crewplan.Aircraft(task, world, pax=pax.Proxy(task, passengers))
    found = crewplan.search(task, world, units, w.local, w.airports, air,
                            [(first, crewplan.as_solution(task, days[first][2], False))], t1 + a.search_s,
                            crew_limit=a.search_s / 4, air_limit=a.search_s / 4, pax=passengers)
    days["repair"] = (found.plan, "named", found.day)
    del air
    t2 = time.monotonic()
    legal = {n: not d.uncrewed for n, (_, _, d) in days.items()}
    p = {n: round(crewplan.worth(d, False), 1) for n, (_, _, d) in days.items()}
    won = min((n for n in days if legal[n]), key=lambda n: (p[n], n))
    plan, mode, ref = days[won]
    fair = crewplan.Aircraft(task, world, pax=pax.Proxy.floor(task, passengers))
    fl = crewbound.Flights.of(task, world.block, fair.net.copies)
    reached = crewbound.reach(task, fl, units, w.local)
    least = crewbound.least(task, fl, units, w.local, crews.Ends(units, w.airports, task.date).least, reached)
    opt, restricted, lp, up, _ = crewplan.optima(task, world, fair, fair.cost(False), reached, a.search_s / 4)
    pieces, _ = crewplan.lower(task, units, w.local, w.airports, fair, fl, least, opt, restricted, lp, up,
                               crewplan.Found(plan, ref, p[won], False), a.cover_s, crewbound.WIDTH)
    bound = max(v for v in pieces.values() if v is not None)
    t3 = time.monotonic()
    played = {**p, "naive": round(crewplan.worth(naive, False), 1)}
    return {"date": date, "task_id": tid, "split": pack.tasks[tid]["split"], "flows": passengers.source,
            "passengers": passengers.to_dict(),
            "reference": {"from": won, "P": p[won], "mode": mode, "plan": {str(k): v for k, v in plan.items()},
                          "bound": round(bound, 1), "gap_pct": round(100 * (p[won] - bound) / p[won], 4),
                          "pieces": {k: None if v is None else round(v, 1) for k, v in pieces.items()}},
            "candidates": {n: {"P": p[n], "legal": legal[n], "uncrewed": len(days[n][2].uncrewed)} for n in days},
            "below_bound": {n: v for n, v in played.items() if v < bound - 0.5},
            "repair": {"from": first, "attempts": [{k: v for k, v in x.items() if k in ("try", "replayed")}
                                                   for x in found.attempts]},
            "consistency": consistency(task, passengers, a.inputs),
            "holds": holds(task, ref.run),
            "controls": {"naive": played["naive"], "B0": p["B0"], "reference": p[won],
                         "naive_over_reference": played["naive"] / p[won], "B0_over_reference": p["B0"] / p[won]},
            "reported": reported(passengers, ref.run),
            "airline": step0.get("airline"),
            "wall_s": {"candidates": round(t1 - t0, 1), "repair": round(t2 - t1, 1), "bound": round(t3 - t2, 1),
                       "total": round(time.monotonic() - t0, 1)},
            "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin"
                                                                                  else 2 ** 20), 2)}


def plan_of(raw: dict) -> dict:
    return {int(k) if k.isdigit() else k: v for k, v in raw.items()}


def consistency(task, passengers: pax.Passengers, inputs: Path) -> dict:
    """The per-date pieces of P1-P4."""
    loads = fleet.route_loads(task.carrier, task.date[:7], inputs)
    covered = [f for f in task.flights if (f.orig, f.dest) in loads]
    p1 = {"booked": sum(f.pax for f in covered), "expected": round(sum(loads[(f.orig, f.dest)] for f in covered), 1),
          "flights": len(task.flights), "covered": len(covered)}
    by_day: dict[int, list[pax.Seat]] = defaultdict(list)
    for s in passengers.later:
        by_day[s.day].append(s)
    booked = expected = 0.0
    months: dict[str, dict] = {}
    for k, seats in sorted(by_day.items()):
        month = later_month(task.date, k)
        if not fleet.t100_path(inputs, month).exists():
            continue
        months.setdefault(month, fleet.route_loads(task.carrier, month, inputs))
        for s in seats:
            if (s.orig, s.dest) in months[month]:
                booked += s.booked
                expected += months[month][(s.orig, s.dest)]
    task_by = task.by_id
    departing: Counter = Counter()
    transfers: Counter = Counter()
    dwell: dict[str, Counter] = defaultdict(Counter)
    for f in task.flights:
        if f.orig in pax.CONNECT:
            departing[f.orig] += f.pax
    for g in passengers.groups:
        if len(g.legs) == 2:
            i, j = task_by[g.legs[0]], task_by[g.legs[1]]
            transfers[j.orig] += g.pax
            dwell[j.orig][(j.sched_dep - i.sched_arr) // BIN * BIN] += g.pax
    return {"P1": p1, "P4": {"booked": booked, "expected": round(expected, 1)},
            "P2": {"departing": dict(departing), "transfers": dict(transfers)},
            "P3": {h: {str(b): n for b, n in sorted(c.items())} for h, c in dwell.items()}}


def later_month(date: str, k: int) -> str:
    return (Date.fromisoformat(date) + timedelta(days=k)).isoformat()[:7]


def holds(task, run: sim.Execution) -> dict:
    """Flights leaving later than the same cancellations' aircraft-free day, crew waits included."""
    cut = {i: {"cancel": True, "decided": task.start} for i, r in run.flights.items() if r.status != FLOWN}
    free = sim.execute(task, references.so_far(task, cut, task.start), sim.truth(task)).flights
    minutes = sorted(r.dep - free[i].dep for i, r in run.flights.items()
                     if r.status == FLOWN and free[i].status == FLOWN and r.dep > free[i].dep)
    n = len(minutes)
    return {"count": n, "median": statistics.median(minutes) if n else None,
            "p90": minutes[int(0.9 * (n - 1))] if n else None,
            "share_le_4": sum(m <= 4 for m in minutes) / n if n else None,
            "share_le_10": sum(m <= 10 for m in minutes) / n if n else None,
            "share_over_15": sum(m > 15 for m in minutes) / n if n else None}


def reported(passengers: pax.Passengers, run: sim.Execution) -> dict:
    groups = {g.id: g for g in passengers.groups}
    trips = sum(g.pax for g in groups.values())
    cut = {i for i, g in groups.items() if any(run.flights[k].status != FLOWN for k in g.legs)}
    hit = [p for p in run.pax.parts if p.kind != pax.FLEW]
    off = Counter()
    for p in run.pax.parts:
        if p.group in cut:
            off[p.kind] += p.pax
    n_off = sum(off.values())
    n_hit = sum(p.pax for p in hit)
    return {"disrupted_share": n_hit / trips, "disrupted_mean_delay": (sum(p.pax * p.delay for p in hit) / n_hit
                                                                      if n_hit else None),
            "off_cancelled": n_off, **{f"off_cancelled_{k}": off[k] / n_off if n_off else None
                                       for k in (pax.REBOOKED, pax.STRANDED, pax.REFUNDED)},
            "misconnected": run.pax.misconnected(), "refunded": run.pax.outcomes()[pax.REFUNDED],
            "overnight": run.pax.outcomes()[pax.STRANDED]}


def brief(row: dict) -> str:
    if "error" in row:
        return "ERROR " + row["error"].strip().splitlines()[-1]
    r = row["reference"]
    c = row["consistency"]["P1"]
    return (f"reference {r['from']} {r['P']:.0f}, bound {r['bound']:.0f}, gap {r['gap_pct']}%; P1 "
            f"{c['booked'] / c['expected']:.3f}; {row['wall_s']['total']} s")


def survey() -> tuple[dict[tuple[str, str], float], dict[str, tuple[float, int]]]:
    """The survey's B6-to-B6 inbound-transfer share per (period, station), and DB1C's median connection time and
    transfers per station, pooled over its months."""
    with open(str(pax.DATA / "b6-transfers.csv"), newline="") as fh:
        share = {(r["period"], r["hub"]): int(r["transfers"]) / int(r["departing"]) for r in csv.DictReader(fh)
                 if r["orig"] == "*" and r["dest"] == "*"}
    bins: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    with open(str(pax.DATA / "b6-dwell.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["period"] == "all":
                bins[r["hub"]].append((int(r["lo"]), int(r["hi"]), int(r["transfers"])))
    return share, {h: (median_of(sorted(v)), sum(n for *_, n in v)) for h, v in bins.items()}


def median_of(bins: list[tuple[int, int, float]]) -> float:
    """The median of a histogram of [lo, hi) bins, interpolated within its bin."""
    total, run = sum(n for *_, n in bins), 0.0
    for lo, hi, n in bins:
        if run + n >= total / 2:
            return lo + (hi - lo) * (total / 2 - run) / n
        run += n
    return float("nan")


def guards(out: Path) -> dict:
    rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(out / "dates" / "*.json")))]
    ok = [r for r in rows if "error" not in r]
    share, dwell = survey()
    p1 = {r["date"]: r["consistency"]["P1"]["booked"] / r["consistency"]["P1"]["expected"] for r in ok}
    p4 = {r["date"]: r["consistency"]["P4"]["booked"] / r["consistency"]["P4"]["expected"] for r in ok
          if r["consistency"]["P4"]["expected"]}
    pooled: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in ok:
        period = r["flows"].split()[-1]
        for h, n in r["consistency"]["P2"]["departing"].items():
            pooled[(period, h)][0] += r["consistency"]["P2"]["transfers"].get(h, 0)
            pooled[(period, h)][1] += n
    p2 = {f"{k[0]} {k[1]}": {"generated": 100 * t / d, "survey": 100 * share[k],
                             "ok": abs(100 * t / d - 100 * share[k]) <= P2_POINTS}
          for k, (t, d) in sorted(pooled.items()) if d and k in share}
    hist: dict[str, Counter] = defaultdict(Counter)
    for r in ok:
        for h, c in r["consistency"]["P3"].items():
            hist[h].update({int(b): n for b, n in c.items()})
    p3 = {}
    for h, c in sorted(hist.items()):
        if dwell.get(h, (0, 0))[1] >= P3_MIN_SURVEY:
            mine = median_of([(b, b + BIN, n) for b, n in sorted(c.items())])
            p3[h] = {"generated": mine, "survey": dwell[h][0], "ok": abs(mine - dwell[h][0]) <= P3_MINUTES}

    def within(values: dict[str, float]) -> dict:
        good = sum(abs(v - 1) <= P1_TOL for v in values.values())
        return {"dates": len(values), "within": good, "pass": bool(values) and good >= P1_SHARE * len(values),
                "values": {d: round(v, 4) for d, v in values.items()}}

    controls = {r["date"]: r["controls"] for r in ok if r["split"] == "control"}
    gaps = [r["reference"]["gap_pct"] for r in ok]
    return {"consistency": {"P1": within(p1), "P4": within(p4),
                            "P2": {"pass": all(v["ok"] for v in p2.values()), "values": p2},
                            "P3": {"pass": all(v["ok"] for v in p3.values()), "values": p3}},
            "reference": {"won": dict(Counter(r["reference"]["from"] for r in ok)),
                          "gap_pct": {"median": statistics.median(gaps) if gaps else None,
                                      "max": max(gaps, default=None)},
                          "below_bound": {r["date"]: r["below_bound"] for r in ok if r["below_bound"]}},
            "holds": {r["date"]: r["holds"] for r in ok}, "controls": controls,
            "reported": {r["date"]: r["reported"] for r in ok},
            "dates": len(ok), "errors": [r["date"] for r in rows if "error" in r]}


def report(s: dict) -> str:
    c = s["consistency"]
    lines = ["# v3 stage 2: passengers in the crew world (numbers only)", "",
             f"- P1 booked vs T-100: {c['P1']['within']} of {c['P1']['dates']} within 5% "
             f"({'pass' if c['P1']['pass'] else 'fail'})",
             f"- P2 connecting share vs survey: {sum(v['ok'] for v in c['P2']['values'].values())} of "
             f"{len(c['P2']['values'])} within 1.5 points ({'pass' if c['P2']['pass'] else 'fail'})",
             f"- P3 connection time vs DB1C: {sum(v['ok'] for v in c['P3']['values'].values())} of "
             f"{len(c['P3']['values'])} within 20 min ({'pass' if c['P3']['pass'] else 'fail'})",
             f"- P4 following days vs T-100: {c['P4']['within']} of {c['P4']['dates']} within 5% "
             f"({'pass' if c['P4']['pass'] else 'fail'})",
             f"- reference from: {s['reference']['won']}; gap median {s['reference']['gap_pct']['median']}% max "
             f"{s['reference']['gap_pct']['max']}%; played days below the bound: {s['reference']['below_bound']}",
             f"- controls: {s['controls']}"]
    if s["errors"]:
        lines.append(f"- errors: {s['errors']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
