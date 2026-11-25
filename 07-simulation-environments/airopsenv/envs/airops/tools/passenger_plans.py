"""Step 0 of AirOpsEnv v3, the passengers' premise, as re-registered: with passengers priced by cost P (programs.pax)
and rebooked by the default (first come, first served, the earliest scheduled arrival first), does a passenger-aware
plan beat v2's by enough to matter? Aircraft only, the v2 step-0 world (the whole day from 04:00, the truth); no LLM
is called. With --split all its plans are the passenger-aware aircraft candidates of the reference under P
(tools/passenger_references.py --aircraft).

    python tools/passenger_plans.py --out cache/v3/plans [--workers 24] [--seeds 4] [--split eval]
        [--only DATE,...] [--sensitivity 0.5,2] [--time-limit 1800] [--inputs cache/inputs]
        [--derived cache/derived] [--board cache/dataset/runs/occ-board-sol] [--resume] [--summarize] [--crew]

One job per date, at the date's measured survey shares:
- A_v2: v2's references.anchor plan (the hindsight MIP under v2's cost);
- the passenger-aware MIP (mip.Network with pax.Proxy), started from whichever the proxy prices lowest of A_v2's day,
  the empty plan's and its LP relaxation rounded (Network.rounded), --time-limit seconds in all, searched by --seeds
  single-threaded HiGHS runs at once under different random seeds (forked from the job): its incumbent is theirs at
  the lowest objective, its bound the highest of theirs (each is valid for the one model);
- A_pax: the cheapest under P, default rebooking, of the MIP's incumbent at its limit, the rounded LP plan, A_v2 and
  the empty plan (ties in that order); the winner, every candidate's P and plan (to re-score any date), and the
  MIP's gap and status are kept;
- G1: P(A_v2) >= 1.05 x P(A_pax) on at least 12 of the 24 eval dates; G2, reported only: the MIP's gap and status;
- reported: the rebooking lever, P(A_pax) - P(A_pax, optimal assignment) (pax.optimal) as a share of P(A_pax); the
  changed cancellations (auto-cancels included) and holds (references.deliberate's), A_v2 against A_pax; misconnects,
  overnight strands and refunds; JetBlue's replayed day (references.airline_plan) under P: of the passengers off its
  cancelled flights the shares rebooked the same day, overnight and refunded, and their mean and median delay; and at
  each --sensitivity factor of the survey flows, A_v2 and A_pax re-scored under P (no new solve).
The crew world (the pack's joint best plan and B0, and the v2 Sol board's days, from the dataset's runs/) is --crew;
it adds to the files already written.
Every job writes dates/<date>.json as it finishes (--resume keeps those written); guards.json and report.md are
recomputed from those files alone (--summarize does only that).
"""

import argparse
import glob
import json
import multiprocessing
import resource
import statistics
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import highspy
from airops_core.build import nextday
from airops_core.core.sim import FLOWN
from airops_core.occ.pack import Pack
from airops_core.programs import crewsim, mip, pax, references, sim
from airops_core.programs.cost import REFUND

ROOT = Path(__file__).resolve().parents[1]
G1_RATIO, G1_DATES = 1.05, 12
MARGIN = 30.0
CANDIDATES = ("incumbent", "rounded", "A_v2", "empty")
SEARCH: tuple | None = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1, help="HiGHS searches of the passenger MIP at once, per job")
    ap.add_argument("--split", default="eval", help="eval, train, control or all")
    ap.add_argument("--only", default="", help="comma-separated dates")
    ap.add_argument("--sensitivity", default="0.5,2")
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--inputs", type=Path, default=ROOT / "cache" / "inputs")
    ap.add_argument("--derived", type=Path, default=ROOT / "cache" / "derived")
    ap.add_argument("--board", type=Path, default=ROOT / "cache" / "dataset" / "runs" / "occ-board-sol")
    ap.add_argument("--pack", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--crew", action="store_true", help="only score the crew world into the files written")
    a = ap.parse_args()
    days = a.out / "dates"
    days.mkdir(parents=True, exist_ok=True)
    if a.crew:
        for f in sorted(days.glob("*.json")):
            row = json.loads(f.read_text())
            if "error" not in row:
                w, passengers = booked(row["date"], 1.0, a)
                row["crew"] = crew_world(w, passengers, a.board)
                f.write_text(json.dumps(row, indent=1))
                print(f"{row['date']}: crew world {len(row['crew'])} days", file=sys.stderr)
    elif not a.summarize:
        tasks = Pack(a.pack).tasks
        only = set(filter(None, a.only.split(",")))
        dates = sorted(v["date"] for v in tasks.values() if (a.split == "all" or v["split"] == a.split)
                       and (not only or v["date"] in only))
        todo = [d for d in dates if not (a.resume and path(days, d).exists())]
        print(f"{len(todo)} jobs on {min(a.workers, max(1, len(todo)))} workers, {a.seeds} HiGHS searches each",
              file=sys.stderr)
        with ProcessPoolExecutor(min(a.workers, max(1, len(todo))), max_tasks_per_child=1) as ex:
            futures = {ex.submit(job, d, a): d for d in todo}
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    row = fut.result()
                except Exception:
                    row = {"date": d, "error": traceback.format_exc()}
                path(days, d).write_text(json.dumps(row, indent=1))
                print(f"{d}: {brief(row)}", file=sys.stderr)
    summary = guards(a.out)
    (a.out / "guards.json").write_text(json.dumps(summary, indent=1))
    (a.out / "report.md").write_text(report(summary))
    print(json.dumps({"G1": summary["G1"]["pass"], "G1_count": summary["G1"]["passing"]}))


def path(days: Path, date: str) -> Path:
    return days / f"{date}.json"


def booked(date: str, k: float, a: argparse.Namespace):
    pack = Pack(a.pack)
    w = pack.world(next(t for t, v in pack.tasks.items() if v["date"] == date))
    flows = pax.Flows.load(date)
    flows = flows if k == 1 else flows.scaled(k)
    return w, pax.book(w.task, flows, pax.Dwell.load(), nextday.seats(w.task, a.inputs, a.derived))


def job(date: str, a: argparse.Namespace) -> dict:
    t0 = time.monotonic()
    w, passengers = booked(date, 1.0, a)
    task, tid = w.task, w.task.task_id
    proxy = pax.Proxy(task, passengers)
    t1 = time.monotonic()
    anchor, _ = references.anchor(task)
    t2 = time.monotonic()
    net = mip.Network(task, pax=proxy)
    plans = {"A_v2": anchor.plan(task.start), "empty": {}}
    rounded = net.rounded(a.time_limit)
    if rounded is not None:
        plans["rounded"] = rounded
    start = min((sim.execute(task, p) for p in plans.values()), key=proxy.cost)
    sol, searches = portfolio(net, net.vector(start), max(60.0, a.time_limit - (time.monotonic() - t2) - MARGIN),
                              a.seeds)
    t3 = time.monotonic()
    plans["incumbent"] = sol.plan(task.start)
    scored = {n: sim.execute(task, plans[n], pax=passengers) for n in CANDIDATES if n in plans}
    won = min(scored, key=lambda n: (scored[n].cost.total, CANDIDATES.index(n)))
    best, v2 = scored[won], scored["A_v2"]
    optimal = pax.optimal(task, passengers, best.flights)
    p_optimal = (best.cost.total - best.pax.delay - best.pax.lodging - best.pax.fare + optimal.delay + optimal.lodging
                 + optimal.fare)
    sensitivity = {}
    for k in (float(x) for x in a.sensitivity.split(",") if x):
        _, scaled = booked(date, k, a)
        pv, pp = (sim.execute(task, plans[n], pax=scaled).cost.total for n in ("A_v2", won))
        sensitivity[f"{k:g}"] = {"A_v2": pv, "A_pax": pp, "ratio": pv / pp}
    return {"date": date, "task_id": tid, "split": Pack(a.pack).tasks[tid]["split"], "flows": passengers.source,
            "passengers": counts(passengers),
            "P": {n: r.cost.total for n, r in scored.items()}, "won": won,
            "plans": {n: {str(k): v for k, v in plans[n].items()} for n in scored},
            "ratio": v2.cost.total / best.cost.total, "G1": v2.cost.total >= G1_RATIO * best.cost.total,
            "rebooking": {"default": best.cost.total, "optimal": p_optimal,
                          "share": (best.cost.total - p_optimal) / best.cost.total},
            "mip": {**sol.to_dict(), "start": round(proxy.cost(start), 1), "wall_s": round(t3 - t2, 1),
                    "searches": searches},
            "anchor": anchor.to_dict(), "changed": changed(task, v2, best),
            "costs": {"A_v2": v2.cost.to_dict(), "A_pax": best.cost.to_dict()},
            "outcomes": {n: {**scored[n].pax.outcomes(), "misconnected": scored[n].pax.misconnected()} for n in
                         ("A_v2", won)},
            "optimal_outcomes": optimal.outcomes(),
            "airline": replay(task, passengers),
            "sensitivity": sensitivity,
            "wall_s": {"build": round(t1 - t0, 1), "anchor": round(t2 - t1, 1), "pax_mip": round(t3 - t2, 1),
                       "total": round(time.monotonic() - t0, 1)},
            "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin"
                                                                                  else 2 ** 20), 2)}


def portfolio(net: mip.Network, x, limit: float, seeds: int) -> tuple[mip.Solution, list[dict]]:
    """The MIP searched under `seeds` random seeds at once: the lowest incumbent, with the highest bound."""
    global SEARCH
    SEARCH = (net, x, limit)
    highspy.Highs.resetGlobalScheduler(True)
    if seeds == 1:
        sols = [search(0)]
    else:
        with ProcessPoolExecutor(seeds, mp_context=multiprocessing.get_context("fork")) as ex:
            sols = list(ex.map(search, range(seeds)))
    best = min(sols, key=lambda s: (s.objective, sols.index(s)))
    bound = max(s.bound for s in sols)
    proven = any(s.proven for s in sols)
    gap = 0.0 if proven else round(100 * max(0.0, best.objective - bound) / max(1.0, abs(best.objective)), 4)
    return (replace(best, status="Optimal" if proven else best.status, bound=bound, gap_pct=gap),
            [{"seed": k, **{f: getattr(s, f) for f in ("status", "objective", "gap_pct", "wall_s")},
              "bound": round(s.bound, 1)} for k, s in enumerate(sols)])


def search(seed: int) -> mip.Solution:
    net, x, limit = SEARCH
    return net.solve(limit, canonical=False, start=x, seed=seed)


def counts(p: pax.Passengers) -> dict:
    conn = sum(g.pax for g in p.groups if len(g.legs) == 2)
    trips = sum(g.pax for g in p.groups)
    return {"groups": len(p.groups), "trips": trips, "connecting": conn, "connecting_share": conn / trips,
            "later_flights": len(p.later), "fare": round(p.fare, 1)}


def changed(task, v2: sim.Execution, best: sim.Execution) -> dict:
    out = {}
    for name, run in (("A_v2", v2), ("A_pax", best)):
        cut = {i for i, r in run.flights.items() if r.status != FLOWN}
        plan = {i: {"cancel": True, "decided": task.start} for i in cut}
        held = references.deliberate(task, plan, {i: r.dep for i, r in run.flights.items() if i not in cut},
                                     sim.truth(task), task.start)
        out[name] = (cut, set(held))
    both = [i for i in v2.flights if i not in out["A_v2"][0] | out["A_pax"][0]]
    moved = [abs(best.flights[i].dep - v2.flights[i].dep) for i in both]

    def share(a: set, b: set) -> float | None:
        return len(a ^ b) / len(a | b) if a | b else None

    (c1, h1), (c2, h2) = out["A_v2"], out["A_pax"]
    return {"cancelled": {"A_v2": len(c1), "A_pax": len(c2), "changed_share": share(c1, c2)},
            "held": {"A_v2": len(h1), "A_pax": len(h2), "changed_share": share(h1, h2)},
            "either_changed_share": share(c1 | h1, c2 | h2),
            "moved_over_5": sum(m > 5 for m in moved), "moved_over_15": sum(m > 15 for m in moved)}


def replay(task, passengers: pax.Passengers) -> dict:
    """JetBlue's replayed day under P: what became of the passengers off its cancelled flights."""
    run = sim.execute(task, references.airline_plan(task), pax=passengers)
    groups = {g.id: g for g in passengers.groups}
    cut = {i for i, g in groups.items() if any(run.flights[k].status != FLOWN for k in g.legs)}
    kinds: Counter = Counter()
    travel = []
    for p in run.pax.parts:
        if p.group not in cut:
            continue
        kinds["same_day" if p.kind == pax.REBOOKED else p.kind] += p.pax
        if p.kind != pax.REFUNDED:
            travel.append((p.delay, p.pax))
    n = sum(kinds.values())
    return {"P": run.cost.total, "off_cancelled": n, "share_of_trips": n / sum(g.pax for g in groups.values()),
            **{k: kinds[k] / n if n else None for k in ("same_day", pax.STRANDED, pax.REFUNDED)},
            "mean_delay": sum(d * w for d, w in travel) / sum(w for _, w in travel) if travel else None,
            "median_delay": wmedian(travel)}


def wmedian(pairs: list[tuple[int, float]]) -> int | None:
    pairs = sorted(pairs)
    total, run = sum(w for _, w in pairs), 0.0
    for v, w in pairs:
        run += w
        if run >= total / 2:
            return v
    return None


def crew_world(w, passengers: pax.Passengers, board: Path) -> dict:
    """The pack's joint best plan and B0, and the Sol board's days of the date, under P and under v2's cost."""
    task = w.task
    out = {}
    for name, plan, kw, ref in (("best", w.best_plan, {"fill": False}, w.refs["best"]),
                                ("B0", {}, {"desk": True}, w.refs["B0"]["notice"])):
        day = crewsim.execute(task, plan, w.units, local=w.local, airports=w.airports, pax=passengers, **kw)
        out[name] = {"P": day.total, "v2": ref, "outcomes": day.run.pax.outcomes()}
    size = {u.id: u.size for u in w.units}
    for f in sorted(glob.glob(str(board / "days" / f"{task.task_id}-*.json"))):
        d = json.loads(Path(f).read_text())
        runs = {r["id"]: sim.FlightRun(**r) for r in d["grade"]["executed"]}
        riders: dict[int, int] = {}
        for uid, c in d["crews"].items():
            for duty in c["duties"]:
                for dh in duty["deadheads"]:
                    riders[dh[0]] = riders.get(dh[0], 0) + size[uid]
        day = pax.fly(task, passengers, runs, riders)
        lost = sum(task.flight(i).pax for i, r in runs.items() if r.status != FLOWN)
        v2 = d["grade"]["costs"]
        out[Path(f).stem] = {"P": v2["total"] - v2["delay"] - v2["cancel"] + day.delay + REFUND * lost + day.lodging
                             + day.fare, "v2": v2["total"], "outcomes": day.outcomes()}
    return out


def brief(row: dict) -> str:
    if "error" in row:
        return "ERROR " + row["error"].strip().splitlines()[-1]
    m = row["mip"]
    return (f"P(A_v2)/P(A_pax) {row['ratio']:.3f} ({row['won']} won); mip {m['status']} gap {m['gap_pct']}% "
            f"{m['wall_s']} s")


def guards(out: Path) -> dict:
    rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(out / "dates" / "*.json")))]
    ok = [r for r in rows if "error" not in r]
    base = [r for r in ok if r["split"] == "eval"]
    passing = [r["date"] for r in base if r["G1"]]
    return {"G1": {"rule": f"P(A_v2) >= {G1_RATIO} x P(A_pax) on >= {G1_DATES} of 24 eval dates at measured shares",
                   "dates": len(base), "passing": len(passing), "pass": len(base) == 24 and len(passing) >= G1_DATES,
                   "passing_dates": passing, "ratios": {r["date"]: round(r["ratio"], 4) for r in base}},
            "G2_reported": {r["date"]: {"status": r["mip"]["status"], "gap_pct": r["mip"]["gap_pct"],
                                        "wall_s": r["mip"]["wall_s"]} for r in ok},
            "all_dates": {r["date"]: {"split": r["split"], "ratio": round(r["ratio"], 4), "won": r["won"],
                                      "rebooking_share": round(r["rebooking"]["share"], 4),
                                      "sensitivity": {k: round(v["ratio"], 4) for k, v in r["sensitivity"].items()}}
                          for r in ok},
            "won": dict(Counter(r["won"] for r in ok)),
            "median_ratio": statistics.median([r["ratio"] for r in ok]) if ok else None,
            "errors": [r["date"] for r in rows if "error" in r]}


def report(s: dict) -> str:
    g = s["G1"]
    lines = ["# v3 step 0: passengers (numbers only)", "",
             f"- G1: {'PASS' if g['pass'] else 'FAIL'}, {g['passing']} of {g['dates']} ({g['rule']})",
             f"- A_pax winners: {s['won']}; median P(A_v2)/P(A_pax) {s['median_ratio']}", "",
             "| date | split | P(A_v2)/P(A_pax) | won | rebooking lever | x0.5 | x2 | MIP status | gap % | s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for d, r in sorted(s["all_dates"].items()):
        g2 = s["G2_reported"][d]
        lines.append(f"| {d} | {r['split']} | {r['ratio']:.3f} | {r['won']} | {100 * r['rebooking_share']:.1f}% | "
                     f"{r['sensitivity'].get('0.5', '-')} | {r['sensitivity'].get('2', '-')} | {g2['status']} | "
                     f"{g2['gap_pct']} | {g2['wall_s']} |")
    if s["errors"]:
        lines += ["", f"errors: {s['errors']}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
