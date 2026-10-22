"""A lower bound under cost P for the reference under P, from a finished run of tools/passenger_references.py alone
(its stored passengers and played costs) and the pack, without rerunning it: the capacity-aware floor
(pax.Proxy.lower) in v2's bound machinery. b6-v3's bounds took --limit 1800.

    python tools/passenger_bound.py --reference cache/v3/references --out cache/v3/bound [--split all]
        [--only DATE,...] [--workers 8] [--limit 900] [--resume] [--summarize]

Per date: the aircraft model priced by pax.Proxy.lower (each passenger at the cheapest outcome P could give it, today's
flights at their earliest possible arrival, `arrivals`, seats relaxed), with crewplan.optima's A* and the restricted A*
(every flight kept to the minutes some crew could fly it, crewbound.reach) within --limit seconds each, their HiGHS dual
bounds, and the restricted A*'s plus each unit's least crew cost (crewbound.least). The bound is the largest; v2's
covering bound is left out, its argument not checked for the passenger columns. Every played day of the run (each
candidate, doing nothing) must cost at least the bound; the gap (reference - bound) / reference is published.
dates/<date>.json as each finishes, then guards.json and report.md from those files.
"""

import argparse
import glob
import json
import resource
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from airops_core.occ.pack import Pack
from airops_core.programs import crewbound, crewplan, crews, pax, sim

TOL = 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=float, default=900.0)
    ap.add_argument("--pack", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    days = a.out / "dates"
    days.mkdir(parents=True, exist_ok=True)
    if not a.summarize:
        only = set(filter(None, a.only.split(",")))
        rows = [Path(f) for f in sorted(glob.glob(str(a.reference / "dates" / "*.json")))]
        todo = [f.stem for f in rows if (not only or f.stem in only)
                and not (a.resume and (days / f.name).exists())]
        tasks = Pack(a.pack).tasks
        todo = [d for d in todo if a.split == "all" or next(v["split"] for v in tasks.values() if v["date"] == d)
                == a.split]
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
    print(json.dumps({"valid": summary["valid"], "dates": summary["dates"]}))


def job(date: str, a: argparse.Namespace) -> dict:
    t0 = time.monotonic()
    ref = json.loads((a.reference / "dates" / f"{date}.json").read_text())
    if "error" in ref:
        raise RuntimeError("no reference on this date")
    pack = Pack(a.pack)
    w = pack.world(next(t for t, v in pack.tasks.items() if v["date"] == date))
    task, units = w.task, w.units
    passengers = pax.Passengers.from_dict(ref["passengers"])
    world = sim.truth(task)
    lower = pax.Proxy.lower(task, passengers, arrivals(task, world))
    air = crewplan.Aircraft(task, world, pax=lower)
    t1 = time.monotonic()
    fl = crewbound.Flights.of(task, world.block, air.net.copies)
    reached = crewbound.reach(task, fl, units, w.local)
    least = crewbound.least(task, fl, units, w.local, crews.Ends(units, w.airports, task.date).least, reached)
    t2 = time.monotonic()
    opt, restricted, _, _, _ = crewplan.optima(task, world, air, air.cost(False), reached, a.limit)
    pieces = {"aircraft": opt.bound, "restricted": restricted.bound,
              "restricted_least": restricted.bound + float(least.sum())}
    bound = max(pieces.values())
    played = {n: c["P"] for n, c in ref["candidates"].items()} | {"naive": ref["controls"]["naive"]}
    best = ref["reference"]["P"]
    return {"date": date, "split": ref["split"], "reference": best, "bound": round(bound, 1),
            "gap_pct": round(100 * (best - bound) / best, 4), "floor": ref["reference"]["bound"],
            "pieces": {k: round(v, 1) for k, v in pieces.items()},
            "status": {"aircraft": [opt.status, opt.gap_pct, opt.objective],
                       "restricted": [restricted.status, restricted.gap_pct, restricted.objective]},
            "played": played, "valid": all(v >= bound - TOL for v in played.values()),
            "size": {"cols": opt.cols, "rows": opt.rows, "modes": len(lower.modes),
                     "options": sum(len(m.options) for m in lower.modes)},
            "wall_s": {"build": round(t1 - t0, 1), "reach": round(t2 - t1, 1),
                       "optima": round(time.monotonic() - t2, 1), "total": round(time.monotonic() - t0, 1)},
            "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2 ** 30 if sys.platform == "darwin"
                                                                                  else 2 ** 20), 2)}


def arrivals(task, world: sim.World) -> dict[int, int]:
    """Each flight's earliest possible arrival: its block after the first minute from its earliest that no stop holds
    and its crew may take it, at or before every minute the sim flies it."""
    rules, out = sim.Rules(task, world), {}
    for f in task.flights:
        t, d = None, task.earliest(f, world.lb[f.id])
        while d != t:
            t, d = d, f.legal(rules.held(f, d))
        out[f.id] = d + world.block[f.id]
    return out


def brief(row: dict) -> str:
    if "error" in row:
        return "ERROR " + row["error"].strip().splitlines()[-1]
    return (f"bound {row['bound']:.0f} against reference {row['reference']:.0f}, gap {row['gap_pct']}%, valid "
            f"{row['valid']}, {row['wall_s']['total']} s")


def guards(out: Path) -> dict:
    rows = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(out / "dates" / "*.json")))]
    ok = [r for r in rows if "error" not in r]
    gaps = [r["gap_pct"] for r in ok]
    return {"dates": len(ok), "valid": all(r["valid"] for r in ok),
            "invalid": [r["date"] for r in ok if not r["valid"]],
            "gap_pct": {"median": statistics.median(gaps) if gaps else None, "max": max(gaps, default=None)},
            "bounds": {r["date"]: {"bound": r["bound"], "gap_pct": r["gap_pct"]} for r in ok},
            "errors": [r["date"] for r in rows if "error" in r]}


def report(s: dict) -> str:
    lines = ["# v3 stage 2: capacity-aware lower bound (numbers only)", "",
             f"- dates {s['dates']}, every played day at or above its bound: {s['valid']} {s['invalid'] or ''}",
             f"- gap median {s['gap_pct']['median']}%, max {s['gap_pct']['max']}%", "",
             "| date | bound | gap % |", "|---|---|---|"]
    lines += [f"| {d} | {b['bound']} | {b['gap_pct']} |" for d, b in sorted(s["bounds"].items())]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
