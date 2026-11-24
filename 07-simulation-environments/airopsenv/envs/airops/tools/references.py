"""The references of each task in a pack, and the clauses that decide which windows become eval tasks.

For every task: the floor and the hindsight optimum (refs.mip), the follower and the blind follower (refs.follower),
naive and the airline's own day (refs.policies), all run on the truth with core.sim.execute. The anchor is hindsight.
A task qualifies when every flight has a tail number, weather and ATC are at least 80% of the coded cancellations
((B+C)/(A+B+C)), naive is at least 10% above the anchor, the follower at most 15% above hindsight, hindsight is proven
to 0.01% within 60 s, and it has 30-80 tails.

    python tools/references.py screen tasks/b6-v1/tasks.jsonl [--tasks ID,...] [--workers N] [--json OUT]
    python tools/references.py references tasks/b6-v1/tasks.jsonl [--tasks ID,...] [--workers N]

``screen`` prints the table (and writes it with timings to OUT); ``references`` writes each task's reference block,
rewriting the file one compact JSON line per task.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from airops_core.core.model import Task
from airops_core.core.sim import execute
from airops_core.refs import follower, mip, policies

PROVE_SECONDS = 60.0
WEATHER_ATC = 0.8
NAIVE_MARGIN = 0.10
FOLLOWER_MARGIN = 0.15
TAILS = (30, 80)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["screen", "references"])
    ap.add_argument("tasks", type=Path)
    ap.add_argument("--tasks", dest="only", default="", help="comma-separated task ids (default: all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--json", type=Path, help="screen: write the rows here too")
    args = ap.parse_args()
    raw = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    only = set(filter(None, args.only.split(",")))
    picked = [d for d in raw if not only or d["task_id"] in only]
    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(zip([d["task_id"] for d in picked], pool.map(compute, picked), strict=True))
    if args.command == "screen":
        rows = [{"task_id": tid, **r["timing"], **{k: v for k, v in r["reference"].items() if not k.endswith("_plan")}}
                for tid, r in results.items()]
        print_table(rows)
        if args.json:
            args.json.write_text(json.dumps(rows, indent=1) + "\n")
        return
    for d in raw:
        if d["task_id"] in results:
            d["reference"] = results[d["task_id"]]["reference"]
    args.tasks.write_text("".join(json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n" for d in raw))


def compute(d: dict) -> dict:
    task = Task.from_dict(d)
    t0 = time.monotonic()
    hindsight = mip.solve(task, time_limit=PROVE_SECONDS)
    floor = mip.solve(task, floor=True, time_limit=PROVE_SECONDS)
    t1 = time.monotonic()
    followed = follower.follow(task, follower.PERSIST)
    t2 = time.monotonic()
    blind = follower.follow(task, follower.BLIND)
    t3 = time.monotonic()
    anchor = hindsight.objective
    costs = {"floor": floor.objective, "hindsight": hindsight.objective, "anchor": anchor,
             "follower": followed.cost, "blind": blind.cost, "naive": execute(task, policies.naive(task)).cost.total,
             "airline": execute(task, policies.airline(task)).cost.total}
    checks = clauses(task, costs, hindsight)
    reference = {**costs, "gap_pct": hindsight.gap_pct, "qualifies": all(c["ok"] for c in checks.values()),
                 "clauses": checks, "hindsight_plan": hindsight.stored(), "follower_plan": followed.stored()}
    timing = {"hindsight_s": hindsight.wall_s, "floor_s": floor.wall_s, "refs_s": round(t1 - t0, 1),
              "follower_s": round(t2 - t1, 1), "blind_s": round(t3 - t2, 1), "flights": len(task.flights),
              "tails": len(task.tails),
              "unproven_watches": sum(w["status"] != "Optimal" for w in followed.watches + blind.watches)}
    return {"reference": reference, "timing": timing}


def clauses(task: Task, costs: dict, hindsight: mip.Solution) -> dict[str, dict]:
    codes = [f.actual.cancel_code for f in task.flights
             if f.actual.cancelled and f.actual.cancel_code in ("A", "B", "C")]
    share = round(sum(c in "BC" for c in codes) / len(codes), 3) if codes else None
    untailed = sum(not f.actual.tail for f in task.flights)
    naive = round(costs["naive"] / costs["anchor"] - 1, 4)
    follow = round(costs["follower"] / costs["hindsight"] - 1, 4)
    return {"tail_numbers": {"value": untailed, "ok": untailed == 0},
            "weather_atc": {"value": share, "ok": share is not None and share >= WEATHER_ATC},
            "naive_over_anchor": {"value": naive, "ok": naive >= NAIVE_MARGIN},
            "follower_over_hindsight": {"value": follow, "ok": follow <= FOLLOWER_MARGIN},
            "hindsight_proven": {"value": hindsight.gap_pct,
                                 "ok": hindsight.proven and hindsight.wall_s <= PROVE_SECONDS},
            "tails": {"value": len(task.tails), "ok": TAILS[0] <= len(task.tails) <= TAILS[1]}}


def print_table(rows: list[dict]) -> None:
    head = ["task_id", "tails", "flights", "floor", "hindsight", "follower", "blind", "naive", "airline", "gap_pct",
            "hindsight_s", "follower_s", "blind_s", "unproven_watches", "qualifies"]
    print("\t".join(head + ["failed"]))
    for r in rows:
        failed = ",".join(k for k, c in r["clauses"].items() if not c["ok"])
        print("\t".join(str(r[k]) for k in head) + "\t" + failed)


if __name__ == "__main__":
    main()
