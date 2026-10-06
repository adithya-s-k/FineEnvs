"""Build the frontier tier: congested fortnights with quay cranes and a movement limit.

    python -m berth_core.build_frontier --out ../tasks/berth-frontier-v0 --n 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .build import TEST_WEEK_MOD
from .data import load_calls
from .generate import build_frontier_task

_CALLS = None


def _one(args):
    global _CALLS
    if _CALLS is None:
        _CALLS = load_calls()
    quay, week, seed, limit = args
    split = "test" if week % TEST_WEEK_MOD == 0 else "train"
    task, stats = build_frontier_task(_CALLS, quay, week, seed, split, time_limit=limit, workers=2)
    stats.update(quay=quay, week=week, seed=seed)
    return (task.to_dict() if task else None), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--weeks", default="")
    ap.add_argument("--time-limit", type=float, default=180.0)
    ap.add_argument("--procs", type=int, default=4)
    args = ap.parse_args()
    weeks = [int(w) for w in args.weeks.split(",")] if args.weeks else list(range(2, 51, 2))
    jobs = [(q, w, 0, args.time_limit) for w in weeks for q in ("36A", "24B")]
    with ProcessPoolExecutor(args.procs) as ex:
        results = list(ex.map(_one, jobs))
    kept = [t for t, _ in results if t]
    kept = sorted(kept, key=lambda t: hashlib.sha1(t["task_id"].encode()).hexdigest())[: args.n]
    kept.sort(key=lambda t: (t["split"], t["quay"], t["week"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(t, ensure_ascii=False, separators=(",", ":")) + "\n" for t in kept).encode()
    (out / "tasks.jsonl").write_bytes(body)
    manifest = {"name": out.name, "tasks": len(kept), "sha256": hashlib.sha256(body).hexdigest(),
                "candidates": len(jobs), "kept_before_selection": sum(1 for t, _ in results if t),
                "source": "Port of Barcelona open data, 2024 container calls at quays 36A and 24B (CC BY-SA 4.0)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "build_log.jsonl").write_text("".join(json.dumps(s) + "\n" for _, s in results))
    print(json.dumps(manifest, indent=2))
    for t in kept:
        r = t["reference"]
        print(t["task_id"], len(t["ships"]), "ships, naive", r["naive_cost"], "opt", r["optimal_cost"], "proven", r["proven_optimal"])


if __name__ == "__main__":
    main()
