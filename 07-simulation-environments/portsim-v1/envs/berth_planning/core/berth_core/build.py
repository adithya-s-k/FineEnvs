"""Build the task pack.

    python -m berth_core.build --out ../tasks/berth-v1 --target 100

Every (quay, ISO week 2-51, difficulty) with seed 0 is a candidate; seeds 1, 2, ... are added until enough pass the
filter. Levels: easy, medium, hard (closures, late ships, crane overruns, extra calls) and expert (the other quay's
traffic diverted here on top). A candidate is kept when the CP-SAT optimum is proven and the naive re-plan leaves room to improve (naive cost
at least MIN_RATIO x optimum and MIN_GAP above it). Weeks divisible by 5 form the test split. Selection is
deterministic: the same data and arguments rebuild the same pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .data import QUAYS, load_calls
from .generate import build_task

MIN_RATIO = 1.15
MIN_GAP = 15
TEST_WEEK_MOD = 5
TEST_SHARE = 0.2

_CALLS = None


def _one(args):
    global _CALLS
    if _CALLS is None:
        _CALLS = load_calls()
    quay, week, seed, difficulty, time_limit = args
    split = "test" if week % TEST_WEEK_MOD == 0 else "train"
    t = time.time()
    task, stats = build_task(_CALLS, quay, week, seed, difficulty, split, time_limit=time_limit, workers=2)
    stats.update(quay=quay, week=week, seed=seed, difficulty=difficulty, seconds=round(time.time() - t, 1))
    if task is not None:
        naive, opt = stats["naive"], stats["optimal"]
        if naive < MIN_RATIO * opt or naive - opt < MIN_GAP:
            stats["reason"] = f"naive {naive} too close to optimum {opt}"
            task = None
    return (task.to_dict() if task else None), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--max-seeds", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--levels", default="easy,medium,hard,expert")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept: dict[str, dict] = {}
    log = []
    levels = args.levels.split(",")
    per_difficulty = {d: args.target // len(levels) + (1 if i < args.target % len(levels) else 0) for i, d in enumerate(levels)}
    for seed in range(args.max_seeds):
        need = {d: per_difficulty[d] - sum(1 for t in kept.values() if t["difficulty"] == d) for d in per_difficulty}
        if all(v <= 0 for v in need.values()):
            break
        jobs = [(q, w, seed, d, args.time_limit) for q in QUAYS for w in range(2, 52) for d in per_difficulty if need[d] > 0]
        with ProcessPoolExecutor(args.procs) as ex:
            futs = [ex.submit(_one, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                task, stats = f.result()
                log.append(stats)
                if task:
                    kept[task["task_id"]] = task
                if i % 25 == 0:
                    print(f"seed {seed}: {i}/{len(jobs)} candidates, {len(kept)} kept", flush=True)
    # Deterministic selection spread over the year: within each level, test and train pools ordered by a hash of
    # the task id; TEST_SHARE of each level comes from the held-out weeks.
    chosen = []
    key = lambda t: hashlib.sha1(t["task_id"].encode()).hexdigest()
    for d, n in per_difficulty.items():
        pool = [t for t in kept.values() if t["difficulty"] == d]
        test = sorted((t for t in pool if t["split"] == "test"), key=key)
        train = sorted((t for t in pool if t["split"] == "train"), key=key)
        n_test = min(len(test), round(n * TEST_SHARE))
        chosen += test[:n_test] + train[: n - n_test]
    chosen.sort(key=lambda t: (t["split"], t["quay"], t["week"], t["difficulty"]))
    body = "".join(json.dumps(t, ensure_ascii=False, separators=(",", ":")) + "\n" for t in chosen).encode()
    (out / "tasks.jsonl").write_bytes(body)
    splits = {}
    for t in chosen:
        splits.setdefault(t["split"], {}).setdefault(t["difficulty"], 0)
        splits[t["split"]][t["difficulty"]] += 1
    manifest = {
        "name": out.name, "tasks": len(chosen), "splits": splits, "sha256": hashlib.sha256(body).hexdigest(),
        "source": "Port of Barcelona open data, 2024 container calls at quays 36A and 24B (CC BY-SA 4.0)",
        "filter": {"min_ratio": MIN_RATIO, "min_gap": MIN_GAP, "test_weeks": f"ISO week % {TEST_WEEK_MOD} == 0"},
        "candidates": len(log), "kept_before_selection": len(kept),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "build_log.jsonl").write_text("".join(json.dumps(s) + "\n" for s in log))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
