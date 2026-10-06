"""Build the dock-v1 packs.

    python -m berth_core.build_dock --split eval --n 100 --out ../tasks/dock-v1-eval
    python -m berth_core.build_dock --split train --n 1000 --out ../tasks/dock-v1-train

Candidates cycle through (tier, window, quay, seed) so tiers stay balanced (n / 4 each) and windows spread over
the split's weeks; a candidate is kept when CP-SAT is within 1 % of its bound and neither the naive re-plan nor a
greedy heuristic comes close (dock.MAX_*_REWARD). Deterministic for the same data and arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .data import load_calls
from .dock import TIERS, build_dock_task, windows

_CALLS = None


def _one(args):
    global _CALLS
    if _CALLS is None:
        _CALLS = load_calls()
    quay, week, n_weeks, tier, seed, split, limit = args
    t0 = time.time()
    try:
        task, stats = build_dock_task(_CALLS, quay, week, n_weeks, tier, seed, split, time_limit=limit, workers=2)
    except Exception as e:  # a generator bug on one candidate is logged, not fatal
        task, stats = None, {"reason": f"error {type(e).__name__}: {e}"}
    stats.update(quay=quay, week=week, n_weeks=n_weeks, tier=tier, seed=seed, seconds=round(time.time() - t0, 1))
    return (task.to_dict() if task else None), stats


def candidates(split: str, tier: str, max_seed: int):
    lo, hi = TIERS[tier]["weeks"]
    out = []
    for seed in range(max_seed):
        for n_weeks in range(lo, hi + 1):
            for w in windows(split, n_weeks):
                for quay in ("36A", "24B"):
                    out.append((quay, w, n_weeks, tier, seed, split))
    key = lambda c: hashlib.sha1(repr(c).encode()).hexdigest()
    return sorted(out, key=lambda c: (c[4], key(c)))   # seed-major, shuffled within a seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["eval", "train"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--max-seed", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_tier = args.n // len(TIERS)
    kept: dict[str, list[dict]] = {t: [] for t in TIERS}
    log = []
    # resume: candidates already tried (partial/log.jsonl) are skipped, tasks already kept (partial/kept.jsonl) reloaded
    part = out / "partial"
    part.mkdir(exist_ok=True)
    done = set()
    if (part / "log.jsonl").is_file():
        for line in (part / "log.jsonl").read_text().splitlines():
            st = json.loads(line)
            log.append(st)
            done.add((st.get("quay"), st.get("week"), st.get("n_weeks"), st.get("tier"), st.get("seed")))
    if (part / "kept.jsonl").is_file():
        for line in (part / "kept.jsonl").read_text().splitlines():
            t = json.loads(line)
            if len(kept[t["difficulty"]]) < per_tier:
                kept[t["difficulty"]].append(t)
    queues = {t: [c for c in candidates(args.split, t, args.max_seed) if c[:5] not in done] for t in TIERS}
    print(f"resuming: {len(log)} candidates tried, " + ", ".join(f"{t} {len(v)}" for t, v in kept.items()), flush=True)
    with ProcessPoolExecutor(args.procs) as ex:
        while any(len(kept[t]) < per_tier for t in TIERS):
            batch = []
            for t in TIERS:
                need = per_tier - len(kept[t])
                for _ in range(min(len(queues[t]), min(need, 20) + 2 if need > 0 else 0)):
                    batch.append(queues[t].pop(0) + (args.time_limit,))
            if not batch:
                break
            for f in as_completed([ex.submit(_one, b) for b in batch]):
                task, stats = f.result()
                log.append(stats)
                with open(part / "log.jsonl", "a") as fh:
                    fh.write(json.dumps(stats) + "\n")
                if task and len(kept[task["difficulty"]]) < per_tier:
                    kept[task["difficulty"]].append(task)
                    with open(part / "kept.jsonl", "a") as fh:
                        fh.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(f"{len(log)} candidates: " + ", ".join(f"{t} {len(v)}/{per_tier}" for t, v in kept.items()), flush=True)
    tasks = sorted((t for v in kept.values() for t in v), key=lambda t: (t["difficulty"], t["quay"], t["week"], t["task_id"]))
    body = "".join(json.dumps(t, ensure_ascii=False, separators=(",", ":")) + "\n" for t in tasks).encode()
    (out / "tasks.jsonl").write_bytes(body)
    counts = {}
    for t in tasks:
        counts[t["difficulty"]] = counts.get(t["difficulty"], 0) + 1
    manifest = {"name": out.name, "split": args.split, "tasks": len(tasks), "tiers": counts,
                "sha256": hashlib.sha256(body).hexdigest(), "candidates": len(log),
                "weeks": sorted({t["week"] for t in tasks}),
                "source": "Port of Barcelona open data, 2024 container calls at quays 36A and 24B (CC BY-SA 4.0)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "build_log.jsonl").write_text("".join(json.dumps(s) + "\n" for s in log))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
