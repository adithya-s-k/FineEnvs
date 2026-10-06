"""Markdown board for a run: mean reward with a 95 % bootstrap CI, per difficulty, feasibility and how often the
plan beat the naive re-plan or matched the optimum.

    python eval/report.py board
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("easy", "medium", "hard", "expert", "standard", "busy", "storm", "extreme", "frontier")


def ci(xs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main(run: str):
    eps = json.loads((ROOT / "results" / "rollouts" / run / "index.json").read_text())["episodes"]
    by = defaultdict(list)
    for e in eps:
        by[e["model"]].append(e)
    rows = sorted(by.items(), key=lambda kv: -statistics.mean(e["reward"] for e in kv[1]))
    LEVELS = [lv for lv in ORDER if any(e["difficulty"] == lv for e in eps)]
    out = ["| Model | n | Mean reward (95% CI) | " + " | ".join(LEVELS) + " | Feasible | Beat naive | Optimal | Median turns |",
           "|---|---:|---|" + "---:|" * len(LEVELS) + "---:|---:|---:|---:|"]
    for m, es in rows:
        r = [e["reward"] for e in es]
        lo, hi = ci(r)
        per = []
        for d in LEVELS:
            xs = [e["reward"] for e in es if e["difficulty"] == d]
            per.append(f"{statistics.mean(xs):.3f}" if xs else "-")
        feas = sum(e["feasible"] for e in es)
        beat = sum(1 for e in es if e["feasible"] and e["cost"] is not None and e["cost"] < e["naive_cost"])
        opt = sum(1 for e in es if e["feasible"] and e["cost"] == e["optimal_cost"])
        out.append(f"| {m} | {len(es)} | **{statistics.mean(r):.3f}** ({lo:.3f}-{hi:.3f}) | " + " | ".join(per)
                   + f" | {100 * feas / len(es):.0f}% | {100 * beat / len(es):.0f}% | {100 * opt / len(es):.0f}% | "
                   f"{statistics.median(e['turns'] for e in es):.0f} |")
    tokens = {m: (sum(e["input_tokens"] for e in es), sum(e["output_tokens"] for e in es)) for m, es in rows}
    out += ["", "| Model | input tokens | output tokens | median seconds | ended without submit |", "|---|---:|---:|---:|---:|"]
    for m, es in rows:
        ti, to = tokens[m]
        out.append(f"| {m} | {ti:,} | {to:,} | {statistics.median(e['seconds'] for e in es):.0f} | "
                   f"{sum(not e['submitted'] for e in es)} |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "board")
