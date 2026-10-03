#!/usr/bin/env python3
"""Pull one run's curves out of the Trackio database into the figure's data file.

The database is the one the training job wrote to, so these are the numbers the
run actually logged rather than anything retyped. Download it first:

    hf download --repo-type bucket FineEnvs/geoguesser-trackio-bucket \
        trackio/geoguesser.db --local-dir .

Per-step training reward is one task per optimizer step at temperature 1.0, so
it swings between 0.01 and 0.99 and is unreadable raw. An exponential moving
average is stored alongside it; the raw series stays in the file so the figure
can show both and nobody has to trust the smoothing.

    python scripts/build_run1_curve.py geoguesser.db --run run1-4b
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

# Metrics kept for the readout, mapped to the short names the figure uses.
EXTRA = {
    "train/reward_std": "reward_std",
    "train/entropy": "entropy",
    "train/tools/call_frequency": "tool_calls",
    "train/completions/mean_length": "completion_tokens",
    "train/grad_norm": "grad_norm",
}


def ema(values: list[float], span: int) -> list[float]:
    alpha = 2 / (span + 1)
    out, acc = [], values[0]
    for v in values:
        acc = alpha * v + (1 - alpha) * acc
        out.append(round(acc, 4))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db", type=Path, help="trackio sqlite database")
    ap.add_argument("--run", default="run1-4b")
    ap.add_argument("--span", type=int, default=50, help="EMA span, in steps")
    ap.add_argument(
        "--progression",
        type=Path,
        default=Path("app/src/content/assets/data/run1-progression.json"),
        help="per-checkpoint eval scores, for the second series",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("app/src/content/assets/data/run1-curve.json")
    )
    args = ap.parse_args()

    db = sqlite3.connect(args.db)
    rows: dict[int, dict] = {}
    for step, blob in db.execute(
        "select step, metrics from metrics where run_name=?", (args.run,)
    ):
        rows.setdefault(step, {}).update(json.loads(blob))

    # `step` counts logging events; `train/global_step` is the optimizer step
    # the checkpoints are named after, so the two series share an axis.
    to_global = {s: d["train/global_step"] for s, d in rows.items() if "train/global_step" in d}

    train = []
    for step in sorted(rows):
        d = rows[step]
        if "train/reward" not in d:
            continue
        gs = to_global.get(step, to_global.get(step + 1))
        if gs is None:
            continue
        point = {"step": int(gs), "reward": round(d["train/reward"], 4)}
        for key, short in EXTRA.items():
            if key in d:
                point[short] = round(d[key], 4)
        train.append(point)
    train.sort(key=lambda p: p["step"])

    smooth = ema([p["reward"] for p in train], args.span)
    for p, s in zip(train, smooth):
        p["smooth"] = s

    prog = json.loads(args.progression.read_text())
    evals = [
        {
            "step": row["step"],
            "mean": row["mean"],
            "best": row["best"],
            "median_km": row["median_km"],
            "turns": row["turns"],
        }
        for row in prog
    ]
    evals.sort(key=lambda r: r["step"])
    baseline = next(r["mean"] for r in evals if r["step"] == 0)

    payload = {
        "run": args.run,
        "span": args.span,
        "baseline": baseline,
        "train": train,
        "eval": evals,
    }
    args.out.write_text(json.dumps(payload))
    print(
        f"{len(train)} training steps, {len(evals)} eval points, "
        f"baseline {baseline} -> {args.out} "
        f"({args.out.stat().st_size / 1024:.0f} KiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
