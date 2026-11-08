"""Summarize Harbor jobs on QED-Nano tasks: one row per trial, then per job.

    python -m fineproof_harbor.report jobs/val16-oracle jobs/val16-qwen9b --out summary.json

Per trial: the reward (rubric score / 7), whether it was graded, the grader's reason, and the problem's base
difficulty (QED-Nano's mean over 128 Qwen3-4B-Thinking attempts, from task.toml), so a run can be read against it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tomllib
from pathlib import Path


def trials(job: Path):
    for trial in sorted(p for p in job.iterdir() if p.is_dir() and (p / "result.json").exists()):
        result = json.loads((trial / "result.json").read_text())
        if not result.get("finished_at"):
            continue
        grading = trial / "verifier" / "grading.json"
        g = json.loads(grading.read_text()) if grading.exists() else {}
        task_dir = Path(result["task_id"]["path"])
        if not task_dir.is_absolute():  # Harbor records the path as given, relative to where it was run
            task_dir = next((base / task_dir for base in (Path.cwd(), *job.resolve().parents)
                             if (base / task_dir / "task.toml").exists()), task_dir)
        meta = tomllib.loads((task_dir / "task.toml").read_text()).get("metadata", {}) if task_dir.exists() else {}
        reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        exc = result.get("exception_info") or {}
        yield {
            "task": task_dir.name,
            "reward": reward,
            "scores": g.get("scores"),
            "reason": (g.get("reason") or exc.get("exception_type") or "")[:160],
            "proof_chars": g.get("proof_chars"),
            "base_reward": meta.get("reward_mean"),
        }


def summarize(rows: list[dict]) -> dict:
    graded = [r for r in rows if r["reward"] is not None]
    mean = (lambda xs: round(statistics.fmean(xs), 4) if xs else None)
    return {
        "trials": len(rows),
        "graded": len(graded),
        "mean_reward": mean([r["reward"] for r in graded]),
        "full_marks": mean([float(r["reward"] >= 0.999) for r in graded]),
        "any_credit": mean([float(r["reward"] > 0) for r in graded]),
        "mean_base_reward": mean([r["base_reward"] for r in graded if r["base_reward"] is not None]),
        "ungraded": [f"{r['task']}: {r['reason']}" for r in rows if r["reward"] is None],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = {}
    for job in args.jobs:
        rows = list(trials(job))
        report[job.name] = {"summary": summarize(rows), "rows": rows}
        print(f"== {job.name}: {json.dumps(report[job.name]['summary'])}")
        for r in rows:
            reward = "—" if r["reward"] is None else f"{r['reward']:.3f}"
            base = "—" if r["base_reward"] is None else f"{r['base_reward']:.2f}"
            print(f"   {r['task']:18} reward {reward:>6}  base {base:>5}  {r['reason'][:70]}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
