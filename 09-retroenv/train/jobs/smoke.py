"""Run two updates, check the saved checkpoints, then reload the adapter and evaluate it on two tasks."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sync", "async"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--space-id")
    parser.add_argument("--vllm-gpus", type=int)
    args, _ = parser.parse_known_args()  # the launcher's shared options are ignored here
    output = Path(args.output).resolve()
    train, evaluation = output / "train", output / "reload-eval"
    common = ["--mode", args.mode, "--model", args.model] + (["--space-id", args.space_id] if args.space_id else [])
    run = [sys.executable, str(ROOT / "train/jobs/run.py")]
    layout = ["--vllm-gpus", str(args.vllm_gpus)] if args.vllm_gpus else []
    subprocess.run([*run, "train", *common, *layout, "--smoke", "--output", str(train)], check=True, cwd=ROOT)

    for step in (1, 2):
        path = train / f"checkpoint-{step}"
        state = json.loads((path / "trainer_state.json").read_text())
        assert state["global_step"] == step, f"{path}: global_step {state['global_step']}"
        for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt"):
            assert (path / name).stat().st_size > 0, f"{path} is missing {name}"
    updates = [row for row in state["log_history"] if "loss" in row or "grad_norm" in row]
    assert updates, "No optimizer metrics were saved"
    for row in updates:
        for key in ("loss", "grad_norm"):
            assert key not in row or math.isfinite(row[key]), f"Non-finite {key}: {row}"
    assert any(p.stat().st_size for p in (train / "trackio").rglob("*") if p.suffix in {".db", ".jsonl"}), (
        "No local Trackio metrics"
    )
    # A run whose groups all tie trains nothing. Two steps of four rollouts must move the weights.
    assert any(row.get("grad_norm", 0) > 0 for row in updates), f"Every update had zero gradient: {updates}"
    episodes = [json.loads(line) for line in (train / "episodes.jsonl").read_text().splitlines() if line]
    assert episodes, "No episode traces were written"

    checkpoint = train / "checkpoint-2"
    subprocess.run(
        [
            *run,
            "eval",
            *common,
            "--checkpoint",
            str(checkpoint),
            "--tasks",
            str(args.tasks),
            "--output",
            str(evaluation),
        ],
        check=True,
        cwd=ROOT,
    )
    summaries = list(evaluation.glob("*/summary.json"))
    assert summaries, "The evaluation wrote no summary"
    summary = json.loads(summaries[0].read_text())
    assert summary["episodes_graded"] == summary["episodes_expected"], f"Ungraded episodes: {summary}"
    report = {
        "mode": args.mode,
        "model": args.model,
        "checkpoint": str(checkpoint),
        "optimizer_metrics": updates,
        "nonzero_gradient_updates": sum(row.get("grad_norm", 0) > 0 for row in updates),
        "episodes": len(episodes),
        "submitted": sum(e["submitted"] for e in episodes) / len(episodes),
        "rewards": [e["reward"] for e in episodes],
        "eval": {key: summary.get(key) for key in ("tasks", "pass_at_1", "mean_reward", "mean_tool_calls")},
        "job": os.getenv("HF_JOB_ID") or os.getenv("JOB_ID"),
        "complete": True,
    }
    (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
