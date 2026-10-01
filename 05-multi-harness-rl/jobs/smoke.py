"""Run two updates, verify the saved checkpoint, then reload it for pass@1 evaluation."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["whitebox", "opencode", "multi_harness"], required=True
    )
    parser.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--space-id")
    # Accept the launcher's shared options; the smoke always uses two updates.
    parser.add_argument("--steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = output / "train"
    evaluation = output / "reload-eval"
    checkpoint = train / "checkpoint-2"
    common = ["--mode", args.mode, "--model", args.model]
    if args.space_id:
        common += ["--space-id", args.space_id]
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "jobs/run.py"),
            "train",
            *common,
            "--smoke",
            "--output",
            str(train),
        ],
        check=True,
        cwd=ROOT,
    )
    for step in (1, 2):
        path = train / f"checkpoint-{step}"
        state = json.loads((path / "trainer_state.json").read_text())
        assert state["global_step"] == step
        assert list(path.glob("*.safetensors")), "No saved model weights"
        for filename in (
            "config.json",
            "tokenizer_config.json",
            "optimizer.pt",
            "scheduler.pt",
        ):
            assert (path / filename).stat().st_size > 0, (
                f"Missing checkpoint file: {filename}"
            )
    metrics = state.get("log_history", [])
    updates = [row for row in metrics if "loss" in row or "grad_norm" in row]
    assert updates, "No optimizer metrics were saved"
    for row in updates:
        for key in ("loss", "grad_norm"):
            if key in row:
                assert math.isfinite(row[key]), f"Non-finite {key}"
    assert any(
        p.stat().st_size
        for p in (train / "trackio").rglob("*")
        if p.suffix in {".db", ".jsonl"}
    ), "No local Trackio metrics"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "jobs/run.py"),
            "eval",
            *common,
            "--checkpoint",
            str(checkpoint),
            "--step",
            "2",
            "--tasks",
            str(args.tasks),
            "--concurrency",
            str(args.concurrency),
            "--output",
            str(evaluation),
        ],
        check=True,
        cwd=ROOT,
    )
    summary = json.loads((evaluation / "summary.json").read_text())
    assert summary["graded"] == summary["expected"]
    report = {
        "mode": args.mode,
        "model": args.model,
        "checkpoint": str(checkpoint),
        "steps": 2,
        "optimizer_metrics": updates,
        "nonzero_gradient_updates": sum(row.get("grad_norm", 0) > 0 for row in updates),
        "eval": summary,
        "dependencies": json.loads((train / "dependencies.json").read_text())
        if (train / "dependencies.json").exists()
        else None,
        "job": os.getenv("SLURM_JOB_ID") or os.getenv("HF_JOB_ID"),
        "complete": True,
    }
    (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
