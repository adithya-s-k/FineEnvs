"""Two updates, check the saved checkpoints, then reload checkpoint 2 and score it on a few test problems.

Adapted from 05-multi-harness-rl/jobs/smoke.py; a judge score is fractional, so "graded" means a finite reward.
"""

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
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--space-id")
    parser.add_argument("--sandbox", default="daytona", choices=["daytona", "hf-sandbox"],
                        help="Where Harbor runs each task's sandboxes (hf-sandbox: HF Jobs, needs a docker_image task)")
    # Accept the launcher's shared options; the smoke always runs two updates.
    for option in ("--steps", "--save-steps", "--step", "--samples"):
        parser.add_argument(option, type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train, evaluation = output / "train", output / "reload-eval"
    common = ["--model", args.model, "--sandbox", args.sandbox] + (["--space-id", args.space_id] if args.space_id else [])
    for flag, value in (("--rounds", args.rounds), ("--max-tokens", args.max_tokens)):
        if value is not None:
            common += [flag, str(value)]
    subprocess.run([sys.executable, str(ROOT / "jobs/run.py"), "train", *common, "--smoke", "--output", str(train)],
                   check=True, cwd=ROOT)
    for step in (1, 2):
        path = train / f"checkpoint-{step}"
        state = json.loads((path / "trainer_state.json").read_text())
        assert state["global_step"] == step
        assert list(path.glob("*.safetensors")), "No saved model weights"
    updates = [row for row in state.get("log_history", []) if "loss" in row or "grad_norm" in row]
    assert updates, "No optimizer metrics were saved"
    for row in updates:
        for key in ("loss", "grad_norm"):
            if key in row:
                assert math.isfinite(row[key]), f"Non-finite {key}"
    # Checked before the eval, so a run that cannot learn fails without paying for one.
    assert any(row.get("grad_norm", 0) > 0 for row in updates), (
        "No update had a gradient: every group's rewards tied, so this model and these problems give GRPO no signal")
    ratios = [row["ratio"] for row in updates if "ratio" in row]
    assert all(abs(r - 1) < 0.05 for r in ratios), (
        f"Importance ratios {ratios} are far from 1: the captured logprobs disagree with the trainer's")
    subprocess.run([sys.executable, str(ROOT / "jobs/run.py"), "eval", *common, "--checkpoint",
                    str(train / "checkpoint-2"), "--step", "2", "--tasks", str(args.tasks), "--concurrency",
                    str(args.concurrency), "--output", str(evaluation)], check=True, cwd=ROOT)
    summary = json.loads((evaluation / "summary.json").read_text())
    assert summary["graded"] == summary["expected"]
    rewards = [row.get("reward") for row in updates if "reward" in row]
    report = {
        "model": args.model,
        "sandbox": args.sandbox,
        "steps": 2,
        "optimizer_metrics": updates,
        "train_reward_logged": rewards,
        "nonzero_gradient_updates": sum(row.get("grad_norm", 0) > 0 for row in updates),
        "eval": summary,
        "job": os.getenv("SLURM_JOB_ID") or os.getenv("HF_JOB_ID"),
        "complete": True,
    }
    (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
