"""Two updates with a simulated crash in between, a third in a resumed run, then reload the last checkpoint and score it.

The trainer dies right after checkpoint 1 (FINEPROOF_CRASH_AFTER_STEP=1), as a leak would kill it; jobs/run.py must
restart everything from that checkpoint and finish update 2. The run's thread census must have recorded. In an HF Job,
the run is then copied to the bucket and deleted locally, as when a Job ends, and run again for a third update:
jobs/run.py must restore it from the bucket (state, logs, latest full checkpoint) and continue from checkpoint 2.

Adapted from 05-multi-harness-rl/jobs/smoke.py; a judge score is fractional, so "graded" means a finite reward.
"""

import argparse
import json
import math
import os
import shutil
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
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train, evaluation = output / "train", output / "reload-eval"
    common = ["--model", args.model, "--sandbox", args.sandbox] + (["--space-id", args.space_id] if args.space_id else [])
    for flag, value in (("--rounds", args.rounds), ("--max-tokens", args.max_tokens)):
        if value is not None:
            common += [flag, str(value)]
    train_extra = ["--learning-rate", str(args.learning_rate)] if args.learning_rate else []

    def train_to(steps, env=None):
        subprocess.run([sys.executable, str(ROOT / "jobs/run.py"), "train", *common, *train_extra, "--smoke",
                        "--steps", str(steps), "--output", str(train)], check=True, cwd=ROOT,
                       env={**os.environ, **(env or {})})

    def loss_lines():
        return sum("{'loss'" in line for line in (train / "train.log").read_text(errors="replace").splitlines())

    train_to(2, {"FINEPROOF_CRASH_AFTER_STEP": "1"})
    restarts = [json.loads(line) for line in (train / "restarts.jsonl").read_text().splitlines()]
    assert len(restarts) == 1 and restarts[0]["from"] == "checkpoint-1", f"Expected one resume from checkpoint-1: {restarts}"
    census = [json.loads(line) for line in (train / "threads.jsonl").read_text().splitlines()]
    assert census and census[-1]["threads"] > 0, "The thread census recorded nothing"
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
    # The objective in objective.py and tasks.py is the one that ran.
    objective = json.loads((train / "training_config.json").read_text())["objective"]
    assert objective["loss_norm_tokens"] == (args.max_tokens or 32768) and objective["truncated_reward"] == "zero"
    rollouts = [json.loads(line) for line in (train / "rollouts.jsonl").read_text().splitlines() if line.strip()]
    graded = [r for r in rollouts if isinstance(r.get("reward"), (int, float))]
    assert graded and all("truncated" in r for r in graded), "rollouts.jsonl does not say which proofs were cut off"
    assert all(r["reward"] == 0.0 for r in graded if r["truncated"]), "A cut-off proof was rewarded"
    resumed = None
    if os.environ.get("FINEPROOF_OUTPUT_BUCKET") and train.is_relative_to("/workspace/outputs"):
        # A Job that ends early: its outputs are in the bucket and its disk is gone. Submitting it again continues it.
        url = f"hf://buckets/{os.environ['FINEPROOF_OUTPUT_BUCKET']}/{train.relative_to('/workspace/outputs').as_posix()}"
        subprocess.run(["hf", "buckets", "sync", str(train), url, "--format", "quiet"], check=True)
        before = {"loss_lines": loss_lines(), "rollouts": len(rollouts)}
        shutil.rmtree(train)
        train_to(3)
        resumes = [json.loads(line) for line in (train / "resumes.jsonl").read_text().splitlines()]
        assert len(resumes) == 1 and resumes[0]["from"] == "checkpoint-2", f"Expected a resume from checkpoint-2: {resumes}"
        state = json.loads((train / "checkpoint-3" / "trainer_state.json").read_text())
        assert state["global_step"] == 3
        after = {"loss_lines": loss_lines(),
                 "rollouts": len((train / "rollouts.jsonl").read_text().splitlines())}
        assert after["loss_lines"] > before["loss_lines"] and after["rollouts"] > before["rollouts"], (
            f"The resumed run's logs did not continue the first one's: {before} -> {after}")
        assert len((train / "restarts.jsonl").read_text().splitlines()) == 1, "The first run's restart record was lost"
        resumed = {"resumes": resumes, "before": before, "after": after}
    last = 3 if resumed else 2
    subprocess.run([sys.executable, str(ROOT / "jobs/run.py"), "eval", *common, "--checkpoint",
                    str(train / f"checkpoint-{last}"), "--step", str(last), "--tasks", str(args.tasks),
                    "--concurrency", str(args.concurrency), "--output", str(evaluation)], check=True, cwd=ROOT)
    summary = json.loads((evaluation / "summary.json").read_text())
    assert summary["graded"] == summary["expected"]
    rewards = [row.get("reward") for row in updates if "reward" in row]
    report = {
        "model": args.model,
        "restarts": restarts,
        "resumed_from_bucket": resumed,
        "objective": objective,
        "truncated_rollouts": sum(r["truncated"] for r in graded),
        "graded_rollouts": len(graded),
        "threads_last": census[-1],
        "sandbox": args.sandbox,
        "steps": last,
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
