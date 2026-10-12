"""Evaluate every checkpoint a training run saves, as it saves it, and keep the curve.

    python train/jobs/eval_watch.py --run async-grpo-100 --job <training job id> \\
        --bucket FineEnvs/retroenv-rl-runs

Polls <bucket>/<run>/ for checkpoint-N directories (complete once trainer_state.json is there)
and submits one evaluation job per checkpoint: core30 on one H200 with the board's protocol,
output in <bucket>/<run>-eval-<N>/. As evaluations finish it rewrites <bucket>/<run>/evals.md,
pass@1 and reward by step beside the base model's, and prints each new row. It stops once the
training job has ended and every checkpoint has been evaluated. Run it on a laptop, or as a CPU
job with `hf_job.py watch`, which survives the laptop sleeping.
"""

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import huggingface_hub as hub

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_job  # noqa: E402

DONE = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}


def checkpoints(bucket, run):
    """Saved steps whose checkpoint directory is complete."""
    steps = set()
    for item in hub.list_bucket_tree(bucket, prefix=f"{run}/checkpoint-", recursive=True):
        match = re.fullmatch(rf"{re.escape(run)}/checkpoint-(\d+)/trainer_state\.json", getattr(item, "path", ""))
        if match:
            steps.add(int(match.group(1)))
    return sorted(steps)


def summary(bucket, name, scratch):
    """An evaluation's summary.json, or None while it runs."""
    for item in hub.list_bucket_tree(bucket, prefix=f"{name}/", recursive=True):
        path = getattr(item, "path", "")
        if path.endswith("/summary.json") and path.count("/") == 2:
            target = scratch / f"{name}.json"
            hub.download_bucket_files(bucket, [(path, target)])
            return json.loads(target.read_text())
    return None


def table(rows, base):
    lines = [
        "| Step | Pass@1 | Reward | Tool calls | No emit | Tasks |",
        "|---|---|---|---|---|---|",
    ]
    for step, s in [("base", base)] + sorted(rows.items()):
        if s:
            lines.append(
                f"| {step} | {s['pass_at_1']:.3f} | {s['mean_reward']:.3f} | {s['mean_tool_calls']:.1f} | "
                f"{s['no_emit_rate']:.3f} | {s['episodes_graded']}/{s['episodes_expected']} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Training run name (its directory in the bucket)")
    parser.add_argument("--job", required=True, help="Training job id")
    parser.add_argument("--bucket", default="FineEnvs/retroenv-rl-runs")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--set", default="core30")
    parser.add_argument("--base", default="base-eval", help="Evaluation of the untrained model in the same bucket")
    parser.add_argument("--poll", type=int, default=300)
    args = parser.parse_args()
    namespace = args.bucket.split("/")[0]
    scratch = Path(tempfile.mkdtemp())
    submitted, finished = {}, {}
    base = summary(args.bucket, args.base, scratch)
    while True:
        training = hub.inspect_job(job_id=args.job, namespace=namespace).status.stage
        for step in checkpoints(args.bucket, args.run):
            name = f"{args.run}-eval-{step}"
            if step in submitted:
                continue
            if summary(args.bucket, name, scratch):  # evaluated by an earlier watch
                submitted[step] = None
                continue
            job = hf_job.main(
                [
                    "eval", "--name", name, "--bucket", args.bucket, "--model", args.model, "--flavor", "h200",
                    "--set", args.set, "--checkpoint", f"/outputs/{args.run}/checkpoint-{step}",
                    "--label", f"{args.run} step {step}", "--submit",
                ]
            )  # fmt: skip
            submitted[step] = job.id
            print(f"submitted evaluation of step {step}: {job.url}", flush=True)
        for step, job_id in submitted.items():
            if step in finished:
                continue
            result = summary(args.bucket, f"{args.run}-eval-{step}", scratch)
            stage = hub.inspect_job(job_id=job_id, namespace=namespace).status.stage if job_id else "COMPLETED"
            if result or stage in DONE:
                finished[step] = result
                row = f"pass@1 {result['pass_at_1']:.3f}, reward {result['mean_reward']:.3f}" if result else stage
                print(f"step {step}: {row}", flush=True)
                curve = scratch / "evals.md"
                curve.write_text(f"# {args.run} on {args.set}\n\n" + table(finished, base))
                hub.batch_bucket_files(args.bucket, add=[(str(curve), f"{args.run}/evals.md")])
        if (
            training in DONE
            and len(finished) == len(submitted)
            and not set(checkpoints(args.bucket, args.run)) - set(submitted)
        ):
            print(table(finished, base), flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
