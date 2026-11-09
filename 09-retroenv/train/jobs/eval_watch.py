"""Evaluate every checkpoint a training run saves, as it saves it, and keep the curve.

    python train/jobs/eval_watch.py --run <run> --job <training job id> --bucket <owner>/<bucket> \\
        --set core150 --harness training --attempts 4 --temperature 0.8 --shards 5 --base base-core150-th

Polls <bucket>/<run>/ for checkpoint-N directories (complete once trainer_state.json is there) and
evaluates each one as --shards jobs on one H200 each: the set's tasks are cut into contiguous
slices in data/eval-<set>.json order (five slices of core150 make the first one core30), so a
checkpoint takes about as long as one slice. An evaluation is mostly waiting on long episodes that
share a GPU (core150 at 4 attempts on one H200: median episode 4 minutes, the slowest 26), so
spreading it over GPUs shortens it more than raising concurrency on one. Outputs go to
<bucket>/<run>-eval-<N>-<k>/ (one job: <run>-eval-<N>/).

Once every slice of a checkpoint is in, its episodes are scored together: pass@1 over every
attempt, pass@k with k = --attempts (a task counts if any attempt solved it), mean reward, and
pass@1 by difficulty tier, written to <bucket>/<run>/evals.md beside the base model's row
(--base, an evaluation of the untrained model in the same bucket). The watch stops once the
training job has ended and every checkpoint is scored. Run it on a laptop, or as a CPU job with
`hf_job.py watch`, which survives the laptop sleeping.
"""

import argparse
import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import huggingface_hub as hub

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_job  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DONE = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}


def checkpoints(bucket, run):
    """Saved steps whose checkpoint directory is complete."""
    steps = set()
    for item in hub.list_bucket_tree(bucket, prefix=f"{run}/checkpoint-", recursive=True):
        match = re.fullmatch(rf"{re.escape(run)}/checkpoint-(\d+)/trainer_state\.json", getattr(item, "path", ""))
        if match:
            steps.add(int(match.group(1)))
    return sorted(steps)


def has_summary(bucket, name):
    return any(
        getattr(item, "path", "").endswith("/summary.json") and item.path.count("/") == 2
        for item in hub.list_bucket_tree(bucket, prefix=f"{name}/", recursive=True)
    )


def score(bucket, names, tiers, attempts, scratch):
    """Pass@1, pass@k, reward and pass@1 by tier over the episodes of one or more evaluations."""
    paths = [
        item.path
        for name in names
        for item in hub.list_bucket_tree(bucket, prefix=f"{name}/", recursive=True)
        if "/episodes/" in getattr(item, "path", "") and item.path.endswith(".json")
    ]
    if not paths:
        return None
    local = scratch / "episodes"
    local.mkdir(parents=True, exist_ok=True)
    hub.download_bucket_files(bucket, [(path, local / path.replace("/", "__")) for path in paths])
    by_task, rewards = defaultdict(list), []
    for path in paths:
        episode = json.loads((local / path.replace("/", "__")).read_text())
        by_task[episode["task_id"]].append(bool(episode["valid"]))
        rewards.append(float(episode.get("reward") or 0.0))
    rate = {task: sum(solved) / len(solved) for task, solved in by_task.items()}
    by_tier = defaultdict(list)
    for task, value in rate.items():
        by_tier[tiers.get(task, "?")].append(value)
    return {
        "pass1": sum(rate.values()) / len(rate),
        "passk": sum(any(solved) for solved in by_task.values()) / len(by_task),
        "reward": sum(rewards) / len(rewards),
        "tiers": {tier: sum(values) / len(values) for tier, values in by_tier.items()},
        "episodes": len(paths),
        "tasks": len(by_task),
    }


def table(rows, base, attempts, expected):
    lines = [
        f"| Step | Pass@1 | Pass@{attempts} | Easy | Medium | Hard | Reward | Episodes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for step, s in [("base", base)] + sorted(rows.items()):
        if isinstance(s, str):
            lines.append(f"| {step} | {s} |  |  |  |  |  |  |")
        elif s:
            tier = lambda name: f"{s['tiers'][name]:.3f}" if name in s["tiers"] else ""  # noqa: E731
            lines.append(
                f"| {step} | {s['pass1']:.3f} | {s['passk']:.3f} | {tier('easy')} | {tier('medium')} | "
                f"{tier('hard')} | {s['reward']:.3f} | {s['episodes']}/{expected} |"
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
    parser.add_argument("--harness", choices=("board", "training"), default="board", help="The evaluations' agent loop")
    parser.add_argument("--attempts", type=int, default=1, help="Episodes per task")
    parser.add_argument("--temperature", type=float, help="Sampling temperature (default: the board's 0)")
    parser.add_argument("--shards", type=int, default=1, help="Evaluation jobs per checkpoint, one H200 each")
    args = parser.parse_args()
    namespace = args.bucket.partition("/")[0]

    record = json.loads((ROOT / "data" / f"eval-{args.set}.json").read_text())
    tiers = {task["task_id"]: task["tier"] for task in record["tasks"]}
    expected = len(record["tasks"]) * args.attempts
    scratch = Path(tempfile.mkdtemp())
    slices = []
    if args.shards > 1:
        # Contiguous slices of the set, published once beside the run for the jobs to read.
        ids = [task["task_id"] for task in record["tasks"]]
        size = -(-len(ids) // args.shards)
        for k in range(args.shards):
            path = scratch / f"{args.set}-{k}of{args.shards}.txt"
            path.write_text("\n".join(ids[k * size : (k + 1) * size]) + "\n")
            slices.append(f"/outputs/{args.run}/evalshards/{path.name}")
            hub.batch_bucket_files(args.bucket, add=[(str(path), f"{args.run}/evalshards/{path.name}")])
    names = lambda step: [f"{args.run}-eval-{step}-{k}" for k in range(args.shards)] if slices else [f"{args.run}-eval-{step}"]  # noqa: E731

    submitted, finished = {}, {}  # step -> {name: job id or None}, step -> score or a failure note
    base = score(args.bucket, [args.base], tiers, args.attempts, scratch / "base")
    while True:
        training = hub.inspect_job(job_id=args.job, namespace=namespace).status.stage
        for step in checkpoints(args.bucket, args.run):
            if step in submitted:
                continue
            submitted[step] = {}
            for k, name in enumerate(names(step)):
                if has_summary(args.bucket, name):  # evaluated by an earlier watch
                    submitted[step][name] = None
                    continue
                job = hf_job.main(
                    [
                        "eval", "--name", name, "--bucket", args.bucket, "--model", args.model, "--flavor", "h200",
                        "--set", args.set, "--checkpoint", f"/outputs/{args.run}/checkpoint-{step}",
                        "--label", f"{args.run} step {step}", "--harness", args.harness,
                        "--attempts", str(args.attempts), "--concurrency", "128",
                        *(["--temperature", str(args.temperature)] if args.temperature is not None else []),
                        *(["--task-ids", slices[k]] if slices else []),
                        "--submit",
                    ]
                )  # fmt: skip
                submitted[step][name] = job.id
                print(f"submitted evaluation of step {step} ({name}): {job.url}", flush=True)
        for step, jobs in submitted.items():
            if step in finished:
                continue
            stages = {
                name: hub.inspect_job(job_id=job, namespace=namespace).status.stage if job else "COMPLETED"
                for name, job in jobs.items()
            }
            if not all(stage in DONE for stage in stages.values()):
                continue
            failed = [name for name, stage in stages.items() if stage != "COMPLETED"]
            result = score(args.bucket, list(jobs), tiers, args.attempts, scratch / f"step-{step}")
            finished[step] = result if result and not failed else f"failed: {', '.join(failed) or 'no episodes'}"
            row = (
                f"pass@1 {result['pass1']:.3f}, pass@{args.attempts} {result['passk']:.3f}, "
                f"tiers {', '.join(f'{t} {v:.3f}' for t, v in sorted(result['tiers'].items()))}"
                if result
                else finished[step]
            )
            print(f"step {step}: {row}" + (f" ({finished[step]})" if failed and result else ""), flush=True)
            curve = scratch / "evals.md"
            curve.write_text(
                f"# {args.run} on {args.set}, {args.harness} harness, {args.attempts} attempts\n\n"
                + table(finished, base, args.attempts, expected)
            )
            hub.batch_bucket_files(args.bucket, add=[(str(curve), f"{args.run}/evals.md")])
        if (
            training in DONE
            and len(finished) == len(submitted)
            and not set(checkpoints(args.bucket, args.run)) - set(submitted)
        ):
            print(table(finished, base, args.attempts, expected), flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
