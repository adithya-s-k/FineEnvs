"""Score a model on the held-out test problems, through the same harness and judge as training.

Adapted from 05-multi-harness-rl/eval/evaluate.py for a fractional reward. Every model and checkpoint sees the
same problems (the first --tasks of the sorted test split) and --samples attempts each. Per attempt it records the
rubric reward (score / 7); the summary has the mean reward, the share of full-marks proofs, and coverage (attempts
that got a grade). Runs resume: graded attempts are kept, ungraded ones are retried.

    python evaluate.py --output runs/eval-base --checkpoint Qwen/Qwen3.5-2B --model Qwen/Qwen3.5-2B ...
"""

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import os
import random

from tasks import AGENT_TIMEOUT_SEC, HARNESS, dataset_dir, load_tasks, session_for

TEMPERATURE = 0.8


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def graded(row):
    return isinstance(row.get("reward"), (int, float)) and math.isfinite(row["reward"])


def episode(args, index, task, split):
    sampling = None if args.hosted else {"temperature": TEMPERATURE, "top_p": 1.0, "top_k": -1}
    session = session_for(args, index, task, split, sampling)
    try:
        session.wait_for_completion(timeout_s=AGENT_TIMEOUT_SEC)  # as in training
        reward = session.verify([]).env_reward
        if reward is None or not math.isfinite(reward):
            raise RuntimeError("No verifier grade")
        return {"reward": float(reward), "trial": getattr(session.result, "trial_name", None)}
    finally:
        session.close()


def summarize(rows, expected):
    """Attempt-level means, then problem-level ones: a problem is the unit, so the standard error is over problems."""
    done = [r for r in rows if graded(r)]
    mean = (lambda xs: sum(xs) / len(xs) if xs else None)
    by_problem = {}
    for r in done:
        by_problem.setdefault(r["task"], []).append(r["reward"])
    per_problem = [sum(v) / len(v) for v in by_problem.values()]
    n = len(per_problem)
    spread = (sum((x - sum(per_problem) / n) ** 2 for x in per_problem) / (n - 1)) ** 0.5 if n > 1 else None
    return {
        "expected": expected,
        "graded": len(done),
        "coverage": len(done) / expected if expected else None,
        "mean_reward": mean([r["reward"] for r in done]),
        "mean_reward_se": spread / n ** 0.5 if spread is not None else None,
        "full_marks": mean([float(r["reward"] >= 0.999) for r in done]),
        "any_credit": mean([float(r["reward"] > 0) for r in done]),
        "problems": n,
        "pass_at_k_full_marks": mean([float(max(v) >= 0.999) for v in by_problem.values()]),
        "best_of_k_mean": mean([max(v) for v in by_problem.values()]),
        "learnable_problems": mean([float(len(set(v)) > 1) for v in by_problem.values()]),
    }


def report(args, output, summary):
    """Chart one checkpoint's eval in Trackio: the summary as `eval/*` at --step, the rollout metrics as
    `eval_rollouts/*` and `eval_judge/*`, and one trace per attempt. summary.json stays the record; a chart that
    fails never fails the eval."""
    os.environ.setdefault("TRACKIO_DIR", str(output.resolve() / "trackio"))
    try:
        import trackio

        import rollout_log

        trials = Path(args.trials)
        records, _ = rollout_log.read_new(output / "rollouts.jsonl", 0)
        records = records or rollout_log.records_from_trials(trials)
        metrics = {f"eval/{k}": v for k, v in summary.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        metrics.update({f"eval_{k}": v for k, v in rollout_log.summarize(records).items()})
        run = trackio.init(project="fineproof-rl", name=args.trackio_run or output.name, space_id=args.space_id,
                           bucket_id=args.trackio_bucket, private=False if args.space_id else None,
                           resume="allow", config=summary.get("identity") or {})
        run.log(metrics, step=args.step)
        n = rollout_log.log_traces(run.log, "eval_rollouts", records, trials, args.step, args.model, kind="eval")
        run.finish()
        print(f"trackio: logged {len(metrics)} metrics and {n} traces at step {args.step}")
    except Exception as exc:  # noqa: BLE001
        print(f"trackio logging failed: {type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B", help="The served model name")
    parser.add_argument("--checkpoint", required=True, help="Model revision or checkpoint being scored")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--data", default="prepared")
    parser.add_argument("--env-llm-url")
    parser.add_argument("--server", default="http://127.0.0.1:8200")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trials", default="runs/trials")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, default=128)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--space-id", default=None)
    parser.add_argument("--trackio-bucket", default=None, help="Bucket the Trackio Space keeps its data in")
    parser.add_argument("--trackio-run", default=None,
                        help="Trackio run to log into, e.g. <training run>-evals so every checkpoint lands on one curve")
    parser.add_argument("--sandbox", default="daytona", choices=["daytona", "hf-sandbox"],
                        help="Where Harbor runs each task's sandboxes (hf-sandbox: HF Jobs, needs a docker_image task)")
    parser.add_argument("--dataset", help="The served test folder, if not <data>/datasets/test")
    parser.add_argument("--task-seed", type=int, help="Score a seeded random --tasks subset instead of the first ones")
    parser.add_argument("--task-list", help="Score exactly the test tasks named in this file (one per line, # comments)")
    parser.add_argument("--rounds", type=int, help="The server's harness rounds, recorded in identity.json")
    parser.add_argument("--max-tokens", type=int, help="The server's output cap per call, recorded in identity.json")
    parser.add_argument("--hosted", action="store_true",
                        help="Eval rollouts on the server's own engine (e.g. a hosted provider) with its default sampling")
    parser.add_argument("--report-only", action="store_true",
                        help="Log an existing eval (<output>/summary.json and its trials) to Trackio, without running")
    args = parser.parse_args()
    if args.report_only:
        report(args, Path(args.output), json.loads((Path(args.output) / "summary.json").read_text()))
        return
    if args.hosted:
        args.vllm_url = ""
    split = str(Path(args.dataset).resolve()) if args.dataset else dataset_dir(args.data, "test")
    tasks = load_tasks(split)
    selected = list(enumerate(tasks))
    if args.task_list:
        names = [line.strip() for line in Path(args.task_list).read_text().splitlines()
                 if line.strip() and not line.startswith("#")]
        missing = set(names) - {t["name"] for t in tasks}
        if missing:
            raise ValueError(f"{len(missing)} listed tasks are not in {split}: {sorted(missing)[:5]}")
        selected = [(i, t) for i, t in selected if t["name"] in set(names)]
        args.tasks = len(selected)
    if args.task_seed is not None:
        selected = sorted(random.Random(args.task_seed).sample(selected, min(args.tasks, len(selected))))
    selected = selected[: args.tasks]
    output = Path(args.output)
    identity = {
        "protocol_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint,
        "model": args.model,
        "harness": HARNESS,
        "sandbox": args.sandbox,
        "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "task_seed": args.task_seed,
        "task_list": Path(args.task_list).name if args.task_list else None,
        "judge_samples": os.environ.get("JUDGE_SAMPLES") or "task default",
        "judge_model": os.environ.get("JUDGE_MODEL") or "task default",
        "samples": args.samples,
        "temperature": None if args.hosted else TEMPERATURE,
        "task_names": [t["name"] for _, t in selected],
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Output already belongs to a different evaluation")
    write_json(identity_path, identity)
    pairs = [(i, t, s) for i, t in selected for s in range(args.samples)]

    def one(pair):
        index, task, sample = pair
        path = output / "attempts" / f"{task['name']}--{sample}.json"
        if path.exists() and graded(previous := json.loads(path.read_text())):
            return previous
        start = time.monotonic()
        row = {"task": task["name"], "sample": sample, "base_reward": task["base_reward"]}
        try:
            row.update(episode(args, index, task, split))
        except Exception as exc:  # noqa: BLE001 - keep an ungraded attempt for retry
            row.update(reward=None, error=type(exc).__name__, detail=str(exc)[:1000])
        row["seconds"] = round(time.monotonic() - start, 1)
        write_json(path, row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for future in as_completed([pool.submit(one, pair) for pair in pairs]):
            rows.append(future.result())
            write_json(output / "progress.json", summarize(rows, len(pairs)))
    summary = summarize(rows, len(pairs))
    summary["identity"] = identity
    write_json(output / "summary.json", summary)
    report(args, output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "identity"}, indent=2))
    if summary["graded"] != summary["expected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
