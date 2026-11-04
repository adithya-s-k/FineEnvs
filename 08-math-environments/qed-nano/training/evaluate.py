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

from tasks import load_tasks, session_for, split_paths

TEMPERATURE = 0.8


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def graded(row):
    return isinstance(row.get("reward"), (int, float)) and math.isfinite(row["reward"])


def episode(args, index, task, split_path):
    session = session_for(args, index, task, split_path, {"temperature": TEMPERATURE, "top_p": 1.0, "top_k": -1})
    try:
        session.wait_for_completion(timeout_s=1800)
        reward = session.verify([]).env_reward
        if reward is None or not math.isfinite(reward):
            raise RuntimeError("No verifier grade")
        return {"reward": float(reward), "trial": getattr(session.result, "trial_name", None)}
    finally:
        session.close()


def summarize(rows, expected):
    done = [r for r in rows if graded(r)]
    mean = (lambda xs: sum(xs) / len(xs) if xs else None)
    return {
        "expected": expected,
        "graded": len(done),
        "coverage": len(done) / expected if expected else None,
        "mean_reward": mean([r["reward"] for r in done]),
        "full_marks": mean([float(r["reward"] >= 0.999) for r in done]),
        "any_credit": mean([float(r["reward"] > 0) for r in done]),
    }


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
    args = parser.parse_args()
    tasks = load_tasks(args.data, "test")
    selected = list(enumerate(tasks))[: args.tasks]
    split_path = split_paths(args.server)["test"]
    output = Path(args.output)
    identity = {
        "protocol_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint,
        "model": args.model,
        "harness": "qed-proof",
        "samples": args.samples,
        "temperature": TEMPERATURE,
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
            row.update(episode(args, index, task, split_path))
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
    import os

    os.environ["TRACKIO_DIR"] = str(output.resolve() / "trackio")
    import trackio

    run = trackio.init(project="qed-nano-harbor-rl", name=output.name, space_id=args.space_id, config=identity)
    run.log({f"eval/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}, step=args.step)
    run.finish()
    print(json.dumps({k: v for k, v in summary.items() if k != "identity"}, indent=2))
    if summary["graded"] != summary["expected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
