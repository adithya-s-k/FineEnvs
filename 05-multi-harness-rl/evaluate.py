"""Pass@1 on fixed held-out tasks; resume only missing or ungraded pairs."""

import argparse
import hashlib
import inspect
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from envs.tasks import load_tasks

HARNESSES = ("opencode", "claude-code", "codex", "mini-swe-agent")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def harbor_episode(args, index, task, harness):
    from harbor_env import HarborEnv
    from harbor_env.harness import HarborSession

    session = HarborSession(
        env=HarborEnv(args.server, message_timeout_s=1800),
        owns_env=True,
        split=str(Path(args.data).resolve() / "datasets/test"),
        task_index=index,
        instruction=task["instruction"],
        harness=harness,
        sandbox="daytona",
        llm_url=args.vllm_url,
        model=args.model,
        reward_key="correctness",
        sampling={"temperature": 0.8, "top_p": 1.0, "top_k": -1},
        agent_step_limit=17,
        agent_timeout_sec=600,
    )
    try:
        session.wait_for_completion(timeout_s=1200)
        correctness = session.verify([]).env_reward
        if correctness not in (0, 1):
            raise RuntimeError("No verifier grade")
        trace = (
            session.fetch_training_trace()
        )  # Checks engine tokens, logprobs and masks.
        path = Path(args.trials) / session.result.trial_name / "agent/trajectory.json"
        try:
            atif = json.loads(path.read_text())
            if not atif["schema_version"].startswith("ATIF-"):
                raise ValueError("Unknown trajectory format")
            ids = [
                c["tool_call_id"]
                for s in atif["steps"]
                if s.get("source") == "agent"
                for c in s.get("tool_calls", []) or []
            ]
            calls = len(ids) if all(ids) and len(set(ids)) == len(ids) else None
        except (OSError, ValueError, KeyError, TypeError):
            calls = None
        return {
            "correctness": correctness,
            "tool_calls": calls,
            "generated_tokens": sum(len(t.completion_token_ids) for t in trace.turns),
            "prompt_tokens": sum(len(t.prompt_token_ids) for t in trace.turns),
            "model_calls": len(trace.turns),
        }
    finally:
        session.close()


def whitebox_episode(args, index, task, harness):
    from openai import OpenAI
    from transformers.utils import get_json_schema

    from train_whitebox import SYSTEM, BashEnvironment

    environment = BashEnvironment()
    messages = [{"role": "system", "content": SYSTEM}]
    generated, prompt_tokens = 0, 0
    try:
        messages.append(
            {"role": "user", "content": environment.reset(folder=task["folder"])}
        )
        tools = [
            get_json_schema(method)
            for name, method in inspect.getmembers(environment, inspect.ismethod)
            if not name.startswith("_") and name not in {"reset", "get_reward"}
        ]
        allowed = {tool["function"]["name"] for tool in tools}
        with OpenAI(
            base_url=args.vllm_url.rstrip("/") + "/v1", api_key="unused", timeout=600
        ) as client:
            for _ in range(17):
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    tools=tools,
                    temperature=0.8,
                    top_p=1.0,
                    max_tokens=4096,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                message = response.choices[0].message
                generated += response.usage.completion_tokens
                prompt_tokens += response.usage.prompt_tokens
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    break
                for call in message.tool_calls:
                    if call.function.name not in allowed:
                        value = "Unknown tool"
                    else:
                        try:
                            value = getattr(environment, call.function.name)(
                                **json.loads(call.function.arguments)
                            )
                        except (TypeError, ValueError, RuntimeError) as exc:
                            value = type(exc).__name__
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": str(value)}
                    )
                if environment._submitted:
                    break
        value = environment.get_reward()
        if not math.isfinite(value):
            raise RuntimeError("No verifier grade")
        return {
            "correctness": environment._correctness,
            "tool_calls": environment._calls,
            "generated_tokens": generated,
            "prompt_tokens": prompt_tokens,
        }
    finally:
        environment._close()


def summarize(rows, expected):
    graded = [row for row in rows if row.get("correctness") in (0, 1)]

    def mean(key):
        values = [row[key] for row in graded if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "expected": expected,
        "graded": len(graded),
        "coverage": len(graded) / expected,
        "pass_at_1_observed": mean("correctness"),
        "combined_reward": mean("reward"),
        "mean_tool_calls": mean("tool_calls"),
        "tool_count_coverage": sum(r.get("tool_calls") is not None for r in graded),
        "mean_generated_tokens": mean("generated_tokens"),
        "mean_prompt_tokens": mean("prompt_tokens"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="LiquidAI/LFM2.5-2.6B", help="The served model name"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Immutable model revision or checkpoint identity",
    )
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--mode", choices=["blackbox", "whitebox"], default="blackbox")
    parser.add_argument("--data", default="prepared")
    parser.add_argument("--server", default="http://127.0.0.1:8200")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trials", default="runs/trials")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, default=250)
    parser.add_argument("--concurrency", type=int, default=35)
    parser.add_argument("--space-id", default=None)
    args = parser.parse_args()
    if not 1 <= args.tasks <= 250 or args.concurrency < 1:
        parser.error("Use 1 to 250 tasks and positive concurrency")
    tasks = load_tasks(args.data, "test")
    # Stable subset: every model and checkpoint sees the same task/harness pairs.
    selected = list(enumerate(tasks))[: args.tasks]
    harnesses = ["whitebox"] if args.mode == "whitebox" else HARNESSES
    output = Path(args.output)
    identity = {
        "protocol_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "checkpoint": args.checkpoint,
        "model": args.model,
        "harnesses": harnesses,
        "task_names": [t["name"] for _, t in selected],
        "temperature": 0.8,
        "instruction_hash": hashlib.sha256(
            "".join(t["instruction"] for _, t in selected).encode()
        ).hexdigest(),
    }
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != json.loads(
        json.dumps(identity)
    ):
        raise ValueError("Output already belongs to a different evaluation")
    write_json(identity_path, identity)
    pairs = [
        (index, task, harness) for index, task in selected for harness in harnesses
    ]

    def one(pair):
        index, task, harness = pair
        path = output / "pairs" / f"{task['name']}--{harness}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get("correctness") in (0, 1):
                return previous
        start = time.monotonic()
        row = {
            "task": task["name"],
            "difficulty": task["difficulty"],
            "harness": harness,
        }
        try:
            episode = whitebox_episode if args.mode == "whitebox" else harbor_episode
            row.update(episode(args, index, task, harness))
            calls = row["tool_calls"]
            row["reward"] = row["correctness"] * (
                1 + 1.5 / (15 + calls) if calls else 1
            )
        except Exception as exc:  # noqa: BLE001 - persist an ungraded pair for retry
            row.update(correctness=None, error=type(exc).__name__)
        row["seconds"] = time.monotonic() - start
        write_json(path, row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for future in as_completed([pool.submit(one, pair) for pair in pairs]):
            rows.append(future.result())
            write_json(output / "progress.json", summarize(rows, len(pairs)))
    summary = summarize(rows, len(pairs))
    summary["harnesses"] = {
        h: summarize([r for r in rows if r["harness"] == h], args.tasks)
        for h in harnesses
    }
    summary["difficulty"] = {
        d: summarize(
            [r for r in rows if r["difficulty"] == d],
            sum(t["difficulty"] == d for _, t in selected) * len(harnesses),
        )
        for d in {t["difficulty"] for _, t in selected}
    }
    write_json(output / "summary.json", summary)
    import os

    import trackio

    os.environ["TRACKIO_DIR"] = str(output.resolve() / "trackio")
    run = trackio.init(
        project="smoldataenv-rl",
        name=output.name,
        space_id=args.space_id,
        config=identity,
    )
    run.log(
        {
            f"eval/{key}": value
            for key, value in summary.items()
            if isinstance(value, (int, float))
        },
        step=args.step,
    )
    run.finish()
    print(json.dumps(summary, indent=2))
    if summary["graded"] != summary["expected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
