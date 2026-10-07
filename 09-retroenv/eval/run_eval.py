#!/usr/bin/env python3
"""Evaluate a model on RetroEnv through the OpenEnv server.

Every tool call and the reward come from the server, the same one used for
training. Without ``--server`` the script starts a local server on a free
port for ``--benchmark-dir`` and stops it at the end.

    uv run --extra eval python eval/run_eval.py --provider anthropic \\
        --model claude-opus-5-5 --split test_id --output runs/test_id/opus-5-5

``--provider reference`` runs no model: a scripted expert replays each task's
compliant known routes through the same loop and server (the SFT trajectory generator).

Output directory:
    identity.json   model, sampling, tasks and code hashes; a rerun must match
    episodes/       one JSON per task and attempt, with the full transcript
    progress.json   summary so far, rewritten after every episode
    summary.json    final metrics, with breakdowns by route count, depth and tier

Rerunning the same command resumes: graded episodes are kept, and episodes
that hit a provider or transport error are retried.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import httpx
from retroenv_openenv import agent, agent_anthropic, agent_reference, agent_responses
from retroenv_openenv.client import RetroEnvClient

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {"backend": "anthropic", "key": "ANTHROPIC_API_KEY"},
    # GPT-5.6 models reject function tools with reasoning on chat completions.
    "openai": {"backend": "responses", "endpoint": "https://api.openai.com/v1", "key": "OPENAI_API_KEY"},
    "hf": {
        "backend": "openai",
        "endpoint": "https://router.huggingface.co/v1",
        "key": "HF_TOKEN",
        "token_param": "max_tokens",
        "temperature": 0.0,
    },
    "openrouter": {
        "backend": "openai",
        "endpoint": "https://openrouter.ai/api/v1",
        "key": "OPENROUTER_API_KEY",
        "token_param": "max_tokens",
        "temperature": 0.0,
        "extra_body": {"usage": {"include": True}},
    },
    "custom": {
        "backend": "openai",
        "endpoint": None,
        "key": "OPENAI_API_KEY",
        "token_param": "max_tokens",
        "temperature": 0.0,
    },
    # No model and no key: the scripted expert reads <benchmark-dir>/tasks-private.
    "reference": {"backend": "reference", "key": None},
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@contextmanager
def local_server(benchmark_dir: Path, toolset: str, concurrency: int, log_path: Path) -> Iterator[str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    environment = {
        **os.environ,
        "RETROENV_BENCHMARK_DIR": str(benchmark_dir.resolve()),
        "RETROENV_TOOLSET": toolset,
        "ENABLE_WEB_INTERFACE": "false",
        "MAX_CONCURRENT_ENVS": str(max(16, 2 * concurrency + 4)),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "retroenv_openenv.server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 600  # loading a full release takes a few minutes
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"local server exited; see {log_path}")
                try:
                    if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(f"local server did not become healthy; see {log_path}")
                time.sleep(0.5)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def summarize(rows: list[dict[str, Any]], expected: int, attempts: int) -> dict[str, Any]:
    graded = [row for row in rows if row.get("graded")]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in graded:
        by_task[row["task_id"]].append(row)
    first = [sorted(runs, key=lambda r: r["attempt"])[0] for runs in by_task.values()]

    def mean(values: list[float]) -> float | None:
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    passes = sum(row["valid"] for row in first)
    components = sorted({key for row in first for key in (row.get("components") or {})})
    costs = [row["usage"].get("cost_usd") for row in graded]
    summary: dict[str, Any] = {
        "episodes_expected": expected,
        "episodes_graded": len(graded),
        "coverage": round(len(graded) / expected, 4) if expected else 0.0,
        "tasks": len(first),
        "pass_at_1": mean([float(row["valid"]) for row in first]),
        "pass_at_1_ci95": wilson(passes, len(first)),
        "exact_route_rate": mean([float(row["exact_match"]) for row in first]),
        "mean_reward": mean([row["reward"] for row in first]),
        "components": {key: mean([(row.get("components") or {}).get(key) for row in first]) for key in components},
        "mean_tool_calls": mean([row["tool_calls"] for row in first]),
        "mean_turns": mean([row["turns"] for row in first]),
        "no_emit_rate": mean([float(row["auto_emitted"]) for row in first]),
        "refusal_rate": mean([float(any(e.startswith("refusal") for e in row["errors"])) for row in first]),
        "coerced_submission_rate": mean([float(row.get("submission_coerced", False)) for row in first]),
        "cost_usd": round(sum(c for c in costs if c is not None), 4) if any(c is not None for c in costs) else None,
        "mean_latency_seconds": mean([row["usage"].get("latency_seconds") for row in graded]),
        "prompt_tokens": sum(row["usage"].get("prompt_tokens") or 0 for row in graded),
        "completion_tokens": sum(row["usage"].get("completion_tokens") or 0 for row in graded),
    }
    if attempts > 1:
        summary["pass_at_k"] = {
            str(k): mean(
                [pass_at_k(len(runs), sum(r["valid"] for r in runs), k) for runs in by_task.values() if len(runs) >= k]
            )
            for k in sorted({1, attempts})
        }
    return summary


def breakdown(rows: list[dict[str, Any]], key: str, attempts: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    return {
        name: {
            k: v
            for k, v in summarize(members, len(members), attempts).items()
            if k in ("tasks", "pass_at_1", "exact_route_rate", "mean_reward", "mean_tool_calls")
        }
        for name, members in sorted(groups.items())
    }


def recompute(output: Path) -> int:
    """Rebuild summary.json from episodes/ after a metrics change, without rerunning anything."""
    identity = json.loads((output / "identity.json").read_text())
    rows = [json.loads(path.read_text()) for path in sorted((output / "episodes").glob("*.json"))]
    if not rows:
        raise SystemExit(f"{output} has no episodes to recompute from")
    attempts = identity["attempts"]
    expected = len(identity["task_ids"]) * attempts
    graded = [row for row in rows if row.get("graded")]
    summary = summarize(rows, expected, attempts)
    summary["label"] = identity["label"]
    summary["model"] = identity["model"]
    summary["toolset"] = identity["toolset"]
    summary["by_variant"] = breakdown(graded, "variant", attempts)
    summary["by_max_depth"] = breakdown(graded, "max_depth", attempts)
    if any(row.get("tier") for row in graded):
        summary["by_tier"] = breakdown(graded, "tier", attempts)
    write_json(output / "summary.json", summary)
    print(f"{output}: {summary['episodes_graded']}/{expected} graded", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model", help="required unless --provider reference")
    parser.add_argument("--label", help="display name in summaries (defaults to the model)")
    parser.add_argument("--endpoint", help="OpenAI-compatible base URL (custom provider, or an override)")
    parser.add_argument("--api-key-env", help="environment variable holding the API key")
    parser.add_argument("--server", help="URL of a running RetroEnv server; default starts a local one")
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "data" / "release" / "RetroEnv-RL")
    parser.add_argument(
        "--toolset",
        choices=("full", "unaided"),
        default="full",
        help="tool surface of the local server (a remote server reports its own)",
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--tasks", type=int, help="number of tasks from --start (default: the whole split)")
    parser.add_argument("--task-ids", type=Path, help="file with one task ID per line; overrides --start/--tasks")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, help="per model turn (default 4096; 16000 for Claude and Responses)")
    parser.add_argument("--temperature", type=float, help="override the provider default")
    parser.add_argument("--effort", help="Claude output_config.effort (default: the model's own default)")
    parser.add_argument("--reasoning-effort", help="OpenAI-compatible reasoning_effort")
    parser.add_argument("--tool-choice", default="required", help="OpenAI-compatible tool_choice")
    parser.add_argument("--max-cost", type=float, help="stop scheduling episodes once this many USD are spent")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="rebuild summary.json from the stored episodes; no provider or server calls",
    )
    args = parser.parse_args()

    if args.recompute:
        return recompute(args.output)

    preset = PROVIDERS[args.provider]
    if args.temperature is not None and preset["backend"] != "openai":
        parser.error(
            f"--temperature is not supported by the {args.provider} backend; use --effort or --reasoning-effort"
        )
    if preset["backend"] == "reference":
        args.model = args.model or agent_reference.MODEL
    elif not args.model:
        parser.error("--model is required")
    key_env = args.api_key_env or preset["key"]
    if key_env and not os.getenv(key_env):
        parser.error(f"{key_env} is not set")
    endpoint = args.endpoint or preset.get("endpoint")
    if preset["backend"] not in ("anthropic", "reference") and not endpoint:
        parser.error("--endpoint is required for the custom provider")
    pricing = json.loads((ROOT / "eval" / "pricing.json").read_text())

    if preset["backend"] == "anthropic":
        import anthropic

        llm: Any = anthropic.Anthropic(api_key=os.environ[key_env], max_retries=4, timeout=600)
        config: Any = agent_anthropic.ClaudeConfig(
            model=args.model,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens or 16000,
            effort=args.effort,
        )
        backend = agent_anthropic.run_episode
        sampling = {"max_tokens": config.max_tokens, "effort": config.effort, "tool_choice": "auto"}
    elif preset["backend"] == "reference":
        llm = agent_reference.ReferenceTasks(args.benchmark_dir)
        config = agent.AgentConfig(model=args.model, max_turns=args.max_turns, temperature=None)
        backend = agent_reference.run_episode
        sampling = {"policy": "scripted reference expert"}
    elif preset["backend"] == "responses":
        from openai import OpenAI

        llm = OpenAI(base_url=endpoint, api_key=os.environ[key_env], timeout=600, max_retries=4)
        config = agent_responses.ResponsesConfig(
            model=args.model,
            max_turns=args.max_turns,
            max_output_tokens=args.max_tokens or 16000,
            reasoning_effort=args.reasoning_effort,
        )
        backend = agent_responses.run_episode
        sampling = {
            "max_output_tokens": config.max_output_tokens,
            "reasoning_effort": args.reasoning_effort,
            "tool_choice": "required",
        }
    else:
        from openai import OpenAI

        llm = OpenAI(base_url=endpoint, api_key=os.environ[key_env], timeout=600, max_retries=4)
        temperature = args.temperature if args.temperature is not None else preset.get("temperature")
        config = agent.AgentConfig(
            model=args.model,
            max_turns=args.max_turns,
            max_tokens=args.max_tokens or 4096,
            temperature=temperature,
            tool_choice=args.tool_choice,
            token_param=preset.get("token_param", "max_tokens"),
            reasoning_effort=args.reasoning_effort,
            extra_body=dict(preset.get("extra_body") or {}),
        )
        backend = agent.run_episode
        sampling = {
            "max_tokens": config.max_tokens,
            "temperature": temperature,
            "tool_choice": args.tool_choice,
            "reasoning_effort": args.reasoning_effort,
        }

    private = args.benchmark_dir / "tasks-private" / f"{args.split}.jsonl"
    if not args.server and not private.exists():
        parser.error(f"{private} is missing; download the release from LiteFold/RetroEnv (see eval/README.md)")
    tiers = {}
    if private.exists():
        tiers = {row["task_id"]: row.get("difficulty", {}).get("tier") for row in map(json.loads, private.open())}

    output: Path = args.output
    with (
        local_server(args.benchmark_dir, args.toolset, args.concurrency, output / "server.log")
        if not args.server
        else _nullcontext(args.server)
    ) as url:
        probe = RetroEnvClient(url)
        public = probe.tasks(args.split)
        if args.task_ids:
            wanted = [line.strip() for line in args.task_ids.read_text().splitlines() if line.strip()]
            by_id = {task["task_id"]: task for task in public}
            missing = [task_id for task_id in wanted if task_id not in by_id]
            if missing:
                parser.error(f"{len(missing)} task IDs are not in split {args.split!r}, e.g. {missing[0]}")
            selected = [by_id[task_id] for task_id in wanted]
        else:
            selected = public[args.start : args.start + args.tasks if args.tasks else None]
        if not selected:
            parser.error("the selected task range is empty")
        opening = probe.reset(args.split, index=selected[0]["index"])
        tools = probe.openai_tools()
        probe.call("emit_routes", {"submission": {"routes": []}})
        probe.close()

        code = b"".join(
            Path(module.__file__).read_bytes() for module in (agent, agent_anthropic, agent_reference, agent_responses)
        )
        identity = {
            "schema_version": "retro-eval-run-v2",
            "provider": args.provider,
            "model": args.model,
            "label": args.label or args.model,
            "endpoint": endpoint,
            "split": args.split,
            "task_ids": [task["task_id"] for task in selected],
            "attempts": args.attempts,
            "max_turns": args.max_turns,
            "sampling": sampling,
            "toolset": opening.get("toolset"),
            "max_tool_calls": opening.get("max_tool_calls"),
            "server": "local" if not args.server else args.server,
            "benchmark_public_sha256": sha256(json.dumps(selected, sort_keys=True).encode()),
            "tool_schema_sha256": sha256(json.dumps(tools, sort_keys=True).encode()),
            "agent_code_sha256": sha256(code),
        }
        identity_path = output / "identity.json"
        if identity_path.exists():
            previous = json.loads(identity_path.read_text())
            changed = sorted(k for k in identity if previous.get(k) != json.loads(json.dumps(identity[k])))
            if changed == ["agent_code_sha256"]:
                # Recorded, not enforced: a loop fix for one provider should not void other runs.
                print("note: agent code changed since this run started", file=sys.stderr)
                identity["agent_code_sha256"] = previous["agent_code_sha256"]
            elif changed:
                raise SystemExit(f"{output} belongs to a different evaluation (changed: {changed}); use a new --output")
        write_json(identity_path, identity)

        jobs = [(task, attempt) for task in selected for attempt in range(args.attempts)]
        rows: dict[tuple[str, int], dict[str, Any]] = {}
        lock = threading.Lock()
        spent = [0.0]
        price = pricing.get(args.model)

        def episode_path(task: dict[str, Any], attempt: int) -> Path:
            return output / "episodes" / f"{task['index']:04d}-{task['task_id']}-a{attempt}.json"

        for task, attempt in jobs:
            path = episode_path(task, attempt)
            if path.exists():
                row = json.loads(path.read_text())
                if row.get("graded"):
                    rows[(task["task_id"], attempt)] = row
                    spent[0] += row["usage"].get("cost_usd") or 0.0

        def run(task: dict[str, Any], attempt: int) -> dict[str, Any] | None:
            if args.max_cost is not None:
                with lock:
                    if spent[0] >= args.max_cost:
                        return None
            with RetroEnvClient(url) as env:
                opening = env.reset(
                    args.split, index=task["index"], episode_id=f"{identity['label']}:{task['task_id']}:{attempt}"
                )
                if opening["task_id"] != task["task_id"]:
                    raise RuntimeError(f"server returned {opening['task_id']} for {task['task_id']}")
                result = backend(llm, env, opening, config)
            usage = result["usage"]
            cost = usage.get("reported_cost_usd") or None
            if cost is None and price:
                cost = (usage["prompt_tokens"] * price["input"] + usage["completion_tokens"] * price["output"]) / 1e6
            usage["cost_usd"] = round(cost, 6) if cost is not None else None
            infra = [e for e in result["errors"] if e.startswith("API")]
            row = {
                "task_id": task["task_id"],
                "index": task["index"],
                "attempt": attempt,
                "split": args.split,
                "model": args.model,
                "variant": task["variant"],
                "max_depth": task["max_depth"],
                "tier": tiers.get(task["task_id"]),
                "target_smiles": task["target_smiles"],
                # The OpenAI-format transcript starts after the opening prompt; SFT export needs it.
                "prompt": opening["prompt"],
                "graded": not infra,
                **result,
            }
            write_json(episode_path(task, attempt), row)
            with lock:
                spent[0] += cost or 0.0
            return row

        pending = [(task, attempt) for task, attempt in jobs if (task["task_id"], attempt) not in rows]
        print(f"{len(rows)} graded episodes kept, {len(pending)} to run on {url}", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run, task, attempt): (task, attempt) for task, attempt in pending}
            for future in as_completed(futures):
                task, attempt = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # transport failure: keep it ungraded for the next run
                    print(f"{task['task_id']} a{attempt}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
                if row is None:
                    continue
                with lock:
                    rows[(task["task_id"], attempt)] = row
                    done = list(rows.values())
                write_json(output / "progress.json", summarize(done, len(jobs), args.attempts))
                print(
                    f"[{len(done)}/{len(jobs)}] {task['task_id']} reward={row['reward']:.3f} "
                    f"valid={row['valid']} exact={row['exact_match']} cost=${spent[0]:.2f}",
                    file=sys.stderr,
                )

    final_rows = list(rows.values())
    summary = summarize(final_rows, len(jobs), args.attempts)
    summary["label"] = identity["label"]
    summary["model"] = args.model
    summary["toolset"] = identity["toolset"]
    summary["by_variant"] = breakdown([r for r in final_rows if r.get("graded")], "variant", args.attempts)
    summary["by_max_depth"] = breakdown([r for r in final_rows if r.get("graded")], "max_depth", args.attempts)
    if tiers:
        summary["by_tier"] = breakdown([r for r in final_rows if r.get("graded")], "tier", args.attempts)
    write_json(output / "summary.json", summary)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("coverage", "pass_at_1", "pass_at_1_ci95", "exact_route_rate", "mean_reward", "cost_usd")
            },
            indent=2,
        )
    )
    return 0 if summary["episodes_graded"] == len(jobs) else 2


@contextmanager
def _nullcontext(value: str) -> Iterator[str]:
    yield value


if __name__ == "__main__":
    raise SystemExit(main())
