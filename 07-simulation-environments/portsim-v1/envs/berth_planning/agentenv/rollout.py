"""Run PortSimEnv v1 on AgentEnv: one model on a few tasks, each played by the portsim-llm agent and graded by
portsim-verifier, and print what happened.

    uv run python rollout.py --smoke                                         # no model or key: 1.0, 0.176471 and 0
    HF_TOKEN=hf_... uv run python rollout.py                                 # GLM-5.3-Flash on the HF router, 1 task
    uv run python rollout.py --model anthropic/claude-sonnet-5-5 --tasks 3   # through LITELLM_* or [model]
    uv run python rollout.py --model hf:Qwen/Qwen3.8-27B:ovhcloud            # ../openenv's model spellings work too

It stages a bundle in runs/bundle (the env's Docker build context, the agent, the verifier and one task file per task)
and runs it in this process with agent-env, which builds the images on first use, starts its own registry on
127.0.0.1:5000 and keeps its stores under ~/.local/state/agent-env. Each run writes runs/<UTC>/results.jsonl and
runs/<UTC>/transcripts/<task>.json. Exit 0 when every run finished, 1 when one failed (or --smoke scored wrong), 2 when
a check or the bundle failed before anything ran, 128 + the signal number when interrupted.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from agent_env.bundle import BundleError, BundleRun, Outcome, RunInterrupted, TaskRun, run_bundle
from agent_env.config import ConfigError, get_config
from agent_env.store.routing import namespace_routing
from agent_env.utils.docker_build import docker_unreachable
from berth_core import Task, load_pack

from berth_agentenv import tasks

HERE = Path(__file__).resolve().parent
ENV_ROOT = HERE.parent
RUNS = HERE / "runs"
BUNDLE = RUNS / "bundle"
PACKS = ("dock-v1-eval", "dock-v1-train")
HF_ROUTER = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "zai-org/GLM-5.3-Flash:baseten"
SMOKE_SCORES = {"smoke": 1.0, "wiring-infeasible": 0.176471, "wiring-nosubmit": 0.0}
ENV_TOML = 'environment_name = "portsim"\nenv_provider_type = "server"\n'
SKIP = shutil.ignore_patterns("__pycache__", "*.egg-info", "build", "tests", ".pytest_cache", ".venv")


def priced_models() -> dict[str, tuple]:
    """The agent's PRICES, read without importing it (it needs the agent image's packages)."""
    for node in ast.parse((HERE / "agent" / "agent.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PRICES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise LookupError("agent/agent.py has no PRICES")


def model_id(spec: str) -> str:
    """../openenv's spellings as portsim-llm's ids: hf:M is M, anthropic:X and openai:X are anthropic/X and openai/X."""
    provider, colon, name = spec.partition(":")
    if colon and provider == "hf":
        return name
    if colon and provider in ("anthropic", "openai"):
        return f"{provider}/{name}"
    return spec


def endpoint() -> str | None:
    """Where the agent's model calls go: LITELLM_BASE_URL and LITELLM_API_KEY or [model] in .agentenv/config.toml,
    else the Hugging Face router with HF_TOKEN; None when neither is there."""
    config = get_config()
    try:
        config.get_litellm_api_key()
    except ConfigError:
        if token := os.environ.get("HF_TOKEN"):
            os.environ["LITELLM_BASE_URL"], os.environ["LITELLM_API_KEY"] = HF_ROUTER, token
            return HF_ROUTER
        return None
    try:
        return config.get_litellm_base_url()
    except ConfigError:
        return None


def stage(task_files: dict[str, list[dict]], agent: bool) -> None:
    """runs/bundle: the env folder holds what agentenv/Dockerfile copies from envs/berth_planning, so the same
    Dockerfile builds it."""
    shutil.rmtree(BUNDLE, ignore_errors=True)
    env = BUNDLE / "envs" / "portsim"
    shutil.copytree(ENV_ROOT / "core", env / "core", ignore=SKIP)
    for pack in PACKS:
        shutil.copytree(ENV_ROOT / "tasks" / pack, env / "tasks" / pack, ignore=SKIP)
    shutil.copytree(HERE / "berth_agentenv", env / "agentenv" / "berth_agentenv", ignore=SKIP)
    shutil.copy2(HERE / "pyproject.toml", env / "agentenv" / "pyproject.toml")
    shutil.copy2(HERE / "Dockerfile", env / "Dockerfile")
    (env / "env.toml").write_text(ENV_TOML)
    if agent:
        shutil.copytree(HERE / "agent", BUNDLE / "agents" / "portsim-llm", ignore=SKIP)
    verifier = BUNDLE / "artifacts" / "portsim-verifier"
    verifier.mkdir(parents=True)
    shutil.copy2(HERE / "verifier" / "verify.py", verifier / "verify.py")
    (BUNDLE / "tasks").mkdir()
    for name, task_steps in task_files.items():
        (BUNDLE / "tasks" / f"{name}.json").write_bytes(tasks.encode(task_steps))


def trajectory(url: str) -> dict:
    with namespace_routing():
        return json.loads(get_config().get_object_store_at(url).get(url))


def row(name: str, task: Task, model: str | None, run: TaskRun) -> tuple[dict, dict | None]:
    """The run's results.jsonl row and the agent's record, from the context the run left."""
    verification = (run.context.metadata.get("verifications") or {}).get("portsim") or {}
    episode = verification["results"][0]["episode"] if verification.get("results") else {}
    grade = episode.get("grade") or {}
    play = next((p for p in run.context.prompt_responses if p.step_id == "play"), None)
    summary = (play.structured_output or {}) if play else {}
    record = trajectory(play.agent_trajectory_object_url) if play and play.agent_trajectory_object_url else None
    failed = run.failed_steps[-1] if run.failed_steps else {}
    error = (play and play.error_message) or failed.get("error") or (run.error and str(run.error))
    return {"task": name, "task_id": task.task_id, "model": model, "outcome": run.outcome.value,
            "instance": run.instance_id, "seconds": round(run.duration, 1), "reward": verification.get("score"),
            "end_reason": episode.get("end_reason"), "feasible": grade.get("feasible"), "plan_cost": grade.get("cost"),
            "naive_cost": task.reference["naive_cost"], "optimal_cost": task.reference["optimal_cost"],
            "checks": episode.get("checks_used"), "calls": episode.get("calls_used"),
            **{key: summary.get(key) for key in ("turns", "tool_calls", "input_tokens", "output_tokens",
                                                 "cached_tokens", "cache_write_tokens", "cost_usd")},
            "failed_step": failed.get("step_id"), "error_code": (play and play.error_code) or failed.get("error_type"),
            "error": error and error[:500]}, record


def show(r: dict, task: Task, record: dict | None) -> None:
    print(f"\n=== {r['task']} ({task.task_id}, {task.difficulty}, {len(task.ships)} ships) ===")
    for m in (record or {}).get("messages", [])[2:]:
        if m["role"] == "assistant":
            calls = ", ".join(c["name"] for c in m.get("tool_calls") or [])
            print(f"assistant: {m['content'][:200]!r} -> {calls or 'no tool call'}")
        elif m["role"] == "tool":
            print(f"  {m['name']}: {m['content'][:240]}")
    agent = "" if r["turns"] is None else f"  turns {r['turns']}  ${r['cost_usd'] or 0:.4f}"
    print(f"reward {r['reward']}  end {r['end_reason']}  cost {r['plan_cost']}  naive {r['naive_cost']}  "
          f"optimum {r['optimal_cost']}{agent}  {r['outcome']}  instance {r['instance']}")
    if r["error"]:
        print(f"failed_step {r['failed_step']}  error_code {r['error_code']}  error {r['error']}")


def report(result: BundleRun, chosen: dict[str, Task], model: str | None, smoke: bool) -> int:
    out = RUNS / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out / "transcripts").mkdir(parents=True)
    code = 0
    with (out / "results.jsonl").open("w") as results:
        for run in result.runs:
            name = run.entry.name
            r, record = row(name, chosen[name], model, run)
            results.write(json.dumps(r) + "\n")
            if record is not None:
                (out / "transcripts" / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
            show(r, chosen[name], record)
            wrong = smoke and (r["reward"] is None or round(r["reward"], 6) != SMOKE_SCORES[name])
            if wrong:
                print(f"expected {SMOKE_SCORES[name]}")
            if wrong or run.outcome in (Outcome.FAILED, Outcome.CANCELLED):
                code = 1
    print(f"\nwrote {out.relative_to(HERE)}/results.jsonl")
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"portsim-llm's model (default {DEFAULT_MODEL})")
    ap.add_argument("--tasks", type=int, default=1, help="the first N tasks of --split, in pack order")
    ap.add_argument("--task-id", action="append", default=[], help="a task to play (repeatable; overrides --tasks)")
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--episode-cap-usd", type=float, default=5.0,
                    help="the agent stops before a request that could take the episode past this")
    ap.add_argument("--hf-bill-to", default=os.environ.get("HF_BILL_TO"),
                    help="bill the agent's Hugging Face router calls to this organization (default $HF_BILL_TO)")
    ap.add_argument("--smoke", action="store_true",
                    help="the three wiring tasks: no model, no agent; passes on scores 1.0, 0.176471 and 0")
    ap.add_argument("--keep", action="store_true", help="keep the sandboxes up until Enter")
    args = ap.parse_args()

    model = None
    if not args.smoke:
        model = model_id(args.model)
        if model not in (priced := priced_models()):
            print(f"no price for {model}: portsim-llm plays {', '.join(priced)}", file=sys.stderr)
            return 2
        if (base := endpoint()) is None:
            print("no model endpoint: set LITELLM_BASE_URL and LITELLM_API_KEY, [model] in .agentenv/config.toml, or "
                  "HF_TOKEN for the Hugging Face router", file=sys.stderr)
            return 2
        if base == HF_ROUTER and ":" not in model:
            print(f"the Hugging Face router doesn't serve {model}: set LITELLM_BASE_URL and LITELLM_API_KEY or [model] "
                  "in .agentenv/config.toml", file=sys.stderr)
            return 2
    if why := docker_unreachable():
        print(f"agent-env runs the env in Docker, and {why}", file=sys.stderr)
        return 2

    pack = load_pack()
    if args.smoke:
        task = pack.get(tasks.WIRING_TASK)
        files = tasks.wiring_tasks(task)
        chosen = dict.fromkeys(files, task)
    else:
        try:
            picked = ([pack.get(i) for i in args.task_id] if args.task_id
                      else [t for t in pack.tasks if t.split == args.split][:args.tasks])
        except KeyError as e:
            print(f"unknown task id {e}", file=sys.stderr)
            return 2
        files, chosen = {}, {}
        for task in picked:
            task_steps = tasks.steps(task, args.episode_cap_usd, args.hf_bill_to)
            next(s for s in task_steps if s["id"] == "play")["max_turns"] = args.max_turns
            files[task.task_id], chosen[task.task_id] = task_steps, task
    stage(files, agent=not args.smoke)

    try:
        result = run_bundle(BUNDLE, tasks=list(files), model=model, keep=args.keep,
                            on_progress=lambda line: print(line, flush=True))
    except BundleError as e:
        print(e, file=sys.stderr)
        return 2
    except RunInterrupted as stopped:
        report(stopped.result, chosen, model, args.smoke)
        return 128 + stopped.signum
    code = report(result, chosen, model, args.smoke)
    if args.keep:
        try:
            input("The sandboxes are up; press Enter to tear them down. ")
        finally:
            result.teardown()
    return code


if __name__ == "__main__":
    sys.exit(main())
