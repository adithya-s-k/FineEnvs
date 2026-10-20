"""Run AirOpsEnv on AgentEnv: a few days of airops-day (v1), each played in watches by the airops-llm agent and graded
by airops-verifier, or of airops-occ (v2), each run by two airops-desk agents (joint) or one at both desks (one) and
graded by airops-occ-verifier, and print what happened.

    uv run -p 3.12 python rollout.py --smoke                            # no model: v1's wiring-noplay, the naive day
    uv run -p 3.12 python rollout.py --env occ --smoke                  # no model: v2's wiring-noplay, the crew system
    uv run -p 3.12 python rollout.py --env occ --script                 # no model: v2's scripted desks, both modes
    uv run -p 3.12 python rollout.py --model anthropic/claude-haiku-5-5 # v1, the eval day with the fewest watches
    uv run -p 3.12 python rollout.py --env occ --model openai/gpt-6-luna --mode one --task-id b6-20250111-0400

It needs the fetched packs (tools/fetch_packs.py). It stages a bundle in runs/bundle (the env's Docker build context,
the agent, the verifier and one task file per day) and runs it in this process with agent-env, which builds the images
and the gateway on first use, starts its own registry on 127.0.0.1:5000 and keeps its stores under
~/.local/state/agent-env. Each run writes runs/<UTC>/results.jsonl and runs/<UTC>/transcripts/. Exit 0 when every run
scored as expected, 1 when one failed (or a zero-spend run scored wrong), 2 when a check or the bundle failed before
anything ran, 128 + the signal number when interrupted.
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
from airops_core.occ.pack import Pack
from airops_core.schedule import AGENT, ENV, VERIFIER, timetable

from airops_agentenv import tasks
from airops_agentenv.occ import tasks as occ_tasks
from airops_agentenv.occ.live import ENV as OCC_ENV

HERE = Path(__file__).resolve().parent
ENV_ROOT = HERE.parent
sys.path.insert(0, str(ENV_ROOT / "tools"))
import fetch_packs  # noqa: E402

RUNS = HERE / "runs"
BUNDLE = RUNS / "bundle"
SMOKE = {"day": "wiring-noplay", "occ": "occ-wiring-noplay"}
"""Each env's own name, so agent-env's local store keeps one task per env."""
HF_ROUTER = "https://router.huggingface.co/v1"
NO_MODEL = {"LITELLM_BASE_URL": "http://127.0.0.1:9", "LITELLM_API_KEY": "sk-no-model"}
"""deploy_agent wants a model endpoint: the scripted desks get one nothing listens on, with a key no provider knows."""
SKIP = shutil.ignore_patterns("__pycache__", "*.egg-info", "tests", ".pytest_cache", ".venv", ".ruff_cache")
SETUPS = {"day": (ENV, "Dockerfile", "b6-v1", AGENT, VERIFIER, "airops"),
          "occ": (OCC_ENV, "Dockerfile.occ", "b6-v2", occ_tasks.AGENT, occ_tasks.VERIFIER, "airops-occ")}
"""Per env: its id, Dockerfile, pack, agent, verifier artifact and verifier id."""


def priced_models(agent: str) -> dict[str, tuple]:
    """The agent's PRICES, read without importing it (it needs the agent image's packages)."""
    for node in ast.parse((HERE / "agents" / agent / "agent.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PRICES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise LookupError(f"agents/{agent}/agent.py has no PRICES")


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


def stage(env: str, task_files: dict[str, list[dict]], agent: bool) -> None:
    """runs/bundle: the env folder holds what its Dockerfile copies from envs/airops, so the same Dockerfile builds it,
    with its own pack only."""
    env_id, dockerfile, pack, agent_id, verifier, _ = SETUPS[env]
    shutil.rmtree(BUNDLE, ignore_errors=True)
    folder = BUNDLE / "envs" / env_id
    shutil.copytree(ENV_ROOT / "core", folder / "core", ignore=SKIP)
    shutil.copytree(HERE / "airops_agentenv", folder / "agentenv" / "airops_agentenv", ignore=SKIP)
    shutil.copy2(HERE / "pyproject.toml", folder / "agentenv" / "pyproject.toml")
    shutil.copytree(ENV_ROOT / "tasks" / pack, folder / "tasks" / pack)
    shutil.copy2(HERE / dockerfile, folder / "Dockerfile")
    (folder / "env.toml").write_text(f'environment_name = "{env_id}"\nenv_provider_type = "gateway"\n')
    if agent:
        shutil.copytree(HERE / "agents" / agent_id, BUNDLE / "agents" / agent_id, ignore=SKIP)
    (BUNDLE / "artifacts" / verifier).mkdir(parents=True)
    shutil.copy2(HERE / "verifiers" / verifier / "verify.py", BUNDLE / "artifacts" / verifier / "verify.py")
    (BUNDLE / "tasks").mkdir()
    for name, steps in task_files.items():
        (BUNDLE / "tasks" / f"{name}.json").write_text(json.dumps(steps, indent=2, ensure_ascii=False) + "\n")


def trajectory(url: str) -> dict:
    with namespace_routing():
        return json.loads(get_config().get_object_store_at(url).get(url))


def row(env: str, name: str, expected: float | None, model: str | None, run: TaskRun) -> tuple[dict, dict]:
    """The run's results.jsonl row, and each play step's agent record by step id."""
    verification = (run.context.metadata.get("verifications") or {}).get(SETUPS[env][5]) or {}
    day = verification["results"][0]["episode"] if verification.get("results") else {}
    grade = day.get("grade") or {}
    plays = [p for p in run.context.prompt_responses if p.step_id.startswith("play")]
    records = {p.step_id: trajectory(p.agent_trajectory_object_url) for p in plays if p.agent_trajectory_object_url}
    desks = {p.step_id: {k: (p.structured_output or {}).get(k) for k in (
        "end_reason", "turns", "tool_calls", "input_tokens", "output_tokens", "cached_tokens", "cost_usd")}
        for p in plays}
    failed = run.failed_steps[-1] if run.failed_steps else {}
    error = next((p.error_message for p in plays if p.error_message), None) or failed.get("error") or (
        run.error and str(run.error))
    cost = grade["cost"] if env == "occ" else grade.get("cost", {}).get("total")
    return {"task": name, "env": SETUPS[env][0], "model": model, "outcome": run.outcome.value,
            "instance": run.instance_id, "seconds": round(run.duration, 1), "reward": verification.get("score"),
            "expected_cost": expected, "audit_ok": (day.get("audit") or {}).get("ok"), "day_cost": cost,
            "reward_L": grade.get("reward_L"), "desks": desks,
            "spend_usd": round(sum(d["cost_usd"] or 0 for d in desks.values()), 6),
            "failed_step": failed.get("step_id"), "error": error and error[:500]}, records


def show(r: dict, records: dict) -> None:
    print(f"\n=== {r['task']} on {r['env']} ===")
    for step, record in records.items():
        for m in record.get("messages", [])[2:]:
            if m["role"] == "assistant":
                calls = ", ".join(c["name"] for c in m.get("tool_calls") or [])
                print(f"{step} assistant: {(m.get('content') or '')[:160]!r} -> {calls or 'no tool call'}")
            elif m["role"] == "tool":
                print(f"  {m['name']}: {m['content'][:200]}")
    print(f"reward {r['reward']}  audit {r['audit_ok']}  cost {r['day_cost']}  expected {r['expected_cost']}  "
          f"spend ${r['spend_usd']:.4f}  {r['outcome']}  instance {r['instance']}")
    if r["error"]:
        print(f"failed_step {r['failed_step']}  error {r['error']}")


def report(result: BundleRun, env: str, expected: dict[str, float | None], model: str | None, check: bool) -> int:
    out = RUNS / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out / "transcripts").mkdir(parents=True)
    code = 0
    with (out / "results.jsonl").open("w") as results:
        for run in result.runs:
            r, records = row(env, run.entry.name, expected[run.entry.name], model, run)
            results.write(json.dumps(r) + "\n")
            for step, record in records.items():
                (out / "transcripts" / f"{run.entry.name}-{step}.json").write_text(json.dumps(record, indent=1) + "\n")
            show(r, records)
            wrong = check and (r["reward"] is None or r["day_cost"] != r["expected_cost"] or r["audit_ok"] is not True
                               or r["spend_usd"] != 0)
            if wrong:
                print(f"expected a scored day costing {r['expected_cost']} with the audit passing and no spend")
            if wrong or run.outcome in (Outcome.FAILED, Outcome.CANCELLED):
                code = 1
    print(f"\nwrote {out.relative_to(HERE)}/results.jsonl")
    return code


def v1_files(args, model: str | None) -> tuple[dict[str, list[dict]], dict[str, float | None]]:
    days = tasks.pack_tasks()
    if args.smoke:
        first = next(t for t in days if t.split == "eval")
        return {SMOKE["day"]: tasks.noplay_steps(first)}, {SMOKE["day"]: first.reference["naive"]}
    by_id = {t.task_id: t for t in days}
    if unknown := [i for i in args.task_id if i not in by_id]:
        raise ValueError(f"unknown task ids on airops-day: {', '.join(unknown)}")
    evals = sorted((t for t in days if t.split == "eval"), key=lambda t: (len(timetable(t)), t.task_id))
    picked = [by_id[i] for i in args.task_id] or evals[:args.tasks]
    return {t.task_id: tasks.steps(t, args.episode_cap_usd, args.hf_bill_to) for t in picked}, dict.fromkeys(
        (t.task_id for t in picked), None)


def v2_files(args, model: str | None) -> tuple[dict[str, list[dict]], dict[str, float | None]]:
    pack = Pack()
    evals = sorted((t for t, v in pack.tasks.items() if v["split"] == "eval"), key=lambda t: pack.tasks[t]["date"])
    if unknown := [i for i in args.task_id if i not in pack.tasks]:
        raise ValueError(f"unknown task ids on airops-occ: {', '.join(unknown)}")
    picked = args.task_id or evals[:args.tasks]
    files, expected = {}, {}
    for task_id in [evals[0]] if args.smoke else picked:
        w = pack.world(task_id)
        refs = w.refs
        if args.smoke:
            files[SMOKE["occ"]], expected[SMOKE["occ"]] = occ_tasks.steps(w.task, "idle"), refs["idle"]
            continue
        for mode in ("joint", "one") if args.mode == "both" else (args.mode,):
            name = f"{task_id}-{mode}"
            if args.script:
                files[name] = occ_tasks.steps(w.task, mode, policy="script")
                expected[name] = refs["scripted"][mode]
            else:
                files[name], expected[name] = occ_tasks.steps(w.task, mode, cap=args.episode_cap_usd), None
    return files, expected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", default="day", choices=list(SETUPS), help="day: airops-day (v1); occ: airops-occ (v2)")
    ap.add_argument("--model", help="the agent's model (a LiteLLM id the agent prices)")
    ap.add_argument("--tasks", type=int, default=1, help="the first N eval days: v1 by fewest watches, v2 by date")
    ap.add_argument("--task-id", action="append", default=[], help="a day to play (repeatable; overrides --tasks)")
    ap.add_argument("--mode", default="both", choices=["joint", "one", "both"], help="v2: two desks, one, or both runs")
    ap.add_argument("--episode-cap-usd", type=float, default=2.0,
                    help="v1: the episode's cap; v2: each desk's; the agent stops before a request that could pass it")
    ap.add_argument("--hf-bill-to", default=os.environ.get("HF_BILL_TO"),
                    help="v1: bill the agent's Hugging Face router calls to this organization (default $HF_BILL_TO)")
    ap.add_argument("--smoke", action="store_true", help="wiring-noplay: no model, no agent")
    ap.add_argument("--script", action="store_true", help="v2: the scripted desks, no model; each day must cost the "
                                                          "pack's scripted reference")
    ap.add_argument("--keep", action="store_true", help="keep the sandboxes up until Enter")
    args = ap.parse_args()
    if args.script and args.env != "occ":
        ap.error("--script plays v2's scripted desks: add --env occ")
    if problems := fetch_packs.problems():
        print("\n".join(problems + ["fetch the packs first, from envs/airops: uv run -p 3.12 --with pyarrow python "
                                    "tools/fetch_packs.py"]), file=sys.stderr)
        return 2
    zero = args.smoke or args.script
    model = None
    if args.script:
        os.environ.update(NO_MODEL)
    elif not args.smoke:
        agent = SETUPS[args.env][3]
        if not args.model or args.model not in (priced := priced_models(agent)):
            print(f"--model must be one {agent} prices: {', '.join(priced)}", file=sys.stderr)
            return 2
        model = args.model
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
    try:
        files, expected = (v2_files if args.env == "occ" else v1_files)(args, model)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    stage(args.env, files, agent=not args.smoke)
    try:
        result = run_bundle(BUNDLE, tasks=list(files), model=model, keep=args.keep,
                            on_progress=lambda line: print(line, flush=True))
    except BundleError as e:
        print(e, file=sys.stderr)
        return 2
    except RunInterrupted as stopped:
        report(stopped.result, args.env, expected, model, zero)
        return 128 + stopped.signum
    code = report(result, args.env, expected, model, zero)
    if args.keep:
        try:
            input("The sandboxes are up; press Enter to tear them down. ")
        finally:
            result.teardown()
    return code


if __name__ == "__main__":
    sys.exit(main())
