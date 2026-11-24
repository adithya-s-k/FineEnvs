"""Run PortSimEnv live on AgentEnv: one model on a few weeks of portsim-live, portsim-marine, portsim-wind or
portsim-lines, each played in watches by the portsim-llm agent (on portsim-lines with a portsim-line agent for each firm
with a ship in the week) and graded by portsim-live-verifier, and print what happened.

    uv run python rollout.py --smoke                                       # no model or key: wiring-noplay scores 0
    uv run python rollout.py --smoke --env marine
    HF_TOKEN=hf_... uv run python rollout.py                               # GLM-5.3-Flash on the HF router, 1 week
    uv run python rollout.py --env marine --model anthropic/claude-sonnet-5-5 --tasks 3   # through LITELLM_* or [model]
    uv run python rollout.py --env wind --task-id dock-36A-w06x1-standard-0-e07           # after tools/build_wind.py
    uv run python rollout.py --env lines --model openai/gpt-6-astra                       # after tools/build_lines.py

It stages a bundle in runs/bundle (the env's Docker build context, the agents from portsim-v1, the verifier and one
task file per week) and runs it in this process with agent-env, which builds the images and the gateway on first use,
starts its own registry on 127.0.0.1:5000 and keeps its stores under ~/.local/state/agent-env. Each run writes
runs/<UTC>/results.jsonl and runs/<UTC>/transcripts/<task>.json. Exit 0 when every run finished, 1 when one failed (or
--smoke scored wrong), 2 when a check or the bundle failed before anything ran, 128 + the signal number when
interrupted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
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
from berth_core import Task, TaskPack
from port_week_core import lines, marine, packs, wind
from port_week_core.schedule import schedule

from port_week_agentenv import tasks
from port_week_agentenv.contract import LINE_AGENT, LINES_ENV, LIVE_ENV, MARINE_ENV, WIND_ENV

HERE = Path(__file__).resolve().parent
SIMULATION = HERE.parents[3]
V1 = SIMULATION / "portsim-v1" / "envs" / "berth_planning"
WEEK = HERE.parent
PINS = WEEK.parents[1] / "data" / "meteocat" / "pins.json"
AGENT = V1 / "agentenv" / "agent"
LINE_AGENT_DIR = V1 / "agentenv" / "line_agent"
RUNS = HERE / "runs"
BUNDLE = RUNS / "bundle"
ENVS = {"live": LIVE_ENV, "marine": MARINE_ENV, "wind": WIND_ENV, "lines": LINES_ENV}
WIRING_TASKS = {LIVE_ENV: "dock-24B-w07x1-busy-0", MARINE_ENV: "dock-24B-w07x1-busy-0",
                WIND_ENV: "dock-24B-w06x1-busy-0-e15", LINES_ENV: "dock-24B-w06x1-busy-0-e15"}
BUILT = {WIND_ENV: (packs.WIND_PACK, "build_wind.py"), LINES_ENV: (packs.LINES_PACK, "build_lines.py")}
SMOKE = "wiring-noplay"
HF_ROUTER = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "zai-org/GLM-5.3-Flash:baseten"
SKIP = shutil.ignore_patterns("__pycache__", "*.egg-info", "build", "tests", ".pytest_cache", ".venv")


def priced_models() -> dict[str, tuple]:
    """The agent's PRICES, read without importing it (it needs the agent image's packages)."""
    for node in ast.parse((AGENT / "agent.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PRICES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise LookupError(f"{AGENT / 'agent.py'} has no PRICES")


def model_id(spec: str) -> str:
    """portsim-v1's openenv spellings as portsim-llm's ids: hf:M is M, anthropic:X and openai:X are anthropic/X and
    openai/X."""
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


def drifted(env: str) -> bool:
    """The built wind or lines weeks differ from the ones the published runs played, and the build's verify didn't
    accept it."""
    pack = BUILT[env][0]
    body = (packs.tasks_root() / pack / "tasks.jsonl").read_bytes()
    published = json.loads(PINS.read_text())["outputs"][f"{pack}/tasks.jsonl"]["sha256"]
    manifest = json.loads((packs.tasks_root() / pack / "manifest.json").read_text())
    return hashlib.sha256(body).hexdigest() != published and "drift" not in manifest


def week_pack(env: str) -> TaskPack:
    return (lines.pack() if env == LINES_ENV else wind.pack() if env == WIND_ENV else marine.pack()
            if env == MARINE_ENV else packs.live_pack())


def references(env: str) -> dict[str, dict]:
    refs = (lines.references() if env == LINES_ENV else wind.references() if env == WIND_ENV else marine.references()
            if env == MARINE_ENV else packs.live_references())
    return {r["task_id"]: r for r in refs}


def watches(env: str, task: Task) -> int:
    return len(lines.timetable(task) if env == LINES_ENV else schedule(task))


def qualifying(env: str) -> list[str]:
    """The weeks where 1.0 is reachable live, fewest watches first."""
    watches = {t: len(r["watch_hours"]) for t, r in references(env).items() if r["qualifies"]}
    return sorted(watches, key=lambda t: (watches[t], t))


def stage(env: str, task_files: dict[str, list[dict]], agent: bool) -> None:
    """runs/bundle: the env folder holds what agentenv/Dockerfile copies from 07-simulation-environments, so the same
    Dockerfile builds it; the wind weeks only for portsim-wind and the lines weeks only for portsim-lines, and never
    the weather or the references. portsim-lines adds the portsim-line agent."""
    shutil.rmtree(BUNDLE, ignore_errors=True)
    folder = BUNDLE / "envs" / env
    for source in (V1 / "core", V1 / "tasks" / "dock-v1-eval", V1 / "tasks" / "dock-v1-train", WEEK / "core",
                   HERE / "port_week_agentenv"):
        shutil.copytree(source, folder / source.relative_to(SIMULATION), ignore=SKIP)
    shutil.copy2(HERE / "pyproject.toml", folder / (HERE / "pyproject.toml").relative_to(SIMULATION))
    built = [pack for e, (pack, _) in BUILT.items() if e != env]
    shutil.copytree(packs.tasks_root(), folder / (WEEK / "tasks").relative_to(SIMULATION),
                    ignore=shutil.ignore_patterns("wind", "lines", *built))
    shutil.copy2(HERE / "Dockerfile", folder / "Dockerfile")
    (folder / "env.toml").write_text(f'environment_name = "{env}"\nenv_provider_type = "gateway"\n')
    if agent:
        shutil.copytree(AGENT, BUNDLE / "agents" / "portsim-llm", ignore=SKIP)
        if env == LINES_ENV:
            shutil.copytree(LINE_AGENT_DIR, BUNDLE / "agents" / LINE_AGENT, ignore=SKIP)
    verifier = BUNDLE / "artifacts" / "portsim-live-verifier"
    verifier.mkdir(parents=True)
    shutil.copy2(HERE / "verifier" / "verify.py", verifier / "verify.py")
    (BUNDLE / "tasks").mkdir()
    for name, task_steps in task_files.items():
        (BUNDLE / "tasks" / f"{name}.json").write_bytes(tasks.encode(task_steps))


def trajectory(url: str) -> dict:
    with namespace_routing():
        return json.loads(get_config().get_object_store_at(url).get(url))


def row(name: str, env: str, task: Task, reference: dict | None, model: str | None,
        run: TaskRun) -> tuple[dict, dict | None]:
    """The run's results.jsonl row and the agent's record, from the context the run left."""
    verification = (run.context.metadata.get("verifications") or {}).get("portsim") or {}
    week = verification["results"][0]["episode"] if verification.get("results") else {}
    grade = week.get("grade") or {}
    play = next((p for p in run.context.prompt_responses if p.step_id == "play"), None)
    summary = (play.structured_output or {}) if play else {}
    record = trajectory(play.agent_trajectory_object_url) if play and play.agent_trajectory_object_url else None
    failed = run.failed_steps[-1] if run.failed_steps else {}
    error = (play and play.error_message) or failed.get("error") or (run.error and str(run.error))
    return {"task": name, "task_id": task.task_id, "env": env, "model": model, "outcome": run.outcome.value,
            "instance": run.instance_id, "seconds": round(run.duration, 1), "reward": verification.get("score"),
            "end_reason": week.get("end_reason"), "audit_ok": (week.get("audit") or {}).get("ok"),
            "feasible": grade.get("feasible"), "plan_cost": grade.get("cost"),
            "optimal_cost": grade.get("optimal_cost"), "excused_cost": grade.get("excused_cost"),
            "regret": grade.get("regret"), "watches": week.get("watches"),
            "checks": sum(week["planning_calls"]) if week else None, "calls": week.get("calls_used"),
            "rolling_reward": reference and reference["rolling"]["reward"],
            "naive_reward": reference and reference["naive"]["reward"],
            **({"anchor_hours": grade.get("anchor_hours"),
                "messages": len(week["lines"]["exchanges"]) if week else None} if env == LINES_ENV else {}),
            **{key: summary.get(key) for key in ("turns", "tool_calls", "input_tokens", "output_tokens",
                                                 "cached_tokens", "cache_write_tokens", "cost_usd")},
            "failed_step": failed.get("step_id"), "error_code": (play and play.error_code) or failed.get("error_type"),
            "error": error and error[:500]}, record


def show(r: dict, task: Task, record: dict | None) -> None:
    print(f"\n=== {r['task']} ({task.task_id}, {task.difficulty}, {len(task.ships)} ships, "
          f"{watches(r['env'], task)} watches on {r['env']}) ===")
    for m in (record or {}).get("messages", [])[2:]:
        if m["role"] == "assistant":
            calls = ", ".join(c["name"] for c in m.get("tool_calls") or [])
            print(f"assistant: {m['content'][:200]!r} -> {calls or 'no tool call'}")
        elif m["role"] == "tool":
            print(f"  {m['name']}: {m['content'][:240]}")
    agent = "" if r["turns"] is None else f"  turns {r['turns']}  ${r['cost_usd'] or 0:.4f}"
    print(f"reward {r['reward']}  end {r['end_reason']}  audit {r['audit_ok']}  cost {r['plan_cost']}  "
          f"optimum {r['optimal_cost']}  regret {r['regret']}  excused {r['excused_cost']}  "
          f"rolling {r['rolling_reward']}  naive {r['naive_reward']}{agent}  {r['outcome']}  instance {r['instance']}")
    if r["error"]:
        print(f"failed_step {r['failed_step']}  error_code {r['error_code']}  error {r['error']}")


def report(result: BundleRun, env: str, chosen: dict[str, Task], refs: dict[str, dict], model: str | None,
           smoke: bool) -> int:
    out = RUNS / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out / "transcripts").mkdir(parents=True)
    code = 0
    with (out / "results.jsonl").open("w") as results:
        for run in result.runs:
            name = run.entry.name
            task = chosen[name]
            r, record = row(name, env, task, refs.get(task.task_id), model, run)
            results.write(json.dumps(r) + "\n")
            if record is not None:
                (out / "transcripts" / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
            show(r, task, record)
            wrong = smoke and (r["reward"] != 0 or r["audit_ok"] is not True)
            if wrong:
                print("expected reward 0 with the audit passing")
            if wrong or run.outcome in (Outcome.FAILED, Outcome.CANCELLED):
                code = 1
    print(f"\nwrote {out.relative_to(HERE)}/results.jsonl")
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", default="live", choices=list(ENVS), help="the env to play (default live)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"portsim-llm's model (default {DEFAULT_MODEL})")
    ap.add_argument("--tasks", type=int, default=1, help="the first N qualifying weeks, fewest watches first")
    ap.add_argument("--task-id", action="append", default=[], help="a week to play (repeatable; overrides --tasks)")
    ap.add_argument("--episode-cap-usd", type=float, default=5.0,
                    help="the agent stops before a request that could take the episode past this")
    ap.add_argument("--hf-bill-to", default=os.environ.get("HF_BILL_TO"),
                    help="bill the agent's Hugging Face router calls to this organization (default $HF_BILL_TO)")
    ap.add_argument("--smoke", action="store_true",
                    help="wiring-noplay: no model, no agent; passes on reward 0 with the audit passing")
    ap.add_argument("--keep", action="store_true", help="keep the sandboxes up until Enter")
    args = ap.parse_args()
    env = ENVS[args.env]

    try:
        pack, refs = week_pack(env), references(env)
    except wind.NotBuilt as e:
        print(e, file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"{e}: PORT_WEEK_TASKS_DIR, when set, must also hold the committed dock-v1-marine/, live/ and marine/",
              file=sys.stderr)
        return 2
    if env in BUILT and drifted(env):
        print(f"the built {args.env} weeks differ from the published ones: python tools/{BUILT[env][1]} verify lists "
              "them, and --accept-drift lets them play", file=sys.stderr)
        return 2
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

    if args.smoke:
        task = pack.get(WIRING_TASKS[env])
        files, chosen = {SMOKE: tasks.wiring_noplay(env, task)}, {SMOKE: task}
    else:
        try:
            picked = [pack.get(i) for i in args.task_id or qualifying(env)[:args.tasks]]
            files = {t.task_id: tasks.steps(env, t, args.episode_cap_usd, args.hf_bill_to) for t in picked}
        except KeyError as e:
            print(f"unknown task id {e} on {env}", file=sys.stderr)
            return 2
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        chosen = {t.task_id: t for t in picked}
    stage(env, files, agent=not args.smoke)

    try:
        result = run_bundle(BUNDLE, tasks=list(files), model=model, keep=args.keep,
                            on_progress=lambda line: print(line, flush=True))
    except BundleError as e:
        print(e, file=sys.stderr)
        return 2
    except RunInterrupted as stopped:
        report(stopped.result, env, chosen, refs, model, args.smoke)
        return 128 + stopped.signum
    code = report(result, env, chosen, refs, model, args.smoke)
    if args.keep:
        try:
            input("The sandboxes are up; press Enter to tear them down. ")
        finally:
            result.teardown()
    return code


if __name__ == "__main__":
    sys.exit(main())
