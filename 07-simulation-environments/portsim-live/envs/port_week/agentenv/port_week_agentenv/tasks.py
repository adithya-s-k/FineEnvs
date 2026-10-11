"""The AgentEnv task steps for a week played live: load it, hide port_notice, register a trigger per watch, play it
with portsim-llm on the gateway's frozen clock, end it and grade it with portsim-live-verifier; and wiring-noplay, the
same steps without the agent, which needs no model and scores 0 with the audit passing."""

import json

from berth_core import Task
from port_week_core.marine import MarineWeek
from port_week_core.prompts import LIVE_OPENING, LIVE_RULES, WIND_RULES, live_rules
from port_week_core.schedule import MAX_TURNS, WIND_MAX_TURNS, schedule
from port_week_core.wind import WindWeek
from port_week_core.world import Week

from .contract import END_WEEK_URI, LIVE_ENV, LIVE_LOAD_URI, MARINE_ENV, NOTICE_TOOL, WIND_ENV, triggers

PLAY = ("deploy-agent", "play")
VARIANTS = {LIVE_ENV: {}, MARINE_ENV: {"week": MarineWeek},
            WIND_ENV: {"week": WindWeek, "rules": WIND_RULES, "max_turns": WIND_MAX_TURNS}}
"""The keyword arguments of live_steps for each env, as the agentenv-portsim plugin's `tasks generate` passes them."""


def agent_env_vars(episode_cap_usd: float, hf_bill_to: str | None) -> dict[str, str]:
    return {"PORTSIM_MAX_COST_USD": format(episode_cap_usd, "g"), **({"HF_BILL_TO": hf_bill_to} if hf_bill_to else {})}


def live_steps(task: Task, episode_cap_usd: float, hf_bill_to: str | None = None, *, env: str = LIVE_ENV,
               week: type[Week] = Week, rules: str = LIVE_RULES, max_turns: int = MAX_TURNS) -> list[dict]:
    watches = schedule(task)
    return [
        {"id": "deploy", "type": "deploy_env", "env_id": env},
        {"id": "load-task", "type": "apply_server_config", "env_id": env,
         "directives": [{"service": env, "uri": LIVE_LOAD_URI, "args": {"task_id": task.task_id}}]},
        {"id": "hide-port-notice", "type": "modify_env_tool_access", "env_id": env, "action": "disable",
         "role": "default", "tools": [NOTICE_TOOL]},
        {"id": "watches", "type": "register_env_triggers", "env_id": env, "watch_roles": ["default"],
         "triggers": triggers(watches)},
        {"id": "deploy-agent", "type": "deploy_agent", "a2a_agent_id": "portsim-llm", "env_ids": [env],
         "env_vars": agent_env_vars(episode_cap_usd, hf_bill_to)},
        {"id": "clock", "type": "sync_env_clock", "env_id": env, "virtual_time": task.week_start_utc,
         "virtual_seconds_per_real_second": 0, "tolerate_missing_sync_time": False},
        {"id": "play", "type": "prompt_agent", "prompt_id": task.task_id, "system_prompt": live_rules(task, rules),
         "prompt": LIVE_OPENING.format(situation=week(task).situation()["situation"]), "max_turns": max_turns,
         "model_params": {"max_tokens": 32000}, "timeout_seconds": 7200},
        {"id": "end-week", "type": "apply_server_config", "env_id": env,
         "directives": [{"service": env, "uri": END_WEEK_URI, "args": {}}]},
        {"id": "grade", "type": "env_outcome_verifier", "env_id": env, "file_artifact_id": "portsim-live-verifier",
         "verifier_id": "portsim", "score_aggregator": "weighted_average"},
    ]


def steps(env: str, task: Task, episode_cap_usd: float, hf_bill_to: str | None = None) -> list[dict]:
    """The week's steps on ``env``: portsim-live, portsim-marine or portsim-wind."""
    return live_steps(task, episode_cap_usd, hf_bill_to, env=env, **VARIANTS[env])


def wiring_noplay(env: str, task: Task) -> list[dict]:
    """The week's steps without the agent: end-week delivers every watch's notices itself and the week ends with no
    window confirmed."""
    return [s for s in steps(env, task, 5.0) if s["id"] not in PLAY]


def encode(task_steps: list[dict]) -> bytes:
    """A task file as a bundle holds it."""
    return (json.dumps(task_steps, indent=2, ensure_ascii=False) + "\n").encode()
