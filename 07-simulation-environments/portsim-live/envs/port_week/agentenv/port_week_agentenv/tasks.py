"""The AgentEnv task steps for a week played live: load it, hide port_notice, register a trigger per watch, play it
with portsim-llm on the gateway's frozen clock, end it and grade it with portsim-live-verifier; on portsim-lines, each
firm with a ship in the week also deployed as its own portsim-line agent and peered with the planner; and
wiring-noplay, the same steps without the agents, which needs no model and scores 0 with the audit passing."""

import json

from berth_core import Task
from port_week_core import lines
from port_week_core.marine import MarineWeek
from port_week_core.prompts import LINES_RULES, LIVE_OPENING, LIVE_RULES, WIND_RULES, live_rules
from port_week_core.schedule import LINES_MAX_TURNS, MAX_TURNS, WIND_MAX_TURNS, schedule
from port_week_core.wind import WindWeek
from port_week_core.world import Week

from .contract import (
    END_WEEK_URI,
    LINE_AGENT,
    LINE_ROLE,
    LINE_TOOLS,
    LINES_ENV,
    LIVE_ENV,
    LIVE_LOAD_URI,
    MARINE_ENV,
    NOTICE_TOOL,
    WIND_ENV,
    triggers,
)

AGENTS = ("deploy_agent", "peer_agents", "prompt_agent")
VARIANTS = {LIVE_ENV: {}, MARINE_ENV: {"week": MarineWeek},
            WIND_ENV: {"week": WindWeek, "rules": WIND_RULES, "max_turns": WIND_MAX_TURNS}}
"""The keyword arguments of live_steps for each env, as the agentenv-portsim plugin's `tasks generate` passes them."""


def agent_env_vars(episode_cap_usd: float | None, hf_bill_to: str | None) -> dict[str, str]:
    """The agent's environment: a cost cap only when one is asked for (a sweep's), so a published task carries none."""
    return ({"PORTSIM_MAX_COST_USD": format(episode_cap_usd, "g")} if episode_cap_usd is not None else {}) | (
        {"HF_BILL_TO": hf_bill_to} if hf_bill_to else {})


def agent_env(episode_cap_usd: float | None, hf_bill_to: str | None) -> dict[str, dict[str, str]]:
    env = agent_env_vars(episode_cap_usd, hf_bill_to)
    return {"env_vars": env} if env else {}


def live_steps(task: Task, episode_cap_usd: float | None = None, hf_bill_to: str | None = None, *, env: str = LIVE_ENV,
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
         **agent_env(episode_cap_usd, hf_bill_to)},
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


def lines_steps(task: Task, episode_cap_usd: float | None = None, hf_bill_to: str | None = None) -> list[dict]:
    """The wind week's steps on portsim-lines, with each firm that has a ship in the week deployed as its own
    portsim-line agent under the line role, which sees only the agents' two tools, and peered with the planner."""
    env, firms = LINES_ENV, sorted(task.rules["lines"]["firms"])
    steps = live_steps(task, episode_cap_usd, hf_bill_to, env=env, week=lines.LinesWeek, rules=LINES_RULES,
                       max_turns=LINES_MAX_TURNS)
    by_id = {s["id"]: s for s in steps}
    by_id["hide-port-notice"]["tools"] = [NOTICE_TOOL, *LINE_TOOLS]
    by_id["watches"]["triggers"] = triggers(lines.timetable(task))
    roles = [{"id": "agents-only", "type": "modify_env_tool_access", "env_id": env, "action": "disable",
              "role": LINE_ROLE, "tools": ["*"]},
             {"id": "agents-tools", "type": "modify_env_tool_access", "env_id": env, "action": "enable",
              "role": LINE_ROLE, "tools": list(LINE_TOOLS)}]
    agents = [{"id": f"deploy-{f}", "type": "deploy_agent", "a2a_agent_id": LINE_AGENT, "agent_name": f,
               "role": LINE_ROLE, "env_ids": [env]} for f in firms]
    peers = {"id": "peers", "type": "peer_agents",
             "peerings": [{"source_agent_name": "default-agent", "peer_agent_names": firms}]}
    out = []
    for step in steps:
        out.append(step)
        if step["id"] == "hide-port-notice":
            out += roles
        elif step["id"] == "deploy-agent":
            out += agents + [peers]
    return out


def steps(env: str, task: Task, episode_cap_usd: float | None = None, hf_bill_to: str | None = None) -> list[dict]:
    """The week's steps on ``env``: portsim-live, portsim-marine, portsim-wind or portsim-lines."""
    if env == LINES_ENV:
        return lines_steps(task, episode_cap_usd, hf_bill_to)
    return live_steps(task, episode_cap_usd, hf_bill_to, env=env, **VARIANTS[env])


def wiring_noplay(env: str, task: Task) -> list[dict]:
    """The week's steps without the agents: end-week delivers every watch's notices itself and the week ends with no
    window confirmed."""
    return [s for s in steps(env, task) if s["type"] not in AGENTS]


def encode(task_steps: list[dict]) -> bytes:
    """A task file as a bundle holds it."""
    return (json.dumps(task_steps, indent=2, ensure_ascii=False) + "\n").encode()
