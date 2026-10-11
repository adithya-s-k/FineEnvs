"""The AgentEnv task steps for PortSimEnv v1: one task per PortSim task, played by portsim-llm with upstream's prompts
and limits and graded by portsim-verifier, and the three wiring tasks, which need no model and no agent."""

import json

from berth_core import Task, situation
from berth_core import rules as berth_rules

from .server import MAX_CHECKS

SUBMIT_PLAN_URI = "urn:portsim:submit-plan/v1"
WIRING_TASK = "dock-24B-w07x1-busy-0"
OPENING = ("{situation}\n\nMake the new berth plan. Use check_plan to test drafts, then call submit_plan with your "
           "final plan.")


def agent_env_vars(episode_cap_usd: float, hf_bill_to: str | None) -> dict[str, str]:
    return {"PORTSIM_MAX_COST_USD": format(episode_cap_usd, "g"), **({"HF_BILL_TO": hf_bill_to} if hf_bill_to else {})}


def steps(task: Task, episode_cap_usd: float, hf_bill_to: str | None = None) -> list[dict]:
    return [
        {"id": "deploy", "type": "deploy_env", "env_id": "portsim"},
        {"id": "load-task", "type": "apply_server_config", "env_id": "portsim",
         "directives": [{"service": "portsim", "uri": "urn:portsim:load-task/v1", "args": {"task_id": task.task_id}}]},
        {"id": "deploy-agent", "type": "deploy_agent", "a2a_agent_id": "portsim-llm", "env_ids": ["portsim"],
         "env_vars": agent_env_vars(episode_cap_usd, hf_bill_to)},
        {"id": "play", "type": "prompt_agent", "prompt_id": task.task_id,
         "system_prompt": berth_rules(task, MAX_CHECKS), "prompt": OPENING.format(situation=situation(task)),
         "max_turns": 12, "model_params": {"max_tokens": 32000}, "timeout_seconds": 7200},
        {"id": "grade", "type": "env_outcome_verifier", "env_id": "portsim", "file_artifact_id": "portsim-verifier",
         "verifier_id": "portsim", "score_aggregator": "weighted_average"},
    ]


def wiring_steps(task: Task, plan: list[dict] | None = None) -> list[dict]:
    """Load ``task``, submit ``plan`` through urn:portsim:submit-plan/v1 as the submit_plan tool would (nothing when it
    is None) and grade: the play steps of ``steps`` without a model or an agent."""
    deploy, load, *_, grade = steps(task, 0)
    submit = [{"id": "submit", "type": "apply_server_config", "env_id": "portsim",
               "directives": [{"service": "portsim", "uri": SUBMIT_PLAN_URI, "args": {"plan": plan}}]}]
    return [deploy, load, *(submit if plan is not None else []), grade]


def wiring_tasks(task: Task) -> dict[str, list[dict]]:
    """smoke submits the task's stored optimal plan (1.0), wiring-infeasible that plan with ship 1 berthed on top of
    ship 0 (at most 0.2) and wiring-nosubmit nothing (0)."""
    optimal = task.reference["optimal_plan"]
    first, second, *rest = optimal
    clash = [first, {**second, "berth_hour": first["berth_hour"], "section": first["section"]}, *rest]
    return {"smoke": wiring_steps(task, optimal), "wiring-infeasible": wiring_steps(task, clash),
            "wiring-nosubmit": wiring_steps(task)}


def encode(task_steps: list[dict]) -> bytes:
    """A task file as a bundle holds it."""
    return (json.dumps(task_steps, indent=2, ensure_ascii=False) + "\n").encode()
