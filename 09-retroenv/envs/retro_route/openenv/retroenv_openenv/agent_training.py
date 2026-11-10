"""The training harness as an evaluation agent: what a policy sees during RL, for any model.

RL (train/async_grpo.py, TRL's tool loop) shows the model `RemoteRetroRouteEnv`'s methods as its
tools (transformers' `get_json_schema` of each, frozen in training_tools.json), answers every call
with that class's result string (calls left, a reminder near the budget, a malformed submission
handed back up to twice), allows 20 model turns and ends an episode at the first turn without a
tool call. Nothing forces or auto-submits a route: an episode that never calls emit_routes scores
0. The evaluation board's agent (agent.py) differs on each of these, and a policy trained in one
context did not carry its gains to the other. This runs the training loop over an
OpenAI-compatible endpoint, so a base model, a trained checkpoint and an API model are measured in
the context RL optimises.

Regenerate training_tools.json after changing a tool's signature or docstring:
    uv run --with transformers python -c "import inspect, json; from transformers.utils import get_json_schema; \\
      from retroenv_openenv.client import RemoteRetroRouteEnv as E; e = E.__new__(E); \\
      print(json.dumps([get_json_schema(getattr(e, n)) for n, _ in inspect.getmembers(E, inspect.isfunction) \\
      if n not in ('reset', 'get_reward') and not n.startswith('_')], indent=1))"
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import SYSTEM_PROMPT, AgentConfig, _request_with_retry, recorded
from .client import RemoteRetroRouteEnv, RetroEnvClient

TOOLS: list[dict[str, Any]] = json.loads(Path(__file__).with_name("training_tools.json").read_text())
TOOL_NAMES = {tool["function"]["name"] for tool in TOOLS}
MAX_TURNS = 20  # train/async_grpo.py: max_tool_calling_iterations
PROMPT_TURNS = 16  # the system prompt training sends says 16 turns (train/sync_grpo.py task_dataset)
# vLLM's refusal of a turn whose prompt plus max_tokens passes the context.
CONTEXT_FULL = re.compile(r"maximum context length is (\d+) tokens.*?prompt contains at least (\d+) input tokens", re.S)


def _request(llm: Any, config: AgentConfig, messages: list[dict[str, Any]]) -> Any:
    """One turn. Near the end of the context, the turn gets the tokens left, as in training.

    TRL caps each turn at what the episode's context has left; an endpoint refuses a request whose
    max_tokens passes it, which would end an episode that training would have let finish.
    """
    for _ in range(3):
        try:
            return _request_with_retry(llm, config, messages, TOOLS)
        except Exception as exc:
            match = CONTEXT_FULL.search(str(exc))
            left = int(match.group(1)) - int(match.group(2)) if match else 0
            if left < 1 or left >= config.max_tokens:
                raise
            config = replace(config, max_tokens=left)
    return _request_with_retry(llm, config, messages, TOOLS)


def run_episode(llm: Any, env: RetroEnvClient, opening: dict[str, Any], config: AgentConfig) -> dict[str, Any]:
    trainee = RemoteRetroRouteEnv.attached(env, opening)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_turns=PROMPT_TURNS)},
        {"role": "user", "content": opening["prompt"]},
    ]
    transcript: list[dict[str, Any]] = []
    errors: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reported_cost_usd": 0.0}
    resolved_models: set[str] = set()
    score: dict[str, Any] | None = None
    turns = 0
    started = time.perf_counter()

    for _ in range(config.max_turns):
        try:
            response = _request(llm, config, messages)
        except Exception as exc:  # provider errors end the episode; it is still scored
            errors.append(f"API error: {type(exc).__name__}: {str(exc)[:500]}")
            break
        turns += 1
        if response.usage:
            usage["prompt_tokens"] += int(response.usage.prompt_tokens or 0)
            usage["completion_tokens"] += int(response.usage.completion_tokens or 0)
            extra = getattr(response.usage, "model_extra", None) or {}
            usage["reported_cost_usd"] += float(extra.get("cost") or 0.0)
        if getattr(response, "model", None):
            resolved_models.add(str(response.model))
        if not response.choices:
            errors.append("API returned no choices")
            break
        message = response.choices[0].message
        calls = message.tool_calls or []
        assistant: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if calls:
            assistant["tool_calls"] = [call.model_dump(exclude_none=True) for call in calls]
        transcript.append(recorded(assistant, message))
        thinking = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        if config.send_reasoning and thinking:
            # vLLM hands an earlier turn's `reasoning` to the chat template as reasoning_content
            # (the field older releases read), which Qwen's template keeps under preserve_thinking.
            assistant = {**assistant, "reasoning": str(thinking), "reasoning_content": str(thinking)}
        messages.append(assistant)
        if not calls:  # as in training: a turn without a tool call ends the episode
            break
        for call in calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
                if name not in TOOL_NAMES:
                    raise ValueError(f"unknown tool {name!r}")
                content = getattr(trainee, name)(**arguments)
            except (ValueError, TypeError) as exc:
                content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                errors.append(f"{name}: {exc}")
            if name == "emit_routes" and trainee.submitted and score is None:
                score = (json.loads(content).get("score") or {}) if content.startswith("{") else {}
            tool_message = {"role": "tool", "tool_call_id": call.id, "name": name, "content": content}
            messages.append(tool_message)
            transcript.append(tool_message)
        if score is not None:  # graded: later turns cannot change the reward
            break

    submitted = score is not None
    metrics = (score or {}).get("metrics") or {}
    return {
        "reward": trainee._reward if submitted else 0.0,
        "valid": bool(submitted and trainee.passed),
        "exact_match": bool(metrics.get("reference_match", False)),
        "components": (score or {}).get("components", {}),
        "hard_failures": (score or {}).get("hard_failures", []) if submitted else ["no submission"],
        "verification_tier": (score or {}).get("verification_tier"),
        "route_count": metrics.get("route_count", 0),
        "submission": trainee._submission,
        "auto_emitted": False,
        "submission_coerced": False,
        "json_retries": trainee._json_retries,
        "tool_calls": trainee._calls + int(submitted),
        "turns": turns,
        "errors": errors,
        "usage": {
            **usage,
            "reported_cost_usd": round(usage["reported_cost_usd"], 8),
            "latency_seconds": round(time.perf_counter() - started, 3),
        },
        "resolved_models": sorted(resolved_models),
        "transcript": transcript,
    }
