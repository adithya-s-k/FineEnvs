"""OpenAI-compatible tool-calling agent loop against a RetroEnv server.

The final model turn exposes only emit_routes, two empty turns force that
terminal turn, and each tool result carries the turns left. Tools are
discovered from the server and every call goes through it, so the reward is
the server's. An episode that ends without emit_routes is closed by
submitting an empty route set, which scores the verifier's floor.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

import json_repair

from .client import RetroEnvClient

SYSTEM_PROMPT = (
    "You are a retrosynthesis planning agent. Plan with the tools: inspect molecules, search "
    "precedents, check each disconnection with validate_disconnection, and confirm every leaf "
    "with an exact stock_retrieve lookup before claiming in_stock=true. Respect the depth "
    "budget and any constraint in the task. Finish only with emit_routes. You have at most "
    "{max_turns} model turns and must reserve the final turn for emit_routes even if the routes "
    "are incomplete. Pass emit_routes.submission as a JSON object, not as a string. Do not "
    "reveal chain-of-thought; put short evidence-based explanations in reaction metadata."
)


# Shown on the last turn when the endpoint cannot force a tool call.
FINAL_TURN = (
    "This is your final turn. Call emit_routes now with the best route trees you have; no other tool is available."
)


def normalize_arguments(name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Undo one provider-side encoding slip: a route submission sent as a JSON string.

    Some models serialize a deeply nested argument into a string and then break
    its escaping or brackets. The harness repairs that string before the call
    (and flags it) so the score measures the routes, not the transport. Anything
    that does not repair into a route-set object is passed through for the
    verifier to judge.
    """
    value = arguments.get("submission")
    if name != "emit_routes" or not isinstance(value, str):
        return arguments, False
    decoded = json_repair.loads(value)
    if not isinstance(decoded, dict) or "routes" not in decoded:
        return arguments, False
    return {**arguments, "submission": decoded}, True


@dataclass
class AgentConfig:
    model: str
    max_turns: int = 16
    max_tokens: int = 4096
    # None omits the parameter (reasoning models reject explicit temperatures).
    temperature: float | None = 0.0
    tool_choice: str = "required"
    max_empty_turns: int = 2
    # "max_tokens" for most endpoints; OpenAI reasoning models want "max_completion_tokens".
    token_param: str = "max_tokens"
    parallel_tool_calls: bool | None = False
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


RETRY_DELAYS = (5, 15, 30, 60, 90, 120)
TRANSIENT = ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError")
# InternLM reports rate limits as HTTP 400 with this error code.
RATE_LIMIT_CODES = ("-20048",)


def _is_transient(exc: Exception) -> bool:
    return type(exc).__name__ in TRANSIENT or getattr(exc, "code", None) in RATE_LIMIT_CODES


def _request_with_retry(client: Any, config: AgentConfig, messages: list, tools: list) -> Any:
    """Retry rate limits, timeouts, 5xx and empty responses; other errors surface at once."""
    for delay in (*RETRY_DELAYS, None):
        try:
            response = _request(client, config, messages, tools)
        except Exception as exc:
            if delay is None or not _is_transient(exc):
                raise
        else:
            if response.choices or delay is None:
                return response
        time.sleep(delay + random.uniform(0, delay / 4))
    raise RuntimeError("unreachable")


def _request(client: Any, config: AgentConfig, messages: list, tools: list) -> Any:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": config.tool_choice,
        config.token_param: config.max_tokens,
    }
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = config.parallel_tool_calls
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
    if config.extra_body:
        kwargs["extra_body"] = config.extra_body
    return client.chat.completions.create(**kwargs)


def recorded(entry: dict[str, Any], message: Any) -> dict[str, Any]:
    """The transcript form of an assistant turn: the entry sent back to the model, plus its thinking.

    Providers return thinking beside the reply, as ``reasoning`` (OpenRouter) or ``reasoning_content``
    (vLLM-style servers). It is kept out of the history the model is sent, which stays unchanged.
    """
    thinking = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    return {**entry, "reasoning": str(thinking)} if thinking else entry


def close_episode(env: RetroEnvClient) -> tuple[dict[str, Any], Any]:
    """Score an episode the model never closed, and return (score, submission).

    An emit_routes call that failed in transport may still have been scored by
    the server. Submitting an empty route set then returns that earlier score
    with an "error" saying the episode is already complete, so the recorded
    reward is the one the routes actually earned; ``submission`` is None in
    that case, because the empty set was not what was scored.
    """
    outcome = env.call("emit_routes", {"submission": {"routes": []}})
    result = outcome.result if isinstance(outcome.result, dict) else {}
    score = dict(result.get("score") or {})
    already_complete = "error" in result
    if outcome.reward is not None:
        score["reward"] = float(outcome.reward)
    score.setdefault("reward", 0.0)
    if already_complete:
        score["recovered_after_transport_error"] = True
    return score, None if already_complete else {"routes": []}


def run_episode(llm: Any, env: RetroEnvClient, opening: dict[str, Any], config: AgentConfig) -> dict[str, Any]:
    tools = env.openai_tools()
    emit_tool = [tool for tool in tools if tool["function"]["name"] == "emit_routes"]
    allowed = {tool["function"]["name"] for tool in tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_turns=config.max_turns)},
        {"role": "user", "content": opening["prompt"]},
    ]
    transcript: list[dict[str, Any]] = []
    errors: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reported_cost_usd": 0.0}
    resolved_models: set[str] = set()
    submission: Any = None
    final: dict[str, Any] | None = None
    tool_calls = 0
    empty_turns = 0
    force_terminal = False
    coerced = False
    turns = 0
    started = time.perf_counter()

    for turn_index in range(config.max_turns):
        terminal_turn = force_terminal or turn_index == config.max_turns - 1
        # force_terminal already told the model this in the nudge below.
        if terminal_turn and not force_terminal and config.tool_choice != "required":
            final_turn = {"role": "user", "content": FINAL_TURN}
            messages.append(final_turn)
            transcript.append(final_turn)
        try:
            response = _request_with_retry(llm, config, messages, emit_tool if terminal_turn else tools)
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
        if not calls:
            empty_turns += 1
            errors.append(f"no tool call on turn {turn_index + 1}: {(message.content or '')[:200]}")
            assistant = {"role": "assistant", "content": message.content or ""}
            messages.append(assistant)
            transcript.append(recorded(assistant, message))
            if terminal_turn:
                break
            force_terminal = empty_turns >= config.max_empty_turns
            nudge = {
                "role": "user",
                "content": (
                    "The empty-turn limit was reached. On the next turn call emit_routes with the "
                    "best route trees available; no other tool will be exposed."
                    if force_terminal
                    else "No tool call was emitted. Continue by calling one of the available tools "
                    "now; on your final turn you must call emit_routes."
                ),
            }
            messages.append(nudge)
            transcript.append(nudge)
            continue

        assistant = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [call.model_dump(exclude_none=True) for call in calls],
        }
        messages.append(assistant)
        transcript.append(recorded(assistant, message))
        for call in calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as exc:
                result: Any = {"error": f"invalid tool arguments: {exc}"}
                errors.append(result["error"])
            else:
                arguments, was_coerced = normalize_arguments(name, arguments)
                coerced = coerced or was_coerced
                if name not in allowed:
                    result = {"error": f"unknown tool {name!r}"}
                else:
                    outcome = env.call(name, arguments)
                    tool_calls += 1
                    result = outcome.result if outcome.error is None else {"error": outcome.error}
                    if name == "emit_routes":
                        submission = arguments.get("submission")
                    if outcome.done and outcome.reward is not None:
                        score = result.get("score") if isinstance(result, dict) else None
                        final = {**(score or {}), "reward": float(outcome.reward)}
            content = result if isinstance(result, dict) else {"result": result}
            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps({**content, "model_turns_remaining": config.max_turns - turns}, sort_keys=True),
            }
            messages.append(tool_message)
            transcript.append(tool_message)
        if final is not None:
            break

    auto_emitted = False
    if final is None:
        final, fallback = close_episode(env)
        auto_emitted = True
        if fallback is not None:
            submission = fallback

    return {
        "reward": final.get("reward", 0.0),
        "valid": bool(final.get("valid", False)),
        "exact_match": bool((final.get("metrics") or {}).get("reference_match", False)),
        "components": final.get("components", {}),
        "hard_failures": final.get("hard_failures", []),
        "verification_tier": final.get("verification_tier"),
        "route_count": (final.get("metrics") or {}).get("route_count", 0),
        "submission": submission,
        "auto_emitted": auto_emitted,
        "submission_coerced": coerced,
        "tool_calls": tool_calls,
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
