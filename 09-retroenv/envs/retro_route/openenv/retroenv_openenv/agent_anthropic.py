"""Claude agent loop against a RetroEnv server, on the Anthropic Messages API.

Same protocol as ``agent.run_episode`` (turn budget, a final emit_routes turn,
empty-turn recovery, turns-left in every tool result) and the same result
shape. Two Claude-specific choices:

* Current models reject a forced ``tool_choice``, so the final turn exposes
  only emit_routes with ``tool_choice=auto`` and says so in the user turn.
* The history is append-only (assistant content is sent back unchanged, so
  thinking blocks stay valid), and top-level ``cache_control`` caches it.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import anthropic

from .agent import FINAL_TURN, SYSTEM_PROMPT, close_episode, normalize_arguments
from .client import RetroEnvClient

# USD per million tokens: input, output, cache read. Cache writes bill 1.25x input.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-sonnet-4-6": (3.00, 15.00, 0.30),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
    "claude-fable-5-1": (10.00, 50.00, 0.25),
    "claude-fable-5": (10.00, 50.00, 1.00),
}


@dataclass
class ClaudeConfig:
    model: str = "claude-opus-5-5"
    max_turns: int = 16
    max_tokens: int = 16000
    # None keeps the model default (medium on Claude Opus 5.5, high on Sonnet 5.5).
    effort: str | None = None
    max_empty_turns: int = 2
    parallel_tool_calls: bool = False


def _cost(model: str, usage: dict[str, int]) -> float | None:
    price = PRICES.get(model)
    if price is None:
        # Better a visible gap in the board than a confident wrong total.
        print(f"warning: no price for {model!r}; cost will be reported as null", file=sys.stderr)
        return None
    input_price, output_price, cache_read_price = price
    return (
        usage["input_tokens"] * input_price
        + usage["cache_creation_input_tokens"] * input_price * 1.25
        + usage["cache_read_input_tokens"] * cache_read_price
        + usage["output_tokens"] * output_price
    ) / 1_000_000


def _block(block: Any) -> dict[str, Any]:
    value = block.model_dump(exclude_none=True)
    value.pop("signature", None)  # opaque; not useful in a stored transcript
    return value


def run_episode(
    llm: anthropic.Anthropic, env: RetroEnvClient, opening: dict[str, Any], config: ClaudeConfig
) -> dict[str, Any]:
    tools = [
        {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "input_schema": tool["function"]["parameters"],
        }
        for tool in env.openai_tools()
    ]
    emit_tool = [tool for tool in tools if tool["name"] == "emit_routes"]
    allowed = {tool["name"] for tool in tools}
    system = SYSTEM_PROMPT.format(max_turns=config.max_turns)
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": opening["prompt"]}]}]
    transcript: list[dict[str, Any]] = [{"role": "user", "content": opening["prompt"]}]
    errors: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
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
        if terminal_turn:
            messages[-1]["content"].append({"type": "text", "text": FINAL_TURN})
        request: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "system": system,
            "messages": messages,
            "tools": emit_tool if terminal_turn else tools,
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": not config.parallel_tool_calls},
            "cache_control": {"type": "ephemeral"},
        }
        if config.effort:
            request["output_config"] = {"effort": config.effort}
        try:
            response = llm.messages.create(**request)
        except anthropic.APIStatusError as exc:
            errors.append(f"API error {exc.status_code}: {str(exc.message)[:500]}")
            break
        except anthropic.APIConnectionError as exc:
            errors.append(f"API connection error: {exc}")
            break
        turns += 1
        for key in usage:
            usage[key] += int(getattr(response.usage, key, 0) or 0)
        messages.append({"role": "assistant", "content": response.content})
        transcript.append(
            {"role": "assistant", "content": [_block(b) for b in response.content], "stop_reason": response.stop_reason}
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            errors.append(f"refusal: {getattr(details, 'category', None)}")
            break

        calls = [block for block in response.content if block.type == "tool_use"]
        if not calls:
            empty_turns += 1
            text = " ".join(b.text for b in response.content if b.type == "text")
            errors.append(f"no tool call on turn {turn_index + 1} ({response.stop_reason}): {text[:200]}")
            if terminal_turn:
                break
            force_terminal = empty_turns >= config.max_empty_turns
            nudge = (
                "No tool call was made. Call one of the available tools now; on your final "
                "turn you must call emit_routes."
            )
            messages.append({"role": "user", "content": [{"type": "text", "text": nudge}]})
            transcript.append({"role": "user", "content": nudge})
            continue

        results = []
        for call in calls:
            arguments, was_coerced = normalize_arguments(call.name, call.input if isinstance(call.input, dict) else {})
            coerced = coerced or was_coerced
            if call.name not in allowed:
                content: Any = {"error": f"unknown tool {call.name!r}"}
                is_error = True
            else:
                outcome = env.call(call.name, arguments)
                tool_calls += 1
                is_error = outcome.error is not None
                content = outcome.result if not is_error else {"error": outcome.error}
                if call.name == "emit_routes":
                    submission = arguments.get("submission")
                if outcome.done and outcome.reward is not None:
                    score = content.get("score") if isinstance(content, dict) else None
                    final = {**(score or {}), "reward": float(outcome.reward)}
            payload = content if isinstance(content, dict) else {"result": content}
            text = json.dumps({**payload, "model_turns_remaining": config.max_turns - turns}, sort_keys=True)
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": text, "is_error": is_error})
        messages.append({"role": "user", "content": results})
        transcript.append({"role": "user", "content": results})
        if final is not None:
            break

    auto_emitted = False
    if final is None:
        final, fallback = close_episode(env)
        auto_emitted = True
        if fallback is not None:
            submission = fallback

    cost = _cost(config.model, usage)
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
            "prompt_tokens": usage["input_tokens"]
            + usage["cache_read_input_tokens"]
            + usage["cache_creation_input_tokens"],
            "completion_tokens": usage["output_tokens"],
            **usage,
            "reported_cost_usd": round(cost, 8) if cost is not None else None,
            "latency_seconds": round(time.perf_counter() - started, 3),
        },
        "resolved_models": [config.model],
        "transcript": transcript,
    }
