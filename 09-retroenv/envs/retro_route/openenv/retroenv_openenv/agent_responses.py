"""OpenAI Responses API agent loop against a RetroEnv server.

GPT-5.6 models reject function tools with reasoning on /v1/chat/completions,
so they run here. Same protocol and result shape as ``agent.run_episode``. The
history is sent back each turn (``store=False``) with encrypted reasoning items
included, so no conversation state is kept at OpenAI.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from .agent import SYSTEM_PROMPT, close_episode, normalize_arguments
from .client import RetroEnvClient


@dataclass
class ResponsesConfig:
    model: str
    max_turns: int = 16
    max_output_tokens: int = 16000
    reasoning_effort: str | None = None  # None keeps the model default
    max_empty_turns: int = 2


def run_episode(llm: Any, env: RetroEnvClient, opening: dict[str, Any], config: ResponsesConfig) -> dict[str, Any]:
    tools = [
        {
            "type": "function",
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "parameters": tool["function"]["parameters"],
            "strict": False,
        }
        for tool in env.openai_tools()
    ]
    emit_tool = [tool for tool in tools if tool["name"] == "emit_routes"]
    allowed = {tool["name"] for tool in tools}
    items: list[dict[str, Any]] = [
        {"role": "developer", "content": SYSTEM_PROMPT.format(max_turns=config.max_turns)},
        {"role": "user", "content": opening["prompt"]},
    ]
    transcript: list[dict[str, Any]] = [{"role": "user", "content": opening["prompt"]}]
    errors: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
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
        request: dict[str, Any] = {
            "model": config.model,
            "input": items,
            "tools": emit_tool if terminal_turn else tools,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_output_tokens": config.max_output_tokens,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if config.reasoning_effort:
            request["reasoning"] = {"effort": config.reasoning_effort}
        try:
            response = llm.responses.create(**request)
        except Exception as exc:  # provider errors end the episode; it is still scored
            errors.append(f"API error: {type(exc).__name__}: {str(exc)[:500]}")
            break
        turns += 1
        if response.usage:
            usage["prompt_tokens"] += int(response.usage.input_tokens or 0)
            usage["completion_tokens"] += int(response.usage.output_tokens or 0)
            details = response.usage.input_tokens_details
            usage["cached_tokens"] += int(getattr(details, "cached_tokens", 0) or 0)
            out_details = response.usage.output_tokens_details
            usage["reasoning_tokens"] += int(getattr(out_details, "reasoning_tokens", 0) or 0)
        output = [item.model_dump(exclude_none=True) for item in response.output]
        items.extend(output)
        transcript.append(
            {
                "role": "assistant",
                "content": [{k: v for k, v in item.items() if k != "encrypted_content"} for item in output],
            }
        )

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            empty_turns += 1
            text = (response.output_text or "")[:200]
            errors.append(f"no tool call on turn {turn_index + 1}: {text}")
            if terminal_turn:
                break
            force_terminal = empty_turns >= config.max_empty_turns
            nudge = (
                "No tool call was made. Call one of the available tools now; on your final "
                "turn you must call emit_routes."
            )
            items.append({"role": "user", "content": nudge})
            transcript.append({"role": "user", "content": nudge})
            continue

        for call in calls:
            try:
                arguments = json.loads(call.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as exc:
                result: Any = {"error": f"invalid tool arguments: {exc}"}
                errors.append(result["error"])
            else:
                arguments, was_coerced = normalize_arguments(call.name, arguments)
                coerced = coerced or was_coerced
                if call.name not in allowed:
                    result = {"error": f"unknown tool {call.name!r}"}
                else:
                    outcome = env.call(call.name, arguments)
                    tool_calls += 1
                    result = outcome.result if outcome.error is None else {"error": outcome.error}
                    if call.name == "emit_routes":
                        submission = arguments.get("submission")
                    if outcome.done and outcome.reward is not None:
                        score = result.get("score") if isinstance(result, dict) else None
                        final = {**(score or {}), "reward": float(outcome.reward)}
            payload = result if isinstance(result, dict) else {"result": result}
            text = json.dumps({**payload, "model_turns_remaining": config.max_turns - turns}, sort_keys=True)
            items.append({"type": "function_call_output", "call_id": call.call_id, "output": text})
            transcript.append({"role": "tool", "name": call.name, "content": text})
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
            "reported_cost_usd": None,
            "latency_seconds": round(time.perf_counter() - started, 3),
        },
        "resolved_models": [config.model],
        "transcript": transcript,
    }
