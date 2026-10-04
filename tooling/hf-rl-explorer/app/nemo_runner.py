"""One-step NeMo Pivot prediction, scored by NVIDIA's unmodified comparator.

Tool definitions are prediction targets; tool calls are never executed. No task
code, model key, or reference action enters a sandbox. Other Gym agents require
their resources server and are intentionally outside this adapter's scope.
"""
from __future__ import annotations

import json

import httpx
from pydantic import TypeAdapter

from .vendor.nemo_gym.verification_utils import (
    ActionComparator, ExpectedAction, FunctionCallAction, FunctionCallBatchAction,
    MessageAction, StepRewardCategory, ToolCallComparatorConfig,
)

REVISION = "3ef478df1ee163134d32a3f291f0a9e5981d0e52"
CONFIGS = {
    "resources_servers/single_step_tool_use_with_argument_comparison/configs/"
    + name + ".yaml" for name in (
        "single_step_tool_use_with_argument_comparison",
        "toolcall_schema_single_step_tool_use_with_argument_comparison",
    )
}


def supported(agent):
    return agent.get("known") and agent.get("agent_type") == "tool_simulation_agent" and agent.get("config") in CONFIGS


def strict_schema(schema):
    if not isinstance(schema, dict):
        return False
    if schema.get("type") == "object" or "properties" in schema:
        if schema.get("additionalProperties") is not False or set(schema.get("required") or []) != set(schema.get("properties") or {}):
            return False
    return all(strict_schema(v) for k, v in schema.items() if k in ("items", "additionalProperties") and isinstance(v, dict)) and all(
        strict_schema(v) for v in (schema.get("properties") or {}).values()) and all(
        strict_schema(v) for k in ("anyOf", "oneOf", "allOf") for v in schema.get(k, []))


def relaxed_tools(row):
    return [t["name"] for t in (row.get("responses_create_params") or {}).get("tools", [])
            if isinstance(t, dict) and t.get("name") and t.get("strict") is True and not strict_schema(t.get("parameters"))]


def responses_request(row):
    """Keep NeMo's native Responses input, including prior reasoning and tool turns.

    Only function definitions are allowed: provider-hosted MCP, web search and
    code execution would change a prediction task into a side-effecting agent.
    """
    rcp = row.get("responses_create_params")
    if not isinstance(rcp, dict) or not rcp.get("input"):
        raise ValueError("This row has no NeMo Gym Responses request")
    if len(json.dumps(rcp)) > 200_000:
        raise ValueError("This task's model input exceeds the hosted limit")
    raw = rcp["input"]
    if not isinstance(raw, (str, list)) or isinstance(raw, list) and len(raw) > 500:
        raise ValueError("This row needs the native NeMo Gym runtime")
    tools = rcp.get("tools") or []
    if not isinstance(tools, list) or len(tools) > 128 or not all(
        isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("name"), str) for t in tools
    ):
        raise ValueError("Only function prediction tools are supported here; use the native NeMo Gym runtime")
    allowed = {"input", "instructions", "tools", "tool_choice", "temperature", "top_p", "parallel_tool_calls", "reasoning", "text"}
    # Model, token limits, persistence and execution mode belong to the runner.
    body = {k: v for k, v in rcp.items() if k in allowed and v is not None}
    relaxed = set(relaxed_tools(row))
    if tools:
        body["tools"] = [{**t, "strict": False} if t["name"] in relaxed else t for t in tools]
    return body


def prediction(response):
    """Match NeMo's response_utils.extract_action: calls take precedence over text."""
    if response.get("error") or response.get("status") in ("failed", "cancelled"):
        error = response.get("error") or {}
        raise ValueError("Inference provider did not produce a prediction: " + str(error.get("message") or response.get("status"))[:600])
    if not isinstance(response.get("output"), list):
        raise ValueError("Inference provider returned an invalid Responses payload")
    if response.get("status") == "incomplete" and not response["output"]:
        raise ValueError("The model reached its output limit without a prediction. Increase Max output tokens.")
    calls, content = [], None
    for item in response.get("output") or []:
        if item.get("type") == "function_call":
            calls.append({"function": {"name": item["name"], "arguments": item["arguments"]}})
        elif item.get("type") == "message" and item.get("role") == "assistant" and content is None:
            content = next((p["text"] for p in item.get("content", []) if p.get("type") == "output_text"), None)
    return {"tool_calls": calls, "content": content}


def validate(row, agent):
    if not supported(agent):
        raise ValueError("This NeMo resources server is not hosted here")
    responses_request(row)
    if len(json.dumps(row.get("expected_action"))) > 100_000:
        raise ValueError("The reference action exceeds the hosted verifier limit")
    try:
        return TypeAdapter(ExpectedAction).validate_python(row.get("expected_action"))
    except ValueError:
        raise ValueError("The reference action is invalid for this NeMo Gym verifier") from None


def score(expected, message):
    calls = message.get("tool_calls") or []
    if len(calls) > 64:
        raise ValueError("The provider returned too many tool calls")
    actions = [FunctionCallAction(type="function_call", name=c["function"]["name"], arguments=c["function"]["arguments"]) for c in calls]
    actual = (actions[0] if len(actions) == 1 else FunctionCallBatchAction(type="function_call_batch", calls=actions)) if actions else (
        MessageAction(type="message", content=message["content"]) if isinstance(message.get("content"), str) else None)
    if actual is None:
        return 0.0, str(StepRewardCategory.NO_ACTION_FOUND)
    result = ActionComparator(config=ToolCallComparatorConfig(word_count_similarity_threshold=0.1)).compare_action(expected, actual)
    return result.reward, str(result.category)


def run(r):
    payload = r.native_task
    expected = validate(payload["row"], payload["agent"])
    body = responses_request(payload["row"])
    base, key, model = r.agent_api()
    params = r.run["params"]
    body.update(model=model, max_output_tokens=params.get("max_tokens", 2048), stream=False, store=False)
    if params.get("temperature") is not None:
        body["temperature"] = params["temperature"]
    r.check_cancel()
    r.phase("agent", detail="One prediction; tools are not executed")
    if relaxed_tools(payload["row"]):
        r.log("Some source tools declare strict=true with incompatible parameter schemas. Kept the original parameters and disabled strict enforcement for: "
              + ", ".join(relaxed_tools(payload["row"])))
    r.emit("prompt", harness="nemo-gym", text=json.dumps({k: v for k, v in body.items() if k != "model"}, ensure_ascii=False, indent=2))
    response = httpx.post(base + "/responses", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=90, follow_redirects=False)
    if response.status_code != 200:
        raise RuntimeError(f"Inference provider returned HTTP {response.status_code}; check model availability and Inference Providers permission")
    r.check_cancel()
    if len(response.content) > 2_000_000:
        raise ValueError("The model response exceeds the hosted limit")
    result = response.json()
    message = prediction(result)
    if message.get("content"):
        r.emit("text", text=message["content"])
    for call in (message.get("tool_calls") or [])[:64]:
        r.emit("tool", tool=call["function"]["name"], title="Predicted tool call", input=call["function"]["arguments"],
               output="Prediction only; no tool was executed.", status="done")
    usage = result.get("usage") or {}
    r.add_tokens({"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0)})
    r.emit("step", tokens=r.tokens, cost=r.model_cost)
    r.update(n_turns=1)
    r.phase("agent", "done")
    r.phase("verify", detail="NeMo Gym's pinned argument comparator")
    reward, category = score(expected, message)
    r.emit("checks", checks=[{"id": "native-comparator", "passed": reward == 1, "score": reward,
                              "method": "NeMo Gym", "message": category}], reward=reward, summary=category)
    r.phase("verify", "done", category)
    return {"reward": reward}
