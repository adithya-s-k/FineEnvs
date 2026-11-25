from __future__ import annotations

import copy
import hashlib
import json

from trace_audit.core import DataError, validate_record


def stable_id(*values) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:24]


def chat_to_atif(
    messages: list[dict],
    *,
    tools: list | None = None,
    agent_name: str = "source-agent",
    allow_user_tools: bool = False,
) -> dict:
    """Join tool results to ATIF steps and preserve private source-index mappings."""
    if not isinstance(messages, list) or not messages:
        raise DataError("A nonempty chat trajectory is required")
    steps, pending, mapping = [], {}, {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise DataError("Chat message must be an object")
        role = message.get("role")
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise DataError("Text-only v1 cannot flatten structured/multimodal content")
        if role in {"tool", "function"}:
            step = _attach_result(message, content, pending)
            mapping[str(index)] = step["step_id"]
            continue
        if role not in {"system", "user", "assistant", "agent"}:
            raise DataError(f"Unsupported chat role: {role}")
        step = {
            "step_id": len(steps) + 1,
            "source": "agent" if role in {"assistant", "agent"} else role,
            "message": content,
            "extra": {"original_message_index": index},
        }
        _attach_calls(message, step, pending, allow_user_tools)
        if message.get("reasoning_content") and step["source"] == "agent":
            step["reasoning_content"] = message["reasoning_content"]
        steps.append(step)
        mapping[str(index)] = step["step_id"]
    return {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": agent_name, "version": "source-export", "tool_definitions": tools or []},
        "steps": steps,
        "extra": {
            "adapter": "chat-v1",
            "original_message_index_to_step_id": mapping,
            "unobserved_tool_call_ids": [
                {"requestor": who, "id": key} for who, key in sorted(pending)
            ],
        },
    }


def make_record(
    *,
    source: dict,
    task: dict,
    trajectory: dict,
    family: str,
    split: str,
    failed: bool | None = None,
    checker=None,
    reviewer: str | None = None,
    label_source: str = "none",
    failures: list | None = None,
) -> dict:
    source = copy.deepcopy(source)
    row = {
        "record_id": stable_id(source["dataset"], source["revision"], source["original_id"]),
        "source": source,
        "task": copy.deepcopy(task),
        "trajectory": copy.deepcopy(trajectory),
        "outcome": {
            "failed": failed,
            "label_source": label_source if failed is not None else "none",
            "checker": checker if failed is not None else None,
            "reviewer": reviewer,
        },
        "failures": failures or [],
        "group": {"task_family_id": family},
        "split": split,
    }
    validate_record(row)
    return row


def _attach_result(message, content, pending):
    call_id = message.get("tool_call_id")
    requestor = message.get("requestor", "assistant")
    pending_key = (requestor, call_id)
    if pending_key not in pending:
        raise DataError("Orphan tool response; tool calls must be explicitly linked")
    step, normalized_id = pending.pop(pending_key)
    result = {"content": content}
    if requestor == "user":
        result["extra"] = {"participant_call_id": normalized_id, "requestor": "user"}
    else:
        result["source_call_id"] = normalized_id
    if message.get("is_error") is True or message.get("error") is True:
        detail = message.get("error_message")
        result.setdefault("extra", {})["error"] = {
            "is_error": True,
            "message": detail if isinstance(detail, str) else content,
        }
    if "command_exit_code" in message:
        code = message["command_exit_code"]
        if type(code) is not int:
            raise DataError("Command exit status must be an integer")
        result.setdefault("extra", {})["command_status"] = {
            "exit_code": code,
            "nonzero_exit": code != 0,
        }
    step.setdefault("observation", {"results": []})["results"].append(result)
    return step


def _attach_calls(message, step, pending, allow_user_tools):
    for call_index, raw_call in enumerate(message.get("tool_calls") or []):
        user_call = step["source"] == "user" and allow_user_tools
        if step["source"] != "agent" and not user_call:
            raise DataError("Tool call on a non-agent turn")
        function = raw_call.get("function", raw_call)
        original_id = raw_call.get("id")
        pending_key = ("user" if user_call else "assistant", original_id)
        if not isinstance(original_id, str) or pending_key in pending:
            raise DataError("Tool call needs a unique original ID")
        args = _call_arguments(function)
        name = function.get("name")
        if not isinstance(name, str):
            raise DataError("Missing tool function name")
        normalized_id = f"c{step['step_id']}_{call_index + 1}"
        call = {
            "tool_call_id": normalized_id,
            "function_name": name,
            "arguments": args,
            "extra": {"original_tool_call_id": original_id},
        }
        if user_call:
            step["extra"].setdefault("participant_tool_calls", []).append(call)
        else:
            step.setdefault("tool_calls", []).append(call)
        pending[pending_key] = (step, normalized_id)


def _call_arguments(function):
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            # Preserve malformed calls as evidence, rather than dropping the failure.
            args = {"raw_arguments": args}
    if not isinstance(args, dict):
        args = {"raw_arguments": args}
    return args
