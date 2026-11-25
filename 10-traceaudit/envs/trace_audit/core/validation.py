"""Pinned Harbor JSON-schema validation plus ATIF's cross-field invariants, stdlib only.

The schema is generated from the pinned upstream Harbor models. The official validator is also run
by tools/validate_harbor.py; this module does not import the Harbor framework at runtime.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path


class DataError(ValueError):
    pass


SCHEMA = json.loads(Path(__file__).with_name("atif-v1.7.schema.json").read_text(encoding="utf-8"))


def _schema_check(value, schema, path="trajectory"):
    if "$ref" in schema:
        target = SCHEMA
        for key in schema["$ref"].removeprefix("#/").split("/"):
            target = target[key]
        _schema_check(value, target, path)
    _check_alternatives(value, schema, path)
    if "enum" in schema and value not in schema["enum"]:
        raise DataError(f"{path}: invalid enum value")
    if "const" in schema and value != schema["const"]:
        raise DataError(f"{path}: invalid constant")
    kind = schema.get("type")
    valid = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: type(v) is int,
        "number": lambda v: type(v) in (int, float) and math.isfinite(v),
        "boolean": lambda v: type(v) is bool,
        "null": lambda v: v is None,
    }
    if kind and (kind not in valid or not valid[kind](value)):
        raise DataError(f"{path}: expected {kind}")
    _check_object(value, schema, path)
    _check_array(value, schema, path)
    if type(value) in (int, float):
        if (
            not math.isfinite(value)
            or value < schema.get("minimum", -float("inf"))
            or value > schema.get("maximum", float("inf"))
        ):
            raise DataError(f"{path}: number out of range")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        raise DataError(f"{path}: string too short")


def validate_atif(trajectory: dict) -> None:
    _schema_check(trajectory, SCHEMA)
    if trajectory["schema_version"] != "ATIF-v1.7":
        raise DataError("Only ATIF-v1.7 is accepted; convert explicitly")
    if trajectory.get("continued_trajectory_ref") is not None:
        raise DataError(
            "Join continued trajectories before admission; v1 requires the complete trace"
        )
    embedded = trajectory.get("subagent_trajectories") or []
    ids = [child.get("trajectory_id") for child in embedded]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise DataError("Embedded subagents need distinct trajectory_id values")
    for child in embedded:
        validate_atif(child)
    for index, step in enumerate(trajectory["steps"], 1):
        _validate_step(step, index, ids)


def _check_object(value, schema, path):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise DataError(f"{path}: keys must be strings")
        for key in schema.get("required", []):
            if key not in value:
                raise DataError(f"{path}.{key}: required")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, child in value.items():
            if key in properties:
                _schema_check(child, properties[key], f"{path}.{key}")
            elif extra is False:
                raise DataError(f"{path}.{key}: unknown field")
            elif isinstance(extra, dict):
                _schema_check(child, extra, f"{path}.{key}")


def _check_array(value, schema, path):
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get(
            "maxItems", float("inf")
        ):
            raise DataError(f"{path}: invalid array length")
        for index, child in enumerate(value):
            _schema_check(child, schema.get("items", {}), f"{path}[{index}]")


def _check_alternatives(value, schema, path):
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _schema_check(value, option, path)
                break
            except DataError:
                pass
        else:
            raise DataError(f"{path}: no permitted schema alternative matches")


def _validate_step(step, index, ids):
    if step.get("timestamp"):
        try:
            datetime.fromisoformat(step["timestamp"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise DataError("Invalid timestamp") from exc
    if step["step_id"] != index:
        raise DataError("ATIF step IDs must be sequential from 1")
    if step["source"] != "agent" and any(
        step.get(key) is not None
        for key in ("model_name", "reasoning_effort", "reasoning_content", "tool_calls", "metrics")
    ):
        raise DataError("Agent-only fields appear on a non-agent step")
    if (
        step["source"] == "agent"
        and step.get("llm_call_count") == 0
        and any(step.get(k) is not None for k in ("metrics", "reasoning_content"))
    ):
        raise DataError("Deterministic dispatch cannot carry LLM metrics/reasoning")
    calls = [call["tool_call_id"] for call in step.get("tool_calls") or []]
    participant_calls = (step.get("extra") or {}).get("participant_tool_calls")
    participant_ids = []
    if participant_calls is not None:
        if step["source"] != "user" or not isinstance(participant_calls, list):
            raise DataError("Participant calls require an array on a user step")
        for call in participant_calls:
            _schema_check(call, {"$ref": "#/$defs/ToolCall"})
            participant_ids.append(call["tool_call_id"])
        if len(set(participant_ids)) != len(participant_ids):
            raise DataError("Duplicate participant tool call ID")
    if len(set(calls)) != len(calls):
        raise DataError("Duplicate tool call IDs within a step")
    _validate_observations(step, calls, participant_ids, ids)


def _validate_observations(step, calls, participant_ids, ids):
    for result in (step.get("observation") or {}).get("results", []):
        participant_id = (result.get("extra") or {}).get("participant_call_id")
        if participant_id is not None and (
            participant_id not in participant_ids
            or (result.get("extra") or {}).get("requestor") != "user"
        ):
            raise DataError("Participant observation lacks its user call")
        if result.get("source_call_id") is not None and result["source_call_id"] not in calls:
            raise DataError("Observation references a tool call outside its step")
        _validate_result_status(result)
        for ref in result.get("subagent_trajectory_ref") or []:
            if ref.get("trajectory_path"):
                raise DataError("External subagent files must be embedded before admission")
            if ref.get("trajectory_id") not in ids:
                raise DataError("Unresolved embedded subagent reference")


def _validate_result_status(result):
    error = (result.get("extra") or {}).get("error")
    if error is not None and (
        not isinstance(error, dict)
        or type(error.get("is_error")) is not bool
        or not isinstance(error.get("message"), str)
    ):
        raise DataError("Tool error must use extra.error = {is_error: bool, message: str}")
    status = (result.get("extra") or {}).get("command_status")
    if status is not None and (
        not isinstance(status, dict)
        or type(status.get("exit_code")) is not int
        or type(status.get("nonzero_exit")) is not bool
        or status["nonzero_exit"] != (status["exit_code"] != 0)
    ):
        raise DataError(
            "Command status requires an integer exit_code and consistent boolean nonzero_exit"
        )
