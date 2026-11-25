"""Normalize OpenHands chat rows using the test-derived `resolved` outcome."""

from __future__ import annotations

import copy
import json
import re

from trace_audit.core.provenance import resolve_model
from trace_audit.core.records import training_split

from .common import DataError, chat_to_atif, make_record


def convert(
    row: dict, *, revision: str, license: str = "CC-BY-4.0", split: str | None = None
) -> dict:
    messages = row.get("trajectory")
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list):
        raise DataError("SWE-rebench trajectory must be a chat array")
    messages = copy.deepcopy(messages)
    _command_statuses(messages)
    instruction = next((m.get("content") for m in messages if m.get("role") == "user"), None)
    if not isinstance(instruction, str):
        raise DataError("Missing task instruction in the captured user messages")
    instance = row.get("instance_id")
    original_id = row.get("trajectory_id")
    if not isinstance(instance, str) or not isinstance(original_id, str):
        raise DataError("SWE-rebench instance_id and trajectory_id are required")
    resolved = row.get("resolved")
    # Missing/unknown outcomes are not converted into failures. Do not use generated-test scores.
    failed = not bool(resolved) if type(resolved) in (int, bool) and resolved in (0, 1) else None
    family = "swe:" + instance
    if split not in (None, "train", "dev"):
        raise DataError("SWE-rebench is train/dev only in the default held-out-source protocol")
    return make_record(
        source={
            "dataset": "nebius/SWE-rebench-openhands-trajectories",
            "revision": revision,
            "license": license,
            "original_id": original_id,
            "domain": "coding",
            "source_split": "train",
            "agent_model": resolve_model(row.get("agent_model"), row.get("source_model")),
            "user_model": "not_applicable",
            "identity_strings": ["OpenHands", str(row.get("source_model", ""))],
        },
        task={"instruction": instruction, "policy": None, "success_criteria": None},
        trajectory=chat_to_atif(messages, tools=row.get("tools"), agent_name="OpenHands"),
        family=family,
        split=split or training_split(family),
        failed=failed,
        label_source="verified_outcome",
        checker={
            "kind": "published_test_outcome",
            "field": "resolved",
            "revision": revision,
            "limitation": (
                "Test-suite outcome; independent patch correctness review is outstanding."
            ),
        },
    )


def _command_statuses(messages):
    # OpenHands' terminal command-status footer is structured harness output, not an
    # English error heuristic. Only accept the complete footer on a linked shell call.
    calls = {}
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                calls[call.get("id")] = call.get("function", call).get("name")
        elif (
            message.get("role") == "tool"
            and calls.get(message.get("tool_call_id")) == "execute_bash"
        ):
            match = re.search(
                r"(?:^|\n)\[Command finished with exit code (-?\d+)\]\s*\Z",
                message.get("content") or "",
            )
            if match:
                message["command_exit_code"] = int(match[1])
