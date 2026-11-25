"""A recorded airops-day run replayed: its tool calls played back through a Day in the order the env received them,
each output checked byte for byte against the recording, each watch's news delivered after its advance as the triggers
deliver it, and the rest of the day run as end-day runs it."""

import json
from dataclasses import dataclass

from airops_core.core.model import Task
from airops_core.day import Day
from airops_core.schedule import TOOLS
from pydantic import TypeAdapter, ValidationError

from .live import ChangesArg, _json, _raw

CHANGES = TypeAdapter(ChangesArg)
PLANNING = ("check_plan", "confirm_changes")


class ReplayError(Exception):
    """A recording the day doesn't reproduce: made before the env changed, or on another pack."""


@dataclass(frozen=True)
class Replayed:
    day: Day
    steps: list[dict]


def replay(task: Task, messages: list[dict]) -> Replayed:
    """``messages`` is airops-llm's record: each assistant turn's tool calls, and a tool message per call with what it
    returned. Calls to other tools (the gateway's get_time) and calls the agent never sent (arguments that aren't a
    JSON object) are skipped; a call the tools' argument validation refused must have been recorded as an error."""
    day, steps, calls, turn = Day(task), [], {}, 0
    for m in messages:
        if m["role"] == "assistant":
            turn += 1
            calls |= {c["id"]: (turn, c) for c in m.get("tool_calls") or []}
            continue
        if m["role"] != "tool" or day.done:
            continue
        at, call = calls[m["tool_call_id"]]
        args = _parsed(call["arguments"] or "{}")
        if call["name"] not in TOOLS or not isinstance(args, dict):
            continue
        day.calls += 1
        step = {"t": day.t, "watch": day.watch, "turn": at, "call_id": call["id"], "tool": call["name"],
                "args": {"changes": _parsed(args["changes"])} if "changes" in args else args}
        accepted, raw = _validated(call["name"], args)
        if not accepted:
            recorded = _parsed(m["content"])
            if not (isinstance(recorded, dict) and set(recorded) == {"error"}):
                raise ReplayError(f"{call['name']} {call['id']} has arguments its schema refuses, yet recorded "
                                  f"{m['content'][:200]}")
            steps.append(step | {"result": recorded})
            continue
        result = _json(play(day, call["name"], raw))
        if result != m["content"]:
            raise ReplayError(f"{call['name']} {call['id']} in watch {step['watch']} returned {result[:300]}, "
                              f"recorded {m['content'][:300]}")
        steps.append(step | {"result": json.loads(result)})
        if call["name"] == "advance" and not day.done:
            for n in day.watches[day.watch].news:
                day.notice(n, "trigger")
    if not day.done:
        day.end()
    return Replayed(day, steps)


def play(day: Day, name: str, raw) -> dict:
    """What the env's tool returns, before it is serialized."""
    if name == "get_situation":
        result = day.situation()
    elif name == "advance":
        result = day.finish("done") if day.last else day.open()
    elif name == "check_plan":
        result = day.check(raw)
    else:
        result = day.confirm(raw)
    return {**result, "messages": day.drain()}


def _validated(name: str, args: dict) -> tuple[bool, object]:
    """Whether the tool's argument validation passes the call, and the changes it hands the day."""
    if name not in PLANNING:
        return not args, None
    if set(args) != {"changes"}:
        return False, None
    try:
        return True, _raw(CHANGES.validate_python(args["changes"]))
    except ValidationError:
        return False, None


def _parsed(value):
    """A JSON string as what it encodes; anything else as it is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value
