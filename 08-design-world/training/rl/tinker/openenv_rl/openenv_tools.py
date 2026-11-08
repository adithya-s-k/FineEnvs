"""OpenEnv session tools as Tinker cookbook tools."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import ResourceSession
from tinker_cookbook.tool_use import ToolInput, ToolResult
from tinker_cookbook.tool_use.tools import error_tool_result, simple_tool_result

from .content import to_content_parts


@dataclass
class EpisodeLedger:
    """What one episode's tool calls reported, shared by its tools and its grader."""

    rewards: list[float] = field(default_factory=list)
    """Every reward a tool result carried, in call order."""
    done: bool = False
    """Whether a tool result ended the episode (e.g. a `submit` tool)."""
    calls: int = 0
    errors: int = 0
    unread: int = 0
    """Index into `rewards` of the first reward not yet paid out as a step reward."""

    @property
    def last_reward(self) -> float | None:
        return self.rewards[-1] if self.rewards else None


def _reward_of(metadata: dict[str, Any], data: Any) -> float | None:
    value = metadata.get("reward")
    if value is None and isinstance(data, dict):
        value = data.get("reward")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _types(schema: dict[str, Any]) -> set[str]:
    """The JSON types a (sub)schema allows, through anyOf / oneOf."""
    out: set[str] = set()
    t = schema.get("type")
    out.update([t] if isinstance(t, str) else t or [])
    for alt in schema.get("anyOf", []) + schema.get("oneOf", []):
        out |= _types(alt)
    return out


def coerce_to_schema(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Scalars the tool-call parser typed wrongly, recast to the type the tool's schema declares.

    Qwen's XML tool calls carry every value as text (`<parameter=text>120</parameter>`); the cookbook parser guesses
    a JSON type, so a string parameter that looks like a number arrives as an int. vLLM's `qwen3_coder` parser uses
    the schema instead, so without this the same output errors in training and works in evaluation.
    """
    props = schema.get("properties") or {}
    out = dict(arguments)
    for key, value in arguments.items():
        allowed = _types(props.get(key) or {})
        if not allowed or isinstance(value, bool):
            continue
        if "string" in allowed and isinstance(value, (int, float)) and not ({"integer", "number"} & allowed):
            out[key] = str(value)
        elif "array" in allowed and isinstance(value, (str, int, float)) and not ({"string", "integer", "number"} & allowed):
            item = (props[key].get("items") or {}) if isinstance(props.get(key), dict) else {}
            out[key] = [str(value) if "string" in _types(item) else value]
    return out


class SessionTool:
    """One tool of an OpenEnv `ResourceSession`, satisfying the cookbook's `Tool` protocol.

    Calls run in a worker thread because `ResourceSession` is synchronous. A result with
    `done=True` stops the episode; its reward (if any) is recorded in the episode ledger and never
    recomputed here: rewards are produced inside the environment.

    Args:
        session: The episode's session. One session belongs to exactly one episode.
        spec: The MCP tool spec as the session lists it.
        ledger: Shared per-episode record of rewards and termination.
    """

    def __init__(self, session: ResourceSession, spec: Tool, ledger: EpisodeLedger):
        self._session = session
        self._spec = spec
        self._ledger = ledger

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description or ""

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return dict(self._spec.input_schema or {"type": "object", "properties": {}})

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    async def run(self, input: ToolInput) -> ToolResult:
        call_id = input.call_id or ""
        self._ledger.calls += 1
        try:
            arguments = coerce_to_schema(dict(input.arguments), self.parameters_schema)
            result = await asyncio.to_thread(self._session.call_tool, self.name, arguments)
        except Exception as exc:  # noqa: BLE001 - the model sees the error and may recover
            self._ledger.errors += 1
            return error_tool_result(f"{type(exc).__name__}: {exc}", call_id=call_id, name=self.name)
        reward = _reward_of(result.metadata or {}, result.data)
        if reward is not None:
            self._ledger.rewards.append(reward)
        if result.done:
            self._ledger.done = True
        if result.error:
            self._ledger.errors += 1
            return error_tool_result(result.error, call_id=call_id, name=self.name, should_stop=result.done)
        return simple_tool_result(
            to_content_parts(result.data),
            call_id=call_id,
            name=self.name,
            should_stop=result.done,
            metrics={"tool_reward": reward} if reward is not None else None,
        )
