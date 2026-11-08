"""A `ResourceSession` for any MCP environment, for envs that ship no `harness.py` of their own."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation, Tool
from openenv.core.harness import (
    Message,
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    VerifyResult,
)
from openenv.core.mcp_client import MCPToolClient

InitialMessagesBuilder = Callable[[StepResult, Any], list[Message]]
ResetKwargsBuilder = Callable[[Any], dict[str, Any]]


def default_initial_messages(result: StepResult, task: Any) -> list[Message]:
    """The reset observation's `instructions` (or `prompt`) metadata as the opening user message."""
    metadata = getattr(result.observation, "metadata", None) or {}
    text = metadata.get("instructions") or metadata.get("prompt")
    if not text and isinstance(task, dict):
        text = task.get("prompt") or task.get("instruction")
    if not text:
        raise ValueError(
            "no initial prompt: the reset observation has no 'instructions'/'prompt' metadata; "
            "pass initial_messages_builder="
        )
    return [{"role": "user", "content": str(text)}]


class MCPSession(ResourceSession):
    """One episode of an MCP environment over the step (WebSocket) transport.

    The step transport is used rather than the production `/mcp` endpoint because only it carries
    the environment's reward and `done` back with each tool call. Tools are listed after `reset`,
    since an environment's tool set may depend on its reset arguments.

    Args:
        client: An `MCPToolClient` (or its `.sync()` wrapper) connected to the environment.
        task: The task this episode plays; handed to the builders.
        reset_kwargs: Keyword arguments for `reset()`.
        initial_messages_builder: Maps `(reset StepResult, task)` to the opening messages.
        exclude_tools: Tool names the agent must not see (e.g. debugging or admin tools).
    """

    def __init__(
        self,
        client: Any,
        *,
        task: Any = None,
        reset_kwargs: dict[str, Any] | None = None,
        initial_messages_builder: InitialMessagesBuilder = default_initial_messages,
        exclude_tools: Sequence[str] = (),
    ):
        self._client = client.sync() if hasattr(client, "sync") and callable(client.sync) else client
        self._task = task
        self._builder = initial_messages_builder
        self._exclude = set(exclude_tools)
        self._last: StepResult | None = None
        self._closed = False
        try:
            self._reset = self._client.reset(**(reset_kwargs or {}))
            self._tools = [t for t in self._client.list_tools(use_cache=False) if t.name not in self._exclude]
        except Exception:
            self.close()
            raise
        self._names = {t.name for t in self._tools}

    @property
    def reset_result(self) -> StepResult:
        return self._reset

    def initial_messages(self) -> list[Message]:
        return list(self._builder(self._reset, self._task))

    def list_tools(self) -> list[Tool]:
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name not in self._names:
            return ToolResult(error=f"unknown tool: {name}")
        result = self._client.step(CallToolAction(tool_name=name, arguments=dict(arguments)))
        self._last = result
        obs = result.observation
        error = None
        data = None
        if isinstance(obs, CallToolObservation):
            data = obs.result
            if obs.error is not None:
                error = getattr(obs.error, "message", None) or str(obs.error)
        reward = result.reward if result.reward is not None else getattr(obs, "reward", None)
        return ToolResult(
            data=data,
            done=bool(result.done),
            metadata={"reward": reward, **(getattr(obs, "metadata", None) or {})},
            error=error,
        )

    def verify(self, transcript: list[Message], final_state: Any | None = None) -> VerifyResult:
        """The last reward the environment reported (never recomputed, `None` if it gave none)."""
        last = self._last
        return VerifyResult(
            env_reward=None if last is None else last.reward,
            done=bool(last.done) if last is not None else False,
            metrics={"tool_steps": float(len(transcript))},
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


class MCPSessionFactory(ResourceSessionFactory[MCPSession]):
    """Creates an `MCPSession` per episode, each with its own connection (sessions never share one).

    Picklable: it stores only the URL and the builder callables, so it can travel to rollout
    workers. Point several factories at different servers (or a load balancer) to scale out.

    Args:
        base_url: The environment server (local, Docker, HF Space, Modal, ...).
        reset_kwargs_builder: Maps a task to `reset()` keyword arguments. Default: the task itself
            if it is a dict, else `{}`.
        initial_messages_builder: See `MCPSession`.
        exclude_tools: Tool names hidden from the agent.
        message_timeout_s: Per-call timeout; long tool calls (rendering, browsers) need more.
        max_message_size_mb: Screenshots make large messages.
    """

    def __init__(
        self,
        base_url: str,
        *,
        reset_kwargs_builder: ResetKwargsBuilder | None = None,
        initial_messages_builder: InitialMessagesBuilder = default_initial_messages,
        exclude_tools: Sequence[str] = (),
        message_timeout_s: float = 300.0,
        max_message_size_mb: float = 100.0,
    ):
        self.base_url = base_url
        self.reset_kwargs_builder = reset_kwargs_builder
        self.initial_messages_builder = initial_messages_builder
        self.exclude_tools = tuple(exclude_tools)
        self.message_timeout_s = message_timeout_s
        self.max_message_size_mb = max_message_size_mb

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> MCPSession:
        kwargs = (
            self.reset_kwargs_builder(task)
            if self.reset_kwargs_builder
            else (dict(task) if isinstance(task, dict) else {})
        )
        if seed is not None:
            kwargs.setdefault("seed", seed)
        if episode_id is not None:
            kwargs.setdefault("episode_id", episode_id)
        client = MCPToolClient(
            base_url=self.base_url,
            message_timeout_s=self.message_timeout_s,
            max_message_size_mb=self.max_message_size_mb,
        )
        return MCPSession(
            client,
            task=task,
            reset_kwargs=kwargs,
            initial_messages_builder=self.initial_messages_builder,
            exclude_tools=self.exclude_tools,
        )
