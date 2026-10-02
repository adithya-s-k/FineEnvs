"""Serve the same SETA environment used by GRPO, one sandbox per WebSocket session."""

import inspect
import math
import os

from fastmcp import FastMCP
from openenv.core.env_server import create_app
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation, State

from .catalog import TaskCatalog, task_by_name
from .environment import BashEnvironment
from .ui import build_ui


class WhiteboxEnvironment(TaskCatalog, MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State()
        self.episode = BashEnvironment()
        mcp = FastMCP("smoldataenv-whitebox")
        for name, method in inspect.getmembers(self.episode, inspect.ismethod):
            if not name.startswith("_") and name not in {"reset", "get_reward"}:
                mcp.tool(method)

        @mcp.tool
        def start_task(split: str = "test", index: int = 0) -> dict:
            """Select a task and stage its files in a new sandbox."""
            task = self.get_task(split, index)
            observation = self.reset(split=split, task_name=task["name"])
            return observation.metadata

        @mcp.tool
        def grade() -> dict:
            """Grade the final answer and release the sandbox."""
            value = self.episode.get_reward()
            return {
                "reward": value if math.isfinite(value) else None,
                "correctness": getattr(self.episode, "_correctness", None),
                "tool_calls": self.episode._calls,
            }

        super().__init__(mcp)

    def reset(
        self, seed=None, episode_id=None, split="train", task_name=None, **kwargs
    ):
        if task_name is None:
            return Observation(metadata={"status": "Choose a task from the Task API"})
        task = task_by_name(split, task_name)
        instruction = self.episode.reset(folder=task["folder"])
        return Observation(
            metadata={"instruction": instruction, "task_name": task_name}
        )

    @property
    def state(self):
        return self._state

    def _step_impl(self, action, **kwargs):
        raise ValueError("Use an MCP tool action")

    def step(self, action, timeout_s=None, **kwargs):
        return super().step(action, timeout_s=timeout_s or 180, **kwargs)

    async def step_async(self, action, timeout_s=None, **kwargs):
        return await super().step_async(action, timeout_s=timeout_s or 180, **kwargs)

    def close(self):
        try:
            self.episode._close()
        finally:
            super().close()


os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
app = create_app(
    WhiteboxEnvironment,
    CallToolAction,
    CallToolObservation,
    gradio_builder=build_ui,
    show_default_tab=False,
    env_name="smoldataenv_whitebox",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "40")),
)
