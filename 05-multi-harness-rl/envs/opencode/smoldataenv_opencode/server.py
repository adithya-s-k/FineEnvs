"""Run native OpenCode on SmolDataEnvs through the OpenEnv protocol."""

import os
from types import SimpleNamespace

from fastmcp import FastMCP
from openenv.core.env_server import create_app
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation, State

from .catalog import TaskCatalog, task_by_name
from .environment import TaskFactory
from .ui import build_ui


class OpenCodeEnvironment(TaskCatalog, MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = State()
        mcp = FastMCP("smoldataenv-opencode")

        @mcp.tool
        def run_rollout(split: str, task_name: str, model: str, sampling: dict) -> dict:
            """Run OpenCode, grade its answer, and export its typed training trace."""
            row = task_by_name(split, task_name)
            factory = TaskFactory(
                SimpleNamespace(model=model), [row], sampling=sampling
            )
            session = factory.create([{"role": "user", "content": row["instruction"]}])
            try:
                session.wait_for_completion(timeout_s=900)
                try:
                    trace = session.fetch_training_trace()
                except (ValueError, TypeError, KeyError) as exc:
                    return {"capture_error": str(exc)}
                grade = session.verify([]).env_reward
                return {
                    "training_trace": trace.model_dump(mode="json"),
                    "correctness": grade,
                }
            finally:
                session.close()

        super().__init__(mcp)

    def reset(self, seed=None, episode_id=None, **kwargs):
        self._state = State(episode_id=episode_id)
        return Observation(metadata={"status": "Call run_rollout with a task name"})

    @property
    def state(self):
        return self._state

    def _step_impl(self, action, **kwargs):
        raise ValueError("Use an MCP tool action")

    def step(self, action, timeout_s=None, **kwargs):
        return super().step(action, timeout_s=timeout_s or 1800, **kwargs)

    async def step_async(self, action, timeout_s=None, **kwargs):
        return await super().step_async(action, timeout_s=timeout_s or 1800, **kwargs)


os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
app = create_app(
    OpenCodeEnvironment,
    CallToolAction,
    CallToolObservation,
    gradio_builder=build_ui,
    show_default_tab=False,
    env_name="smoldataenv_opencode",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "40")),
)
