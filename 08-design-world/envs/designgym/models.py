"""OpenEnv wire models. Actions and observations are OpenEnv's MCP types;
only the state is specific to this environment."""

from __future__ import annotations

from openenv.core.env_server.types import State
from pydantic import Field


class DesignState(State):
    task_id: str | None = None
    split: str | None = None
    mode: str = "reference"
    surface: str = "tools"
    skin: str = "figma"
    n_elements: int = 0
    done: bool = False
    reward: float | None = None
    editor_url: str | None = None
    grade: dict | None = Field(default=None, description="Score breakdown after submit")
