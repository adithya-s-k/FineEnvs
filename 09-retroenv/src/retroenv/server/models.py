"""OpenEnv wire models."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class RetroRouteAction(Action):
    op: Literal[
        "inspect",
        "pubchem",
        "stock",
        "precedent",
        "propose",
        "validate",
        "classify",
        "conditions",
        "literature",
        "emit",
        # Backward-compatible structured API operations.
        "search",
        "submit",
    ]
    smiles: str | None = None
    product_smiles: str | None = None
    reactants: list[str] | None = None
    reaction_class: str | None = None
    query: str | None = None
    mode: str = "auto"
    limit: int = 10
    route: Any = None
    submission: Any = None


class RetroRouteObservation(Observation):
    prompt: str = ""
    task_id: str = ""
    mode: str = ""
    target_smiles: str = ""
    max_steps: int = 0
    min_routes: int = 1
    max_routes: int = 5
    stock_id: str = ""
    available_tools: list[str] = Field(default_factory=list)
    tool_calls_used: int = 0
    tool_calls_remaining: int = 0
    feedback: str = ""
    tool_result: dict[str, Any] = Field(default_factory=dict)
    score: dict[str, Any] | None = None


class RetroRouteState(State):
    split: str = ""
    task_index: int = -1
    task_id: str = ""
    submitted: bool = False
    tool_calls: int = 0
