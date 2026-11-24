"""The RetroEnv OpenEnv environment: one retrosynthesis episode per WebSocket session.

Tools are MCP calls over the shared ``RetroRouteSession`` from core. Every tool
advertises the JSON schema from ``retroenv.tools``, while argument validation
stays loose so a malformed route submission reaches the verifier and is scored
instead of rejected. ``emit_routes`` ends the episode: that step's observation
carries ``done=True`` and the dense reward, exactly once.
"""

from __future__ import annotations

import asyncio
import functools
import os
import random
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import Tool
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import CallToolObservation
from openenv.core.env_server.types import EnvironmentMetadata, Observation, State
from retroenv.tools import tool_spec

from .config import ENV_NAME, Resources, shared_resources

# OpenEnv's default is 30 s; a scored submission or a stock substructure search can take longer.
TOOL_TIMEOUT_S = float(os.getenv("RETROENV_TOOL_TIMEOUT", "300"))


def _threaded(handler: Any) -> Any:
    """Run a synchronous tool off the event loop, so one slow call cannot stall other sessions."""

    @functools.wraps(handler)
    async def run(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(handler, *args, **kwargs)

    return run


DESCRIPTION = (
    "Plan retrosynthesis routes for a target molecule from a fixed purchasable stock. "
    "Tools inspect molecules, search the stock and train-visible reactions, and check "
    "disconnections against a frozen reaction library; emit_routes submits molecule/reaction trees and returns the "
    "verifier's dense reward."
)


class RetroRouteState(State):
    split: str = ""
    task_index: int = -1
    task_id: str = ""
    toolset: str = "full"
    tool_calls: int = 0
    max_tool_calls: int = 0
    done: bool = False
    reward: float | None = None


class RetroRouteEnvironment(MCPEnvironment):
    # Each WebSocket session gets its own instance; only read-only data is shared.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, resources: Resources | None = None):
        self.resources = resources or shared_resources()
        settings = self.resources.settings
        self.session = self.resources.session()
        self._state = RetroRouteState(toolset=settings.toolset, max_tool_calls=settings.max_tool_calls)
        self._rng = random.Random()
        mcp = FastMCP("retroenv")
        handlers = self._handlers()
        for name in self.session.tool_names:
            spec = tool_spec(name)
            tool = Tool.from_function(_threaded(handlers[name]), name=name, description=spec["description"])
            mcp.add_tool(tool.model_copy(update={"parameters": spec["parameters"]}))
        super().__init__(mcp)

    def _handlers(self) -> dict[str, Any]:
        session = self.session

        def inspect_molecule(smiles: str) -> dict[str, Any]:
            return self._record(session.inspect_molecule(smiles))

        def pubchem_lookup(query: str) -> dict[str, Any]:
            return self._record(session.pubchem_lookup(query))

        def stock_retrieve(query: str, mode: str = "auto", limit: int = 10) -> dict[str, Any]:
            return self._record(session.stock_retrieve(query, mode, limit))

        def reaction_precedent_search(
            product_smiles: str = "", reaction_class: str = "", limit: int = 10
        ) -> dict[str, Any]:
            return self._record(session.reaction_precedent_search(product_smiles, reaction_class, limit))

        def validate_disconnection(product_smiles: str, reactants: list[str] | str) -> dict[str, Any]:
            return self._record(session.validate_disconnection(product_smiles, reactants))

        def reaction_class_lookup(product_smiles: str, reactants: list[str] | str) -> dict[str, Any]:
            return self._record(session.reaction_class_lookup(product_smiles, reactants))

        def reaction_conditions_search(
            product_smiles: str = "",
            reactants: list[str] | str | None = None,
            reaction_class: str = "",
            limit: int = 5,
        ) -> dict[str, Any]:
            return self._record(
                session.reaction_conditions_search(product_smiles, reactants or [], reaction_class, limit)
            )

        def search_literature(product_smiles: str = "", reaction_class: str = "", limit: int = 5) -> dict[str, Any]:
            return self._record(session.search_literature(product_smiles, reaction_class, limit))

        def emit_routes(submission: Any) -> dict[str, Any]:
            return self._record(session.emit_routes(submission))

        return {
            "inspect_molecule": inspect_molecule,
            "pubchem_lookup": pubchem_lookup,
            "stock_retrieve": stock_retrieve,
            "reaction_precedent_search": reaction_precedent_search,
            "validate_disconnection": validate_disconnection,
            "reaction_class_lookup": reaction_class_lookup,
            "reaction_conditions_search": reaction_conditions_search,
            "search_literature": search_literature,
            "emit_routes": emit_routes,
        }

    def _record(self, value: dict[str, Any]) -> dict[str, Any]:
        self._state.step_count += 1
        self._state.tool_calls = self.session.tool_calls
        if self.session.done and not self._state.done:
            self._state.done = True
            self._state.reward = float((self.session.final_score or {}).get("reward", 0.0))
        return value

    # --- Task API (metadata only; never references) -------------------------

    def list_splits(self) -> list[dict[str, Any]]:
        store = self.resources.store
        return [
            {
                "name": split,
                "type": "train" if split == "train" else ("validation" if split == "dev" else "test"),
                "num_tasks": len(store.tasks(split)),
                "default": split == self.resources.settings.default_split,
            }
            for split in store.splits()
        ]

    def num_tasks(self, split: str) -> int:
        return len(self.resources.store.tasks(split))

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        tasks = self.resources.store.tasks(split)
        if not 0 <= int(index) < len(tasks):
            raise IndexError(f"task index {index} is out of range for split {split!r}")
        return {"index": int(index), **tasks[int(index)].to_dict(include_hidden=False)}

    def get_task_range(self, split: str, start: int | None = None, stop: int | None = None) -> list[dict[str, Any]]:
        indices = range(*slice(start, stop).indices(self.num_tasks(split)))
        return [self.get_task(split, index) for index in indices]

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        return self.get_task_range(split)

    # --- Episode lifecycle ---------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        split: str | None = None,
        index: int | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Start an episode. Choose the task by ``index`` or ``task_id``; with
        neither, ``seed`` picks one deterministically, else one is random."""
        store = self.resources.store
        split = split or self.resources.settings.default_split
        tasks = store.tasks(split)
        if not tasks:
            raise IndexError(f"split {split!r} contains no tasks")
        if task_id is not None:
            index = store.position(split, task_id)
            if index is None:
                raise KeyError(f"task {task_id!r} is not in split {split!r}")
        elif index is None:
            index = int(seed) % len(tasks) if seed is not None else self._rng.randrange(len(tasks))
        index = int(index)
        if not 0 <= index < len(tasks):
            raise IndexError(f"task index {index} is out of range for split {split!r}")
        task = tasks[index]
        opening = self.session.reset(task, store.stock(task.stock_id), episode_id=episode_id)
        settings = self.resources.settings
        self._state = RetroRouteState(
            episode_id=self.session.episode_id,
            split=split,
            task_index=index,
            task_id=task.task_id,
            toolset=settings.toolset,
            max_tool_calls=settings.max_tool_calls,
        )
        return Observation(
            done=False,
            reward=None,
            metadata={
                "prompt": opening["prompt"],
                "episode_id": opening["episode_id"],
                "split": split,
                "index": index,
                "task_id": task.task_id,
                "target_smiles": task.target_smiles,
                "variant": task.variant,
                "max_depth": task.max_depth,
                "constraints": task.constraints.to_dict(),
                "min_routes": task.min_routes,
                "max_routes": task.max_routes,
                "stock_id": task.stock_id,
                "toolset": settings.toolset,
                "tools": list(self.session.tool_names),
                "max_tool_calls": settings.max_tool_calls,
            },
        )

    def step(self, action: Any, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        was_done = self.session.done
        observation = super().step(action, timeout_s=timeout_s or TOOL_TIMEOUT_S, **kwargs)
        return self._mark_terminal(observation, was_done)

    async def step_async(self, action: Any, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        was_done = self.session.done
        observation = await super().step_async(action, timeout_s=timeout_s or TOOL_TIMEOUT_S, **kwargs)
        return self._mark_terminal(observation, was_done)

    def _mark_terminal(self, observation: Observation, was_done: bool) -> Observation:
        if self.session.done and isinstance(observation, CallToolObservation):
            observation.done = True
            # Report the episode reward once, on the call that ended it.
            observation.reward = None if was_done else self._state.reward
        return observation

    def _step_impl(self, action: Any, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        return Observation(
            done=self.session.done,
            reward=None,
            metadata={"error": "Use MCP actions: list_tools, then call_tool."},
        )

    @property
    def state(self) -> RetroRouteState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name=ENV_NAME,
            description=DESCRIPTION,
            version="0.3.0",
        )
