"""OpenEnv/MCP wrapper around the pure-Python RetroRoute session."""

from __future__ import annotations

import random
from typing import Any

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation

from ..environment import RetroRouteSession
from ..retrieval import PrecedentIndex
from ..store import TaskStore
from .models import RetroRouteAction, RetroRouteObservation, RetroRouteState


class UnknownSplitError(KeyError, IndexError):
    pass


class RetroRouteEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        store: TaskStore,
        *,
        default_split: str = "train",
        max_tool_calls: int = 32,
        pubchem_cache: dict[str, dict[str, Any]] | None = None,
        precedent_index: PrecedentIndex | None = None,
    ):
        self.store = store
        if default_split not in store.splits():
            raise ValueError(
                f"default split {default_split!r} is not among {store.splits()}"
            )
        self.default_split = default_split
        # Eval-task references never enter retrieval: the searchable precedent
        # library is constructed strictly from the training split.
        training_tasks = store.tasks("train") if "train" in store.splits() else ()
        self.session = RetroRouteSession(
            max_tool_calls=max_tool_calls,
            precedent_index=precedent_index or PrecedentIndex(training_tasks),
            pubchem_cache=pubchem_cache,
        )
        self._state = RetroRouteState()
        self._rng = random.Random()
        self._last_tool_result: dict[str, Any] = {}
        mcp = FastMCP("retro_route_env")
        self._register_tools(mcp)
        super().__init__(mcp)

    def _register_tools(self, mcp: FastMCP) -> None:
        @mcp.tool
        def inspect_molecule(smiles: str) -> dict[str, Any]:
            """Inspect a SMILES with RDKit: formula, scaffold, rings, charge, and stereo."""
            return self._record(self.session.inspect_molecule(smiles))

        @mcp.tool
        def pubchem_lookup(query: str) -> dict[str, Any]:
            """Resolve a SMILES using the frozen PubChem-compatible molecule cache."""
            return self._record(self.session.pubchem_lookup(query))

        @mcp.tool
        def stock_retrieve(
            query: str, mode: str = "auto", limit: int = 10
        ) -> dict[str, Any]:
            """Search only the selected stock; results are capped at 20 molecules."""
            return self._record(self.session.stock_retrieve(query, mode, limit))

        @mcp.tool
        def reaction_precedent_search(
            product_smiles: str = "", reaction_class: str = "", limit: int = 10
        ) -> dict[str, Any]:
            """Find capped analogues from training-visible reaction records."""
            return self._record(
                self.session.reaction_precedent_search(
                    product_smiles, reaction_class, limit
                )
            )

        @mcp.tool
        def validate_disconnection(
            product_smiles: str,
            reactants: list[str],
            reaction_class: str = "",
        ) -> dict[str, Any]:
            """Check one cut against hidden evidence without revealing the answer."""
            return self._record(
                self.session.validate_disconnection(
                    product_smiles, reactants, reaction_class or None
                )
            )

        @mcp.tool
        def reaction_class_lookup(
            product_smiles: str,
            reactants: list[str],
        ) -> dict[str, Any]:
            """Name the class of a supported, agent-supplied cut."""
            return self._record(
                self.session.reaction_class_lookup(product_smiles, reactants)
            )

        @mcp.tool
        def reaction_conditions_search(
            product_smiles: str = "",
            reactants: list[str] | None = None,
            reaction_class: str = "",
            limit: int = 5,
        ) -> dict[str, Any]:
            """Return frozen reported conditions for a cut or close precedent."""
            return self._record(
                self.session.reaction_conditions_search(
                    product_smiles, reactants or [], reaction_class, limit
                )
            )

        @mcp.tool
        def search_literature(
            product_smiles: str = "", reaction_class: str = "", limit: int = 5
        ) -> dict[str, Any]:
            """Search frozen citation metadata; live web is disabled in rollouts."""
            return self._record(
                self.session.search_literature(product_smiles, reaction_class, limit)
            )

        @mcp.tool
        def emit_routes(submission: dict[str, Any]) -> dict[str, Any]:
            """Submit 1-5 renderable molecule/reaction trees and end the episode."""
            return self._record(self.session.emit_routes(submission))

    def _record(self, value: dict[str, Any]) -> dict[str, Any]:
        self._last_tool_result = value
        self._state.tool_calls = self.session.tool_calls
        self._state.submitted = self.session.done
        return value

    def list_splits(self) -> list[dict[str, Any]]:
        return [
            {
                "name": split,
                "type": "train" if split == "train" else ("test" if split in {"eval", "stress"} else "validation"),
                "num_tasks": len(self.store.tasks(split)),
                "default": split == self.default_split,
            }
            for split in self.store.splits()
        ]

    def num_tasks(self, split: str) -> int:
        return len(self.store.tasks(split))

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        return self.store.public_task(split, index)

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        return [self.store.public_task(split, index) for index in range(self.num_tasks(split))]

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        indices = range(*slice(start, stop).indices(self.num_tasks(split)))
        return [self.store.public_task(split, index) for index in indices]

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        split: str | None = None,
        index: int | None = None,
        **kwargs: Any,
    ) -> RetroRouteObservation:
        split = split or self.default_split
        tasks = self.store.tasks(split)
        if not tasks:
            raise IndexError(f"split {split!r} contains no tasks")
        if index is None:
            index = int(seed) % len(tasks) if seed is not None else self._rng.randrange(len(tasks))
        if not 0 <= int(index) < len(tasks):
            raise IndexError(f"task index {index} out of range for {split!r}")
        task = tasks[int(index)]
        opening = self.session.reset(
            task, self.store.stock(task.stock_id), episode_id=episode_id
        )
        self._state = RetroRouteState(
            episode_id=self.session.episode_id,
            step_count=0,
            split=split,
            task_index=int(index),
            task_id=task.task_id,
        )
        return self._observation(opening["feedback"], {})

    @property
    def state(self) -> RetroRouteState:
        return self._state

    def _step_impl(self, action: Action, **kwargs: Any) -> Observation:
        if not isinstance(action, RetroRouteAction):
            raise TypeError(f"unsupported action: {type(action).__name__}")
        self._state.step_count += 1
        if action.op == "inspect":
            result = self.session.inspect_molecule(action.smiles or "")
        elif action.op == "pubchem":
            result = self.session.pubchem_lookup(action.query or action.smiles or "")
        elif action.op == "stock":
            result = self.session.stock_retrieve(
                action.query or "", action.mode, action.limit
            )
        elif action.op == "precedent":
            result = self.session.reaction_precedent_search(
                action.product_smiles or "", action.reaction_class or "", action.limit
            )
        elif action.op == "propose":
            result = self.session.propose_disconnection(
                action.product_smiles or "", action.reactants or [], action.reaction_class
            )
        elif action.op == "validate":
            result = self.session.validate_disconnection(
                action.product_smiles or "", action.reactants or [], action.reaction_class
            )
        elif action.op == "classify":
            result = self.session.reaction_class_lookup(
                action.product_smiles or "", action.reactants or []
            )
        elif action.op == "conditions":
            result = self.session.reaction_conditions_search(
                action.product_smiles or "",
                action.reactants or [],
                action.reaction_class or "",
                action.limit,
            )
        elif action.op == "literature":
            result = self.session.search_literature(
                action.product_smiles or "", action.reaction_class or "", action.limit
            )
        elif action.op == "emit":
            result = self.session.emit_routes(action.submission)
        elif action.op == "search":
            result = self.session.search_building_blocks(action.query or "", action.limit)
        elif action.op == "submit":
            result = self.session.submit_route(action.route)
        else:  # pragma: no cover - Literal validation prevents this
            raise ValueError(action.op)
        self._record(result)
        feedback = result.get("error") or ("Route scored." if self.session.done else "Tool completed.")
        return self._observation(feedback, result)

    def _observation(
        self, feedback: str, tool_result: dict[str, Any]
    ) -> RetroRouteObservation:
        value = self.session.observation(feedback)
        score = self.session.final_score
        return RetroRouteObservation(
            done=self.session.done,
            reward=(score or {}).get("reward", 0.0) if self.session.done else 0.0,
            prompt=value["prompt"],
            task_id=value["task_id"],
            mode=value["mode"],
            target_smiles=value["target_smiles"],
            max_steps=value["max_steps"],
            min_routes=self.session.task.min_routes if self.session.task else 1,
            max_routes=self.session.task.max_routes if self.session.task else 5,
            stock_id=value["stock_id"],
            available_tools=value["available_tools"],
            tool_calls_used=value["tool_calls_used"],
            tool_calls_remaining=value["tool_calls_remaining"],
            feedback=feedback,
            tool_result=tool_result,
            score=score,
            metadata={"split": self._state.split, "task_index": self._state.task_index},
        )
