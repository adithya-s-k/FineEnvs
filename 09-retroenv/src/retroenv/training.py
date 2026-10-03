"""TRL ``environment_factory`` adapter over the same RetroRoute core."""

from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from .environment import RetroRouteSession
from .retrieval import PrecedentIndex
from .store import TaskStore


_TRACE_LOCK = threading.Lock()


@lru_cache(maxsize=8)
def _resources(tasks_dir: str, stocks_dir: str) -> tuple[TaskStore, PrecedentIndex]:
    store = TaskStore(tasks_dir, stocks_dir)
    training_tasks = store.tasks("train") if "train" in store.splits() else ()
    return store, PrecedentIndex(training_tasks)


class RetroRouteTrainingEnv:
    """Text-only multi-tool environment compatible with TRL GRPOTrainer.

    Public methods other than ``reset`` and ``get_reward`` are agent tools.
    The adapter owns no chemistry logic; it delegates to ``RetroRouteSession``,
    the same core used by the OpenEnv/MCP server.
    """

    def __init__(
        self,
        tasks_dir: str | Path | None = None,
        stocks_dir: str | Path | None = None,
        *,
        max_tool_calls: int | None = None,
        trace_path: str | Path | None = None,
    ):
        root = Path(__file__).resolve().parents[2]
        task_path = str(
            tasks_dir or os.getenv("RETROENV_TASKS_DIR", root / "sample/tasks-private")
        )
        stock_path = str(
            stocks_dir or os.getenv("RETROENV_STOCKS_DIR", root / "sample/stocks")
        )
        self.store, precedent_index = _resources(task_path, stock_path)
        self.session = RetroRouteSession(
            max_tool_calls=max_tool_calls
            or int(os.getenv("RETROENV_MAX_TOOL_CALLS", "32")),
            precedent_index=precedent_index,
        )
        self.trace_path = Path(
            trace_path or os.getenv("RETROENV_TRACE_PATH", "")
        ) if (trace_path or os.getenv("RETROENV_TRACE_PATH")) else None
        self._task: dict[str, Any] = {}
        self._trace: list[dict[str, Any]] = []

    def reset(self, split: str = "train", index: int = 0, **_: Any) -> list[dict[str, str]]:
        """Start one deterministic indexed task and return its private-free prompt."""
        task = self.store.task(split, int(index))
        opening = self.session.reset(task, self.store.stock(task.stock_id))
        self._task = {"split": split, "index": int(index), "task_id": task.task_id}
        self._trace = []
        return self._blocks(opening)

    def get_reward(self) -> float:
        """Return environment-owned terminal reward and optionally persist a trace."""
        reward = float((self.session.final_score or {}).get("reward", 0.0))
        if self.trace_path and self._task:
            row = {
                "task": self._task,
                "reward": reward,
                "score": self.session.final_score,
                "turns": self._trace,
            }
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with _TRACE_LOCK, self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return reward

    def inspect_molecule(self, smiles: str) -> list[dict[str, str]]:
        """Inspect formula, scaffold, rings, charge, and stereochemistry."""
        return self._call("inspect_molecule", smiles=smiles)

    def pubchem_lookup(self, query: str) -> list[dict[str, str]]:
        """Canonicalize SMILES using the deterministic local molecule resolver."""
        return self._call("pubchem_lookup", query=query)

    def stock_retrieve(
        self, query: str, mode: str = "auto", limit: int = 10
    ) -> list[dict[str, str]]:
        """Search the selected stock, returning at most 20 candidates."""
        return self._call("stock_retrieve", query=query, mode=mode, limit=limit)

    def reaction_precedent_search(
        self, product_smiles: str = "", reaction_class: str = "", limit: int = 10
    ) -> list[dict[str, str]]:
        """Search capped training-visible reaction precedents."""
        return self._call(
            "reaction_precedent_search",
            product_smiles=product_smiles,
            reaction_class=reaction_class,
            limit=limit,
        )

    def validate_disconnection(
        self,
        product_smiles: str,
        reactants: list[str],
        reaction_class: str = "",
    ) -> list[dict[str, str]]:
        """Validate one agent-supplied retrosynthetic cut."""
        return self._call(
            "validate_disconnection",
            product_smiles=product_smiles,
            reactants=reactants,
            reaction_class=reaction_class or None,
        )

    def reaction_class_lookup(
        self, product_smiles: str, reactants: list[str]
    ) -> list[dict[str, str]]:
        """Name the class of a supported, supplied cut."""
        return self._call(
            "reaction_class_lookup",
            product_smiles=product_smiles,
            reactants=reactants,
        )

    def reaction_conditions_search(
        self,
        product_smiles: str = "",
        reactants: list[str] | None = None,
        reaction_class: str = "",
        limit: int = 5,
    ) -> list[dict[str, str]]:
        """Return frozen reported conditions for a cut or analogue."""
        return self._call(
            "reaction_conditions_search",
            product_smiles=product_smiles,
            reactants=reactants or [],
            reaction_class=reaction_class,
            limit=limit,
        )

    def search_literature(
        self, product_smiles: str = "", reaction_class: str = "", limit: int = 5
    ) -> list[dict[str, str]]:
        """Search frozen patent/citation metadata, never live web."""
        return self._call(
            "search_literature",
            product_smiles=product_smiles,
            reaction_class=reaction_class,
            limit=limit,
        )

    def emit_routes(self, submission: dict[str, Any]) -> list[dict[str, str]]:
        """Emit the final renderable route trees and terminate scoring."""
        return self._call("emit_routes", submission=submission)

    def _call(self, name: str, **arguments: Any) -> list[dict[str, str]]:
        if self.session.done:
            result = {"done": True, "error": "episode is already complete"}
        else:
            result = getattr(self.session, name)(**arguments)
        self._trace.append({"tool": name, "arguments": arguments, "result": result})
        return self._blocks(result)

    @staticmethod
    def _blocks(value: dict[str, Any]) -> list[dict[str, str]]:
        return [{"type": "text", "text": json.dumps(value, sort_keys=True)}]
