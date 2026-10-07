"""Clients for a running RetroEnv server (local or a Space).

``RetroEnvClient`` holds one WebSocket session, so one episode at a time; open
one client per concurrent episode. ``RemoteRetroRouteEnv`` exposes the same
tools as plain methods for TRL's ``environment_factory``, mirroring the
in-process ``retroenv.training.RetroRouteTrainingEnv``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

from .config import ENV_NAME


@dataclass
class ToolOutcome:
    name: str
    result: Any
    done: bool
    reward: float | None
    error: str | None = None


def _tool_payload(result: Any) -> Any:
    """The tool's return value from a CallToolResult (object or JSON dict)."""
    if result is None:
        return None
    data = getattr(result, "data", None)
    if data is not None:
        return data
    if isinstance(result, dict):
        if result.get("data") is not None:
            return result["data"]
        structured = result.get("structured_content") or result.get("structuredContent")
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        for block in result.get("content") or []:
            if block.get("type") == "text":
                try:
                    return json.loads(block["text"])
                except (TypeError, ValueError):
                    return block["text"]
    return result


class RetroEnvClient:
    def __init__(self, base_url: str, message_timeout_s: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self._env = MCPToolClient(self.base_url, message_timeout_s=message_timeout_s).sync()
        self._tools: list[dict[str, Any]] | None = None

    # --- Task API over HTTP ----------------------------------------------

    def _post(self, route: str, payload: dict[str, Any]) -> Any:
        response = httpx.post(f"{self.base_url}/{ENV_NAME}/{route}", json=payload, timeout=600)
        response.raise_for_status()
        return response.json()

    def splits(self) -> list[dict[str, Any]]:
        response = httpx.get(f"{self.base_url}/{ENV_NAME}/splits", timeout=600)
        response.raise_for_status()
        return response.json()

    def num_tasks(self, split: str) -> int:
        value = self._post("num_tasks", {"split": split})
        return int(value["num_tasks"] if isinstance(value, dict) else value)

    def task(self, split: str, index: int) -> dict[str, Any]:
        value = self._post("task", {"split": split, "index": index})
        return value.get("task", value) if isinstance(value, dict) else value

    def tasks(self, split: str, start: int | None = None, stop: int | None = None) -> list[dict[str, Any]]:
        value = self._post("task_range", {"split": split, "start": start, "stop": stop})
        return value.get("tasks", value) if isinstance(value, dict) else value

    # --- Episode ------------------------------------------------------------

    def reset(
        self, split: str, index: int | None = None, task_id: str | None = None, episode_id: str | None = None
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"split": split}
        if index is not None:
            kwargs["index"] = index
        if task_id is not None:
            kwargs["task_id"] = task_id
        if episode_id is not None:
            kwargs["episode_id"] = episode_id
        result = self._env.reset(**kwargs)
        return dict(result.observation.metadata or {})

    def openai_tools(self) -> list[dict[str, Any]]:
        """Discovered MCP tools as OpenAI function definitions."""
        if self._tools is None:
            self._tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
                for tool in self._env.list_tools()
            ]
        return self._tools

    def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        step = self._env.step(CallToolAction(tool_name=name, arguments=arguments))
        observation = step.observation
        error = getattr(observation, "error", None)
        return ToolOutcome(
            name=name,
            result=_tool_payload(getattr(observation, "result", None)),
            done=bool(step.done),
            reward=step.reward,
            error=getattr(error, "message", None) if error else None,
        )

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> "RetroEnvClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class RemoteRetroRouteEnv:
    """TRL ``environment_factory`` tools backed by a RetroEnv server.

    Public methods other than ``reset`` and ``get_reward`` are agent tools.
    Set ``RETROENV_SERVER`` and pass ``environment_factory=RemoteRetroRouteEnv``.
    """

    def __init__(self, server: str | None = None):
        import os

        self._client = RetroEnvClient(server or os.environ["RETROENV_SERVER"])
        self._reward = 0.0

    def reset(self, split: str = "train", index: int = 0, **_: Any) -> str:
        self._reward = 0.0
        return self._client.reset(split=split, index=index)["prompt"]

    def get_reward(self) -> float:
        return self._reward

    def _close(self) -> None:
        """Release the WebSocket session. TRL and the evaluators call this."""
        self._client.close()

    def _call(self, name: str, **arguments: Any) -> str:
        outcome = self._client.call(name, arguments)
        if outcome.reward is not None:
            self._reward = float(outcome.reward)
        payload = outcome.result if outcome.error is None else {"error": outcome.error}
        return json.dumps(payload, sort_keys=True)

    def inspect_molecule(self, smiles: str) -> str:
        """Inspect a molecule with RDKit: formula, scaffold, rings, charge and stereo.

        Args:
            smiles: The molecule as SMILES.
        """
        return self._call("inspect_molecule", smiles=smiles)

    def pubchem_lookup(self, query: str) -> str:
        """Canonicalize a SMILES or look up the frozen molecule cache.

        Args:
            query: A SMILES, or a name or CAS number present in the cache.
        """
        return self._call("pubchem_lookup", query=query)

    def stock_retrieve(self, query: str, mode: str = "auto", limit: int = 10) -> str:
        """Search the purchasable stock: exact, inchikey, class, substructure or similarity.

        Args:
            query: SMILES, InChIKey, class name or SMARTS.
            mode: One of auto, exact, inchikey, class, substructure, similarity.
            limit: Maximum results, at most 20.
        """
        return self._call("stock_retrieve", query=query, mode=mode, limit=limit)

    def reaction_precedent_search(self, product_smiles: str = "", reaction_class: str = "", limit: int = 10) -> str:
        """Find analogous reactions from training-split records.

        Args:
            product_smiles: Product to find analogues for.
            reaction_class: Optional reaction class filter.
            limit: Maximum results, at most 20.
        """
        return self._call(
            "reaction_precedent_search", product_smiles=product_smiles, reaction_class=reaction_class, limit=limit
        )

    def validate_disconnection(self, product_smiles: str, reactants: list[str]) -> str:
        """Check one proposed cut against train-visible reactions and frequent templates.

        Args:
            product_smiles: The product of the step.
            reactants: The proposed reactant SMILES.
        """
        return self._call("validate_disconnection", product_smiles=product_smiles, reactants=reactants)

    def reaction_class_lookup(self, product_smiles: str, reactants: list[str]) -> str:
        """Name the reaction class of a proposed cut.

        Args:
            product_smiles: The product of the step.
            reactants: The proposed reactant SMILES.
        """
        return self._call("reaction_class_lookup", product_smiles=product_smiles, reactants=reactants)

    def reaction_conditions_search(
        self, product_smiles: str = "", reactants: list[str] | None = None, reaction_class: str = "", limit: int = 5
    ) -> str:
        """Find reported conditions for a cut or close precedents.

        Args:
            product_smiles: The product of the step.
            reactants: The proposed reactant SMILES.
            reaction_class: Optional reaction class.
            limit: Maximum results, at most 20.
        """
        return self._call(
            "reaction_conditions_search",
            product_smiles=product_smiles,
            reactants=reactants or [],
            reaction_class=reaction_class,
            limit=limit,
        )

    def search_literature(self, product_smiles: str = "", reaction_class: str = "", limit: int = 5) -> str:
        """Search frozen citation metadata attached to training precedents.

        Args:
            product_smiles: Product to search for.
            reaction_class: Optional reaction class filter.
            limit: Maximum results, at most 20.
        """
        return self._call(
            "search_literature", product_smiles=product_smiles, reaction_class=reaction_class, limit=limit
        )

    def emit_routes(self, submission: dict) -> str:
        """Submit the final route trees and end the episode.

        Args:
            submission: {"routes": [root molecule nodes]} in retro-route-graph-v1 form.
        """
        return self._call("emit_routes", submission=submission)
