"""Clients for a running RetroEnv server (local or a Space).

``RetroEnvClient`` holds one WebSocket session, so one episode at a time; open
one client per concurrent episode. ``RemoteRetroRouteEnv`` exposes the same
tools as plain methods for TRL's ``environment_factory``, mirroring the
in-process ``retroenv.training.RetroRouteTrainingEnv``.

This module imports neither RDKit nor the server stack, so a trainer can install the
package with ``--no-deps`` next to ``openenv`` and ``httpx`` and talk to a server
running in its own environment.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import httpx
from openenv.core.env_server.mcp_types import CallToolAction
from openenv.core.mcp_client import MCPToolClient

# The server's environment name (``config`` re-exports it); defined here so importing the
# client does not load the task store.
ENV_NAME = "retro_route"


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

    The evaluation harness tells the model how many turns it has left and reserves the last one
    for ``emit_routes``; TRL's tool loop does neither, so a rollout tends to research until the
    loop ends and never submits, and a group of unsubmitted rollouts ties at 0. Each tool result
    here therefore carries the calls left in the server's budget, and a reminder to submit once
    few remain. The reward is unchanged: the server's grade of the one submission.

    With ``RETROENV_TRACE_PATH`` set, each graded episode appends one JSON line there (task,
    reward, whether it submitted, tool calls), which says why a group was flat.
    """

    REMIND_AT = 4  # tool calls left when the reminder to submit starts

    def __init__(self, server: str | None = None):
        import os

        self._server = server or os.environ["RETROENV_SERVER"]
        self._trace = os.environ.get("RETROENV_TRACE_PATH")
        self._client = RetroEnvClient(self._server)
        self._reward = 0.0
        self._failed = False
        self._opening: dict[str, Any] = {}
        self._calls = 0
        self._submitted = False
        self._passed = False
        self._failure: str | None = None

    @property
    def submitted(self) -> bool:
        """Whether this episode called emit_routes."""
        return self._submitted

    @property
    def passed(self) -> bool:
        """Whether the submission solved the task (the server's `valid`)."""
        return self._passed

    @property
    def failed(self) -> bool:
        """Whether the session broke, so the episode has no grade."""
        return self._failed

    def reset(self, split: str = "train", index: int = 0, **_: Any) -> str:
        self._reward, self._failed, self._calls, self._submitted, self._passed = 0.0, False, 0, False, False
        self._failure = None
        try:
            self._opening = self._client.reset(split=split, index=index)
        except Exception:
            # A session the server dropped stays dead in the client, and every later call on it
            # fails; open a fresh one once instead of losing this instance for the whole run.
            self._close()
            self._client = RetroEnvClient(self._server)
            self._opening = self._client.reset(split=split, index=index)
        return self._opening["prompt"]

    def get_reward(self) -> float:
        # A rollout whose session broke has no grade, which is not the same as a wrong answer:
        # NaN keeps it out of its group's baseline instead of scoring it 0.
        reward = math.nan if self._failed else self._reward
        if self._trace:
            record = {
                "task_id": self._opening.get("task_id"),
                "index": self._opening.get("index"),
                "reward": None if math.isnan(reward) else reward,
                "submitted": self._submitted,
                "passed": self._passed,
                "failure": self._failure,  # the grader's first hard failure, when the submission failed
                "tool_calls": self._calls,
                "failed": self._failed,
            }
            with open(self._trace, "a", encoding="utf-8") as trace:
                trace.write(json.dumps(record) + "\n")
        return reward

    def _close(self) -> None:
        """Release the WebSocket session. TRL and the evaluators call this."""
        try:
            self._client.close()
        except Exception:
            pass

    def _call(self, name: str, **arguments: Any) -> str:
        try:
            outcome = self._client.call(name, arguments)
        except Exception as exc:
            self._failed = True
            return json.dumps({"error": f"environment unavailable: {type(exc).__name__}"})
        if outcome.reward is not None:
            self._reward = float(outcome.reward)
        payload = outcome.result if outcome.error is None else {"error": outcome.error}
        if not isinstance(payload, dict):
            payload = {"result": payload}
        if name == "emit_routes":
            self._submitted = True
            score = payload.get("score") or {}
            self._passed = bool(score.get("valid"))
            failures = score.get("hard_failures") or ([payload["error"]] if "error" in payload else [])
            self._failure = str(failures[0])[:160] if failures and not self._passed else None
        else:
            self._calls += 1
            left = max(0, int(self._opening.get("max_tool_calls", 32)) - self._calls)
            payload = {**payload, "tool_calls_remaining": left}
            if left <= self.REMIND_AT:
                payload["reminder"] = (
                    "Few tool calls remain. Submit your best routes now with emit_routes; "
                    "an episode that never calls it scores 0."
                )
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

    # TRL shows the model this docstring as the tool's schema, where evaluation shows the server's
    # JSON schema. With only a one-line description, a quarter of training submissions had no
    # route starting at the target (2 of 30 in evaluation), so it says what that schema says.
    def emit_routes(self, submission: dict) -> str:
        """Submit the final route trees and end the episode. Call it exactly once.

        Each item of submission.routes is a root molecule node for the target itself, never
        wrapped in route, root or tree keys, and its smiles is the target exactly as given. A
        molecule node is {"type": "mol", "smiles": "...", "in_stock": false, "children": [...]}:
        an expanded molecule has in_stock false and exactly one reaction child; a leaf has no
        children and in_stock true only if stock_retrieve confirmed it. A reaction node is
        {"type": "reaction", "is_reaction": true, "metadata": {...}, "children": [...]}, whose
        children are its precursor molecule nodes and whose metadata holds explanation,
        reaction_class, confidence (0 to 1), literature (a list) and precursor_roles (an object).

        Args:
            submission: {"routes": [root molecule nodes]}, as described above; an object, not a string.
        """
        return self._call("emit_routes", submission=submission)
