"""Pure-Python multi-turn RetroRoute session used by every serving wrapper."""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterable

from .chemistry import (
    ChemistryError,
    canonicalize_components,
    canonicalize_smiles,
    inspect_molecule as inspect_with_rdkit,
    stable_hash,
)
from .models import RetroTask
from .retrieval import PrecedentIndex, StockIndex, cached_stock_index, molecule_lookup
from .verifier import RouteVerifier


PROMPT = """Plan retrosyntheses for the target {target} in at most {max_steps} step(s).
Return {min_routes} to {max_routes} molecule/reaction trees by calling emit_routes
exactly once. The stock is not in this prompt: stock_retrieve is its only access
surface and a leaf may claim in_stock=true only after an exact lookup hit.

Each tree must alternate molecule -> reaction -> molecule. A molecule has
{{"type":"mol","smiles":"...","in_stock":false,"children":[]}}. A reaction has
type="reaction", is_reaction=true, metadata (explanation, reaction_class,
confidence as a numeric value from 0 to 1, literature as a list, precursor_roles
as an object), and molecule children. Distinct routes
must differ at the first cut. Every item in submission.routes must be the root
molecule object directly: never wrap it in route, root, tree, or route_id keys.
Validate proposed cuts before emitting. Dataset
support is not proof that a reaction will work experimentally."""


class RetroRouteSession:
    """Stateful tool session with private references and a bounded oracle budget."""

    TOOL_NAMES = (
        "inspect_molecule",
        "pubchem_lookup",
        "stock_retrieve",
        "reaction_precedent_search",
        "validate_disconnection",
        "reaction_class_lookup",
        "reaction_conditions_search",
        "search_literature",
        "emit_routes",
    )

    def __init__(
        self,
        *,
        max_tool_calls: int = 32,
        max_search_results: int = 20,
        precedent_index: PrecedentIndex | None = None,
        pubchem_cache: dict[str, dict[str, Any]] | None = None,
    ):
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        self.max_tool_calls = max_tool_calls
        self.max_search_results = max_search_results
        self.precedent_index = precedent_index or PrecedentIndex(())
        self.pubchem_cache = pubchem_cache or {}
        self.verifier = RouteVerifier()
        self.task: RetroTask | None = None
        self.stock: frozenset[str] = frozenset()
        self.stock_index: StockIndex | None = None
        self.episode_id = ""
        self.tool_calls = 0
        self.done = False
        self.candidates: list[dict[str, Any]] = []
        self.validations: list[dict[str, Any]] = []
        self.final_score: dict[str, Any] | None = None

    def reset(
        self,
        task: RetroTask,
        stock: Iterable[str],
        *,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        self.task = task
        raw_stock = stock if isinstance(stock, frozenset) else frozenset(stock)
        self.stock_index = cached_stock_index(raw_stock)
        self.stock = frozenset(self.stock_index.smiles)
        self.episode_id = episode_id or str(uuid.uuid4())
        self.tool_calls = 0
        self.done = False
        self.candidates = []
        self.validations = []
        self.final_score = None
        return self.observation("Episode started.")

    def observation(self, feedback: str = "") -> dict[str, Any]:
        task = self._require_task()
        return {
            "episode_id": self.episode_id,
            "task_id": task.task_id,
            "mode": task.mode,
            "target_smiles": task.target_smiles,
            "max_steps": task.max_steps,
            "stock_id": task.stock_id,
            "prompt": PROMPT.format(
                target=task.target_smiles,
                max_steps=task.max_steps,
                min_routes=task.min_routes,
                max_routes=task.max_routes,
            ),
            "available_tools": list(self.TOOL_NAMES),
            "tool_calls_used": self.tool_calls,
            "tool_calls_remaining": max(0, self.max_tool_calls - self.tool_calls),
            "candidate_count": len(self.candidates),
            "validation_count": len(self.validations),
            "done": self.done,
            "feedback": feedback,
            # Deliberately no reference_routes, source IDs, or reference count.
        }

    def inspect_molecule(self, smiles: str) -> dict[str, Any]:
        blocked = self._consume("inspect_molecule")
        if blocked:
            return blocked
        try:
            return inspect_with_rdkit(smiles)
        except Exception as exc:
            return {"valid": False, "error": str(exc)}

    def pubchem_lookup(self, query: str) -> dict[str, Any]:
        """Resolve a SMILES locally; cached name/CAS hydration is build-time only."""
        blocked = self._consume("pubchem_lookup")
        return blocked or molecule_lookup(query, self.pubchem_cache)

    def stock_retrieve(
        self,
        query: str,
        mode: str = "auto",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search the selected stock with a hard 20-result cap."""
        blocked = self._consume("stock_retrieve")
        if blocked:
            return blocked
        if self.stock_index is None:  # pragma: no cover - reset contract
            raise RuntimeError("reset() must be called before stock retrieval")
        return self.stock_index.retrieve(
            query, mode=mode, limit=min(limit, self.max_search_results)
        )

    def reaction_precedent_search(
        self,
        product_smiles: str = "",
        reaction_class: str = "",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Return capped analogues from the training-visible precedent index."""
        blocked = self._consume("reaction_precedent_search")
        if blocked:
            return blocked
        task = self._require_task()
        try:
            return self.precedent_index.search(
                task_id=task.task_id,
                product_smiles=product_smiles or None,
                reaction_class=reaction_class or None,
                limit=min(limit, self.max_search_results),
            )
        except Exception as exc:
            return {"results": [], "error": str(exc)}

    def propose_disconnection(
        self,
        product_smiles: str,
        reactants: Iterable[str] | str,
        reaction_class: str | None = None,
    ) -> dict[str, Any]:
        blocked = self._consume("propose_disconnection")
        if blocked:
            return blocked
        try:
            product = canonicalize_smiles(product_smiles)
            reactant_values = (
                reactants.split(".") if isinstance(reactants, str) else reactants
            )
            canonical_reactants = canonicalize_components(reactant_values)
            if not canonical_reactants:
                raise ChemistryError("a disconnection needs at least one reactant")
            leakage = product in canonical_reactants
            candidate = {
                "candidate_id": stable_hash(
                    f"{product}<<{'.'.join(canonical_reactants)}:{len(self.candidates)}",
                    prefix="cand_",
                    length=16,
                ),
                "product": product,
                "reactants": list(canonical_reactants),
                "reaction_class": reaction_class or None,
                "structurally_valid": not leakage,
                "warning": (
                    "product leakage: product is unchanged among reactants"
                    if leakage
                    else None
                ),
            }
            self.candidates.append(candidate)
            return candidate
        except Exception as exc:
            return {"structurally_valid": False, "error": str(exc)}

    def validate_step(
        self,
        product_smiles: str,
        reactants: Iterable[str] | str,
        reaction_class: str | None = None,
    ) -> dict[str, Any]:
        blocked = self._consume("validate_step")
        if blocked:
            return blocked
        task = self._require_task()
        reactant_values = reactants.split(".") if isinstance(reactants, str) else reactants
        result = self.verifier.validate_step(
            task, product_smiles, reactant_values, reaction_class
        ).to_dict()
        self.validations.append(result)
        return result

    def validate_disconnection(
        self,
        product_smiles: str,
        reactants: Iterable[str] | str,
        reaction_class: str | None = None,
    ) -> dict[str, Any]:
        """Validate one proposed cut without exposing the reference route."""
        return self.validate_step(product_smiles, reactants, reaction_class)

    def reaction_class_lookup(
        self,
        product_smiles: str,
        reactants: Iterable[str] | str,
    ) -> dict[str, Any]:
        """Name a cut only after the agent supplies a supported disconnection."""
        blocked = self._consume("reaction_class_lookup")
        if blocked:
            return blocked
        task = self._require_task()
        reactant_values = reactants.split(".") if isinstance(reactants, str) else reactants
        result = self.verifier.validate_step(task, product_smiles, reactant_values)
        if result.support == "none":
            return {"supported": False, "reaction_class": None}
        reaction_class = self._matched_step(result.matched_reaction_id).reaction_class
        return {
            "supported": result.valid,
            "reaction_class": reaction_class or "unclassified",
            "support": result.support,
        }

    def reaction_conditions_search(
        self,
        product_smiles: str = "",
        reactants: Iterable[str] | str = (),
        reaction_class: str = "",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Return reported conditions for a supported cut or close precedents."""
        blocked = self._consume("reaction_conditions_search")
        if blocked:
            return blocked
        task = self._require_task()
        if product_smiles and reactants:
            reactant_values = reactants.split(".") if isinstance(reactants, str) else reactants
            validation = self.verifier.validate_step(
                task, product_smiles, reactant_values, reaction_class or None
            )
            if validation.support != "none":
                step = self._matched_step(validation.matched_reaction_id)
                return {
                    "supported": validation.valid,
                    "reaction_class": step.reaction_class,
                    "conditions": list(step.conditions)[:limit],
                    "source": "hidden matched record",
                }
        result = self.precedent_index.search(
            task_id=task.task_id,
            product_smiles=product_smiles or None,
            reaction_class=reaction_class or None,
            limit=min(limit, self.max_search_results),
        )
        return {
            "supported": False,
            "conditions": [
                condition
                for row in result["results"]
                for condition in row.get("conditions", [])
            ][:limit],
            "source": "training-visible analogues",
        }

    def search_literature(
        self,
        product_smiles: str = "",
        reaction_class: str = "",
        limit: int = 5,
    ) -> dict[str, Any]:
        """Search frozen citation metadata attached to training precedents."""
        blocked = self._consume("search_literature")
        if blocked:
            return blocked
        task = self._require_task()
        result = self.precedent_index.search(
            task_id=task.task_id,
            product_smiles=product_smiles or None,
            reaction_class=reaction_class or None,
            limit=min(limit, self.max_search_results),
        )
        citations = [
            citation
            for row in result["results"]
            for citation in row.get("literature", [])
        ]
        return {
            "results": citations[:limit],
            "returned": min(limit, len(citations)),
            "note": "frozen metadata only; live web search is disabled during rollouts",
        }

    def search_building_blocks(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Compatibility alias for old clients; prefer stock_retrieve."""
        return self.stock_retrieve(query, mode="auto", limit=limit)

    def submit_route(self, route: dict[str, Any] | list[dict[str, Any]] | str) -> dict[str, Any]:
        """Compatibility alias for one legacy flat route."""
        task = self._require_task()
        if self.done:
            return {"error": "episode is already complete", "done": True, "score": self.final_score}
        if isinstance(route, str):
            try:
                route = json.loads(route)
            except json.JSONDecodeError as exc:
                route = {"route": []}
                parse_error = f"route is not valid JSON: {exc}"
            else:
                parse_error = None
        else:
            parse_error = None
        result = self.verifier.score_route(task, route, self.stock)
        score = result.to_dict()
        if parse_error:
            score["hard_failures"] = [parse_error]
            score["reward"] = 0.0
            score["valid"] = False
            score["verification_tier"] = "rejected"
        self.done = True
        self.final_score = score
        return {
            "done": True,
            "score": score,
            "tool_calls_used": self.tool_calls,
            "candidate_count": len(self.candidates),
            "validation_count": len(self.validations),
        }

    def emit_routes(self, submission: dict[str, Any] | str) -> dict[str, Any]:
        """Submit the final renderable route trees; this is terminal."""
        self._require_task()
        if self.done:
            return {"error": "episode is already complete", "done": True, "score": self.final_score}
        result = self.verifier.score_submission(self.task, submission, self.stock)
        self.done = True
        self.final_score = result.to_dict()
        return {
            "done": True,
            "score": self.final_score,
            "tool_calls_used": self.tool_calls,
            "candidate_count": len(self.candidates),
            "validation_count": len(self.validations),
        }

    def _matched_step(self, reaction_id: str | None):
        task = self._require_task()
        for route in task.reference_routes:
            for step in route.steps:
                if step.reaction_id == reaction_id:
                    return step
        raise LookupError("matched hidden step disappeared")

    def _consume(self, tool_name: str) -> dict[str, Any] | None:
        self._require_task()
        if self.done:
            return {"error": "episode is complete", "done": True}
        if self.tool_calls >= self.max_tool_calls:
            return {
                "error": "tool-call budget exhausted; emit_routes is still available",
                "tool": tool_name,
                "tool_calls_remaining": 0,
            }
        self.tool_calls += 1
        return None

    def _require_task(self) -> RetroTask:
        if self.task is None:
            raise RuntimeError("reset() must be called before using tools")
        return self.task
