"""Pure-Python multi-turn RetroRoute session used by every serving wrapper."""

from __future__ import annotations

import uuid
from typing import Any, Iterable

from .chemistry import ChemistryError, canonicalize_components, canonicalize_smiles
from .chemistry import inspect_molecule as inspect_with_rdkit
from .classes import CONSTRAINABLE_CLASSES, classify_step
from .models import RetroTask
from .reactions import ReactionLibrary, StepKey, free_key
from .retrieval import PrecedentIndex, StockIndex, cached_stock_index, molecule_lookup
from .tools import tool_names
from .verifier import RouteVerifier

PROMPT = """Plan a retrosynthesis for the target {target}.
Return {route_range} by calling emit_routes exactly once. Each route's longest
linear sequence may have at most {max_depth} reaction(s).{constraints}

The stock is not in this prompt: stock_retrieve is its only access surface. Every
leaf must be in stock and may claim in_stock=true only after an exact lookup hit;
intermediates need not be in stock.

Each tree alternates molecule -> reaction -> molecule. A molecule is
{{"type":"mol","smiles":"...","in_stock":false,"children":[]}}. A reaction has
type="reaction", is_reaction=true, metadata (explanation, reaction_class,
confidence from 0 to 1, literature as a list, precursor_roles as an object), and
its precursor molecules as children. Distinct routes must differ at the first
disconnection. Every item in submission.routes is a root molecule object, never
wrapped in route, root or tree keys.

Each step is judged by a frozen reaction library (known reactions and frequent
reaction templates), not by matching a hidden answer; any valid route counts.
Library support is not proof that a reaction works experimentally."""


def describe_constraints(task: RetroTask) -> str:
    lines = []
    if task.constraints.forbidden_classes:
        names = ", ".join(task.constraints.forbidden_classes)
        lines.append(
            f"Do not use these reaction classes in any step: {names}. Classes are assigned by "
            "reaction_class_lookup, the same labeller the verifier uses."
        )
    if task.constraints.excluded_stock:
        names = ", ".join(task.constraints.excluded_stock)
        lines.append(
            f"These building blocks are unavailable for this task and stock_retrieve will not return them: {names}."
        )
    if task.min_routes > 1:
        lines.append(f"Return at least {task.min_routes} valid routes with different first disconnections.")
    return "".join(f"\nConstraint: {line}" for line in lines)


def route_range(task: RetroTask) -> str:
    if task.min_routes == task.max_routes:
        return f"exactly {task.min_routes} synthesis tree(s)"
    return f"{task.min_routes} to {task.max_routes} synthesis trees"


def task_prompt(task: RetroTask) -> str:
    """The episode prompt: the target, the depth budget and the constraints, never a hidden route."""
    return PROMPT.format(
        target=task.target_smiles,
        route_range=route_range(task),
        max_depth=task.max_depth,
        constraints=describe_constraints(task),
    )


class RetroRouteSession:
    """Stateful tool session; no tool reads the task's hidden routes."""

    def __init__(
        self,
        *,
        library: ReactionLibrary,
        tool_library: ReactionLibrary | None = None,
        precedent_index: PrecedentIndex | None = None,
        max_tool_calls: int = 32,
        max_search_results: int = 20,
        pubchem_cache: dict[str, dict[str, Any]] | None = None,
        toolset: str = "full",
    ):
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        self.toolset = toolset
        self.tool_names = tool_names(toolset)
        self.max_tool_calls = max_tool_calls
        self.max_search_results = max_search_results
        self.verifier = RouteVerifier(library)
        # validate_disconnection must only know train-visible evidence.
        self.tool_library = tool_library or library
        self.precedent_index = precedent_index or PrecedentIndex(())
        self.pubchem_cache = pubchem_cache or {}
        self.task: RetroTask | None = None
        self.stock: frozenset[str] = frozenset()
        self.stock_index: StockIndex | None = None
        self.excluded: frozenset[str] = frozenset()
        self._own_reactions: frozenset[StepKey] = frozenset()
        self.episode_id = ""
        self.tool_calls = 0
        self.done = False
        self.final_score: dict[str, Any] | None = None

    def reset(self, task: RetroTask, stock: Iterable[str], *, episode_id: str | None = None) -> dict[str, Any]:
        self.task = task
        raw_stock = stock if isinstance(stock, frozenset) else frozenset(stock)
        self.stock_index = cached_stock_index(raw_stock)
        self.stock = self.stock_index.known  # one shared set, so the verifier's per-stock cache hits
        self.excluded = frozenset(canonicalize_smiles(item) for item in task.constraints.excluded_stock)
        # A train task must not confirm its own known reactions by corpus lookup.
        self._own_reactions = (
            frozenset(free_key(step.product, step.reactants) for route in task.reference_routes for step in route.steps)
            if task.split == "train"
            else frozenset()
        )
        self.episode_id = episode_id or str(uuid.uuid4())
        self.tool_calls = 0
        self.done = False
        self.final_score = None
        return self.observation("Episode started.")

    def prompt(self) -> str:
        return task_prompt(self._require_task())

    def observation(self, feedback: str = "") -> dict[str, Any]:
        task = self._require_task()
        return {
            "episode_id": self.episode_id,
            "task_id": task.task_id,
            "variant": task.variant,
            "target_smiles": task.target_smiles,
            "max_depth": task.max_depth,
            "min_routes": task.min_routes,
            "max_routes": task.max_routes,
            "constraints": task.constraints.to_dict(),
            "stock_id": task.stock_id,
            "prompt": self.prompt(),
            "available_tools": list(self.tool_names),
            "tool_calls_used": self.tool_calls,
            "tool_calls_remaining": max(0, self.max_tool_calls - self.tool_calls),
            "done": self.done,
            "feedback": feedback,
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

    def stock_retrieve(self, query: str, mode: str = "auto", limit: int = 10) -> dict[str, Any]:
        """Search this task's stock (excluded building blocks are absent) with a hard 20-result cap."""
        blocked = self._consume("stock_retrieve")
        if blocked:
            return blocked
        assert self.stock_index is not None
        return self.stock_index.retrieve(
            query, mode=mode, limit=min(limit, self.max_search_results), excluded=self.excluded
        )

    def reaction_precedent_search(
        self, product_smiles: str = "", reaction_class: str = "", limit: int = 10
    ) -> dict[str, Any]:
        """Capped analogues from the train-visible reaction corpus."""
        blocked = self._consume("reaction_precedent_search")
        if blocked:
            return blocked
        return self._search(product_smiles, reaction_class, limit)

    def validate_disconnection(self, product_smiles: str, reactants: Iterable[str] | str) -> dict[str, Any]:
        """Check one proposed cut against train-visible reactions and templates."""
        blocked = self._consume("validate_disconnection")
        if blocked:
            return blocked
        reactant_values = _reactant_list(reactants)
        support = self.tool_library.support(product_smiles, reactant_values, exclude=self._own_reactions)
        result = {
            "supported": support.supported,
            "support": {"corpus": "known_precedent", "template": "reaction_template"}.get(support.kind, support.kind),
            "template_frequency": support.template_count,
            "stereo_consistent": support.stereo_match,
            "product": support.product,
            "reactants": list(support.reactants),
        }
        if support.error:
            result["error"] = support.error
        if support.product:
            result["reaction_class"] = classify_step(support.product, support.reactants).name
        result["note"] = "train-visible evidence only; the final verifier uses a larger frozen library"
        return result

    def reaction_class_lookup(self, product_smiles: str, reactants: Iterable[str] | str) -> dict[str, Any]:
        """Name the class of any proposed cut with the verifier's own labeller."""
        blocked = self._consume("reaction_class_lookup")
        if blocked:
            return blocked
        task = self._require_task()
        try:
            product = canonicalize_smiles(product_smiles)
            reactant_values = canonicalize_components(_reactant_list(reactants))
        except ChemistryError as exc:
            return {"error": str(exc)}
        label = classify_step(product, reactant_values)
        return {
            **label.to_dict(),
            "forbidden_in_this_task": label.name in task.constraints.forbidden_classes,
            "constrainable_classes": list(CONSTRAINABLE_CLASSES),
        }

    def reaction_conditions_search(
        self, product_smiles: str = "", reactants: Iterable[str] | str = (), reaction_class: str = "", limit: int = 5
    ) -> dict[str, Any]:
        """Reagent sets reported for train-visible analogues of a cut or product."""
        blocked = self._consume("reaction_conditions_search")
        if blocked:
            return blocked
        if not reaction_class and product_smiles and reactants:
            try:
                reaction_class = classify_step(
                    canonicalize_smiles(product_smiles), canonicalize_components(_reactant_list(reactants))
                ).name
            except ChemistryError:
                reaction_class = ""
        result = self._search(product_smiles, reaction_class if reaction_class != "other" else "", limit)
        return {
            "reaction_class": reaction_class or None,
            "conditions": [
                {
                    "analogue_product": row["product_smiles"],
                    "reagents": row["reagents"],
                    "similarity": row["similarity"],
                }
                for row in result.get("results", [])
            ],
            "source": "train-visible analogues",
        }

    def search_literature(self, product_smiles: str = "", reaction_class: str = "", limit: int = 5) -> dict[str, Any]:
        """Patent identifiers attached to train-visible analogues; no live web access."""
        blocked = self._consume("search_literature")
        if blocked:
            return blocked
        result = self._search(product_smiles, reaction_class, limit)
        citations = [
            {"patent": patent, "product_smiles": row["product_smiles"], "reaction_class": row["reaction_class"]}
            for row in result.get("results", [])
            for patent in row["patents"]
        ]
        return {
            "results": citations[:limit],
            "returned": min(limit, len(citations)),
            "note": "frozen metadata only; live web search is disabled during rollouts",
        }

    def emit_routes(self, submission: dict[str, Any] | str) -> dict[str, Any]:
        """Submit the final route trees; this is terminal."""
        task = self._require_task()
        if self.done:
            return {"error": "episode is already complete", "done": True, "score": self.final_score}
        result = self.verifier.score_submission(task, submission, self.stock)
        self.done = True
        self.final_score = result.to_dict()
        return {"done": True, "score": self.final_score, "tool_calls_used": self.tool_calls}

    def _search(self, product_smiles: str, reaction_class: str, limit: int) -> dict[str, Any]:
        try:
            return self.precedent_index.search(
                task=self._require_task(),
                product_smiles=product_smiles or None,
                reaction_class=reaction_class or None,
                limit=min(limit, self.max_search_results),
            )
        except Exception as exc:
            return {"results": [], "error": str(exc)}

    def _consume(self, tool_name: str) -> dict[str, Any] | None:
        self._require_task()
        if self.done:
            return {"error": "episode is complete", "done": True}
        if tool_name not in self.tool_names:
            return {"error": f"{tool_name} is not available in the {self.toolset!r} toolset", "tool": tool_name}
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


def _reactant_list(reactants: Iterable[str] | str) -> list[str]:
    if isinstance(reactants, str):
        return [part for part in reactants.split(".") if part]
    return [str(item) for item in reactants]
