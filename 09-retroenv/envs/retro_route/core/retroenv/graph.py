"""Renderable route-tree schema, tolerant parsing, and route/tree conversion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

from .chemistry import canonicalize_components, canonicalize_smiles
from .models import ReactionStep, ReferenceRoute

GRAPH_SCHEMA_VERSION = "retro-route-graph-v1"


@dataclass
class ParsedTree:
    """One submitted molecule/reaction tree and its parse diagnostics."""

    root_smiles: str | None = None
    steps: list[ReactionStep] = field(default_factory=list)
    molecules_seen: int = 0
    valid_molecules: int = 0
    terminal_claims: list[tuple[str, bool]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Missing or malformed reaction metadata: costs structure points, never invalidates chemistry.
    warnings: list[str] = field(default_factory=list)
    # Longest linear sequence: reactions on the longest root-to-leaf path.
    depth: int = 0

    @property
    def graph_valid(self) -> bool:
        return not self.errors and self.root_smiles is not None and bool(self.steps)

    @property
    def molecule_validity(self) -> float:
        return self.valid_molecules / self.molecules_seen if self.molecules_seen else 0.0

    @property
    def first_cut(self) -> tuple[str, ...]:
        if not self.steps:
            return ()
        root = self.root_smiles
        step = next((item for item in self.steps if item.product == root), self.steps[0])
        return tuple(sorted(step.reactants))


@dataclass
class ParsedSubmission:
    """A route-set payload; parsing never raises so partial reward remains possible."""

    parse_valid: bool
    trees: list[ParsedTree] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw: Any = None


def parse_submission(value: Any) -> ParsedSubmission:
    """Parse ``{"routes": [mol-tree, ...]}`` or one bare molecule tree.

    Invalid chemistry and invalid graph edges are accumulated as diagnostics
    instead of raising. This lets the reward distinguish parse failures from a
    mostly-correct tree with one bad molecule.
    """

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            return ParsedSubmission(False, errors=[f"invalid JSON: {exc}"], raw=value)
    if not isinstance(value, dict):
        return ParsedSubmission(False, errors=["submission must be a JSON object"], raw=value)

    if value.get("type") == "mol":
        routes = [value]
    else:
        routes = value.get("routes")
    if not isinstance(routes, list):
        return ParsedSubmission(False, errors=["submission.routes must be a list"], raw=value)

    parsed = ParsedSubmission(True, raw=value)
    if not routes:
        parsed.errors.append("submission contains no route trees")
    for index, tree in enumerate(routes):
        result = ParsedTree()
        _parse_molecule_node(tree, result, path=f"routes[{index}]", ancestry=())
        parsed.trees.append(result)
    return parsed


def _parse_molecule_node(
    node: Any,
    result: ParsedTree,
    *,
    path: str,
    ancestry: tuple[str, ...],
) -> str | None:
    if not isinstance(node, dict):
        result.errors.append(f"{path}: molecule node must be an object")
        return None
    if node.get("type") != "mol":
        result.errors.append(f"{path}: expected type='mol'")

    result.molecules_seen += 1
    raw_smiles = node.get("smiles")
    try:
        smiles = canonicalize_smiles(str(raw_smiles or ""))
        result.valid_molecules += 1
    except Exception as exc:
        result.errors.append(f"{path}.smiles: {exc}")
        return None
    if result.root_smiles is None:
        result.root_smiles = smiles
    if smiles in ancestry:
        result.errors.append(f"{path}: route cycle through {smiles}")
        return smiles

    if "children" not in node:
        result.errors.append(f"{path}.children is required")
    children = node.get("children", [])
    if not isinstance(children, list):
        result.errors.append(f"{path}.children must be a list")
        children = []
    if not children:
        claim = node.get("in_stock")
        if not isinstance(claim, bool):
            result.errors.append(f"{path}.in_stock must be boolean for a leaf")
            claim = False
        result.terminal_claims.append((smiles, bool(claim)))
        return smiles
    if node.get("in_stock") is not False:
        result.errors.append(f"{path}.in_stock must be false for an expanded molecule")
    if len(children) != 1:
        result.errors.append(f"{path}: a molecule must have zero or one reaction child")

    reaction = children[0] if children else None
    if not isinstance(reaction, dict) or reaction.get("type") != "reaction":
        result.errors.append(f"{path}.children[0]: expected type='reaction'")
        return smiles
    if reaction.get("is_reaction") is not True:
        result.warnings.append(f"{path}.children[0].is_reaction must be true")
    metadata = reaction.get("metadata", {})
    if not isinstance(metadata, dict):
        result.warnings.append(f"{path}.children[0].metadata must be an object")
        metadata = {}
    for required in ("explanation", "confidence", "literature", "precursor_roles"):
        if required not in metadata:
            result.warnings.append(f"{path}.children[0].metadata.{required} is required")
    if not (metadata.get("reaction_class") or metadata.get("classification")):
        result.warnings.append(f"{path}.children[0].metadata.reaction_class is required")
    confidence = metadata.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        result.warnings.append(f"{path}.children[0].metadata.confidence must be between 0 and 1")
    if not isinstance(metadata.get("precursor_roles"), dict):
        result.warnings.append(f"{path}.children[0].metadata.precursor_roles must be an object")
    if "children" not in reaction:
        result.errors.append(f"{path}.children[0].children is required")
    reaction_children = reaction.get("children", [])
    if not isinstance(reaction_children, list) or not reaction_children:
        result.errors.append(f"{path}.children[0]: reaction needs molecule children")
        return smiles

    reactants: list[str] = []
    for index, child in enumerate(reaction_children):
        child_smiles = _parse_molecule_node(
            child,
            result,
            path=f"{path}.children[0].children[{index}]",
            ancestry=(*ancestry, smiles),
        )
        if child_smiles:
            reactants.append(child_smiles)
    if reactants:
        try:
            canonical_reactants = canonicalize_components(reactants)
        except Exception as exc:  # pragma: no cover - child parsing already catches this
            result.errors.append(f"{path}: invalid reactants: {exc}")
        else:
            literature = metadata.get("literature") or []
            if not isinstance(literature, list):
                result.warnings.append(f"{path}: metadata.literature must be a list")
                literature = []
            conditions = metadata.get("conditions") or []
            if isinstance(conditions, dict):
                conditions = [conditions]
            if not isinstance(conditions, list):
                conditions = []
            result.steps.append(
                ReactionStep(
                    product=smiles,
                    reactants=canonical_reactants,
                    reaction_class=_text(metadata.get("reaction_class") or metadata.get("classification")),
                    conditions=tuple(item for item in conditions if isinstance(item, dict)),
                    literature=tuple(item for item in literature if isinstance(item, dict)),
                )
            )
            result.depth = max(result.depth, len(ancestry) + 1)
    return smiles


def route_to_graph(
    target_smiles: str,
    route: ReferenceRoute | Iterable[ReactionStep],
    stock: Iterable[str],
    *,
    source: str = "baseline",
    confidence: float = 1.0,
) -> dict[str, Any]:
    """Convert a flat, connected reference route into the renderable tree schema."""

    steps = tuple(route.steps if isinstance(route, ReferenceRoute) else route)
    by_product = {canonicalize_smiles(step.product): step for step in steps}
    stock_set = (
        _canonicalize_frozen_stock(stock)
        if isinstance(stock, frozenset)
        else frozenset(canonicalize_smiles(item) for item in stock)
    )

    def build(smiles: str, ancestry: tuple[str, ...]) -> dict[str, Any]:
        canonical = canonicalize_smiles(smiles)
        if canonical in ancestry:
            raise ValueError(f"route contains a cycle through {canonical}")
        step = by_product.get(canonical)
        if step is None:
            return {
                "type": "mol",
                "smiles": canonical,
                "in_stock": canonical in stock_set,
                "children": [],
            }
        metadata: dict[str, Any] = {
            "source": source,
            "explanation": (
                f"Dataset-supported disconnection {step.reaction_id}."
                if step.reaction_id
                else "Dataset-supported disconnection."
            ),
            "reaction_class": step.reaction_class or "unknown",
            "classification": step.reaction_class or "unknown",
            "confidence": confidence,
            "policy_probability": confidence,
            "literature": list(step.literature),
            "conditions": list(step.conditions),
            "precursor_roles": {reactant: "precursor" for reactant in step.reactants},
        }
        return {
            "type": "mol",
            "smiles": canonical,
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "is_reaction": True,
                    "metadata": metadata,
                    "children": [build(reactant, (*ancestry, canonical)) for reactant in step.reactants],
                }
            ],
        }

    return build(target_smiles, ())


def routes_to_submission(
    target_smiles: str,
    routes: Iterable[ReferenceRoute],
    stock: Iterable[str],
    *,
    source: str = "baseline",
) -> dict[str, Any]:
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "routes": [route_to_graph(target_smiles, route, stock, source=source) for route in routes],
    }


def _text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


@lru_cache(maxsize=8)
def _canonicalize_frozen_stock(stock: frozenset[str]) -> frozenset[str]:
    return frozenset(canonicalize_smiles(item) for item in stock)
