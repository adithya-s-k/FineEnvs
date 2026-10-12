"""Distil one (task, submission) pair into the raw facts every reward variant needs.

Parsing, RDKit and the template scan are the expensive part and are independent
of how the reward is shaped, so they run once and the result is cached to disk.
Verdicts (supported, valid, duplicate, in budget) are deliberately *not* baked
in: variants recompute them from these facts under their own support rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from retroenv.chemistry import ChemistryError, canonicalize_smiles
from retroenv.classes import classify_step
from retroenv.graph import parse_submission
from retroenv.models import RetroTask
from retroenv.reactions import ReactionLibrary, free_key
from retroenv.retrieval import cached_stock_index
from retroenv.verifier import _canonical_stock, compliant_references, route_depth, signature

from .support import StepProfile, profile_step

EVIDENCE_SCHEMA = "retro-reward-evidence-v1"


@dataclass
class RouteEvidence:
    root: str
    root_ok: bool
    depth: int
    errors: int
    warnings: int
    molecule_validity: float
    steps: list[StepProfile] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    leaves: list[dict[str, Any]] = field(default_factory=list)
    first_cut: list[str] = field(default_factory=list)
    excluded_used: list[str] = field(default_factory=list)
    # Tanimoto to the closest molecule in the task's stock, for every molecule in
    # the tree: how buyable an unresolved frontier molecule already looks.
    nearest_stock: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "root_ok": self.root_ok,
            "depth": self.depth,
            "errors": self.errors,
            "warnings": self.warnings,
            "molecule_validity": self.molecule_validity,
            "steps": [step.to_dict() for step in self.steps],
            "classes": self.classes,
            "leaves": self.leaves,
            "first_cut": self.first_cut,
            "excluded_used": self.excluded_used,
            "nearest_stock": self.nearest_stock,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RouteEvidence":
        return cls(
            root=value["root"],
            root_ok=value["root_ok"],
            depth=value["depth"],
            errors=value["errors"],
            warnings=value["warnings"],
            molecule_validity=value["molecule_validity"],
            steps=[StepProfile.from_dict(step) for step in value["steps"]],
            classes=value["classes"],
            leaves=value["leaves"],
            first_cut=value["first_cut"],
            excluded_used=value["excluded_used"],
            nearest_stock=value.get("nearest_stock", {}),
        )


@dataclass
class SubmissionEvidence:
    task_id: str
    variant: str
    max_depth: int
    min_routes: int
    max_routes: int
    min_known_depth: int
    forbidden_classes: list[str]
    parse_ok: bool
    parse_errors: int
    routes: list[RouteEvidence] = field(default_factory=list)
    reference_signatures: list[list[list[Any]]] = field(default_factory=list)
    label: str = ""
    # The run directory this came from: one model can appear under several harness
    # settings (repair vs --no-repair), which `label` alone would silently merge.
    run: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "variant": self.variant,
            "max_depth": self.max_depth,
            "min_routes": self.min_routes,
            "max_routes": self.max_routes,
            "min_known_depth": self.min_known_depth,
            "forbidden_classes": self.forbidden_classes,
            "parse_ok": self.parse_ok,
            "parse_errors": self.parse_errors,
            "routes": [route.to_dict() for route in self.routes],
            "reference_signatures": self.reference_signatures,
            "label": self.label,
            "run": self.run,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SubmissionEvidence":
        return cls(
            **{k: v for k, v in value.items() if k not in ("routes",)},
            routes=[RouteEvidence.from_dict(route) for route in value["routes"]],
        )


class EvidenceBuilder:
    """Holds the loaded release so a whole run can be distilled in one process."""

    def __init__(self, library: ReactionLibrary, store: Any):
        self.library = library
        self.store = store
        self._tasks: dict[str, RetroTask] = {}
        self._context: dict[str, tuple[frozenset[str], frozenset[str], int, list]] = {}

    def task(self, task_id: str, split: str) -> RetroTask:
        if task_id not in self._tasks:
            for candidate in self.store.tasks(split):
                self._tasks.setdefault(candidate.task_id, candidate)
        return self._tasks[task_id]

    def context(self, task: RetroTask) -> tuple[frozenset[str], frozenset[str], int, list]:
        """Effective stock, excluded blocks, shortest compliant known depth, reference signatures."""
        if task.task_id not in self._context:
            stock = _canonical_stock(self.store.stock(task.stock_id))
            excluded = _canonical_stock(task.constraints.excluded_stock)
            effective = stock - excluded
            forbidden = frozenset(task.constraints.forbidden_classes)
            references = compliant_references(task, effective, forbidden)
            depths = [route_depth(task.target_smiles, ref.steps) for ref in references]
            signatures = [signature(ref.steps) for ref in references]
            min_depth = min(depths, default=None)
            if min_depth is None:
                min_depth = int(task.difficulty.get("min_depth") or task.max_depth)
            cached_stock_index(frozenset(self.store.stock(task.stock_id)))  # fingerprints, shared after fork
            self._context[task.task_id] = (effective, excluded, min_depth, signatures)
        return self._context[task.task_id]

    def stock_index(self, task: RetroTask) -> Any:
        return cached_stock_index(frozenset(self.store.stock(task.stock_id)))

    def build(self, task: RetroTask, submission: Any, *, label: str = "", run: str = "") -> SubmissionEvidence:
        stock, excluded, min_depth, reference_signatures = self.context(task)
        target = canonicalize_smiles(task.target_smiles)
        parsed = parse_submission(submission)
        evidence = SubmissionEvidence(
            task_id=task.task_id,
            variant=task.variant,
            max_depth=task.max_depth,
            min_routes=task.min_routes,
            max_routes=task.max_routes,
            min_known_depth=min_depth,
            forbidden_classes=list(task.constraints.forbidden_classes),
            parse_ok=parsed.parse_valid,
            parse_errors=len(parsed.errors),
            reference_signatures=[_encode(sig) for sig in reference_signatures],
            label=label,
            run=run,
        )
        if not parsed.parse_valid:
            return evidence
        index = self.stock_index(task)
        for tree in parsed.trees:
            route = self._route(tree, target, stock, excluded)
            route.nearest_stock = {
                smiles: _nearest_stock(index, smiles, excluded)
                for smiles in {leaf["smiles"] for leaf in route.leaves} | {step.product for step in route.steps}
                if smiles
            }
            evidence.routes.append(route)
        return evidence

    def _route(self, tree: Any, target: str, stock: frozenset[str], excluded: frozenset[str]) -> RouteEvidence:
        route = RouteEvidence(
            root=tree.root_smiles or "",
            root_ok=tree.root_smiles == target and bool(tree.steps),
            depth=tree.depth,
            errors=len(tree.errors),
            warnings=len(tree.warnings),
            molecule_validity=tree.molecule_validity,
        )
        for step in tree.steps:
            route.steps.append(profile_step(self.library, step.product, step.reactants))
            route.classes.append(classify_step(step.product, step.reactants).name)
        for smiles, claim in tree.terminal_claims:
            route.leaves.append(
                {
                    "smiles": smiles,
                    "claimed": bool(claim),
                    "in_stock": smiles in stock,
                    "heavy_atoms": _heavy(smiles),
                }
            )
            if smiles in excluded:
                route.excluded_used.append(smiles)
        root_step = next((step for step in tree.steps if step.product == tree.root_smiles), None)
        if root_step is not None:
            try:
                route.first_cut = list(free_key(root_step.product, root_step.reactants)[1])
            except ChemistryError:
                route.first_cut = []
        return route


def _nearest_stock(index: Any, smiles: str, excluded: frozenset[str]) -> float:
    """Tanimoto to the closest buyable molecule, through the agent's own stock surface."""
    result = index.retrieve(smiles, mode="similarity", limit=1, excluded=excluded)
    hits = result.get("results") or []
    return round(float(hits[0]["similarity"]), 6) if hits else 0.0


def _encode(sig: Iterable[Any]) -> list[list[Any]]:
    return sorted([product, list(reactants)] for product, reactants in sig)


def _heavy(smiles: str) -> int:
    from .support import _heavy_atoms

    return _heavy_atoms(smiles)


def write_evidence(path: Path, records: Iterable[SubmissionEvidence]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": EVIDENCE_SCHEMA}) + "\n")
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def read_evidence(path: Path) -> list[SubmissionEvidence]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if rows and rows[0].get("schema") == EVIDENCE_SCHEMA:
        rows = rows[1:]
    return [SubmissionEvidence.from_dict(row) for row in rows]
