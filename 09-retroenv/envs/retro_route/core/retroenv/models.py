"""Dependency-light domain models shared by the pipeline and environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TASK_SCHEMA = "retro-task-v2"
VARIANTS = ("standard", "max_depth", "restricted_stock", "forbidden_class", "diversity")
ROUTE_KINDS = ("patent", "witness")


@dataclass(frozen=True)
class ReactionStep:
    product: str
    reactants: tuple[str, ...]
    reaction_class: str | None = None
    reaction_id: str | None = None
    conditions: tuple[dict[str, Any], ...] = ()
    literature: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReactionStep":
        if not isinstance(value, dict):
            raise TypeError("each route step must be an object")
        reactants = value.get("reactants")
        if isinstance(reactants, str):
            reactants = [part for part in reactants.split(".") if part]
        if not isinstance(reactants, (list, tuple)):
            raise TypeError("step.reactants must be a list or dot-separated string")
        return cls(
            product=str(value.get("product", "")).strip(),
            reactants=tuple(str(item).strip() for item in reactants if str(item).strip()),
            reaction_class=_optional_text(value.get("reaction_class")),
            reaction_id=_optional_text(value.get("reaction_id")),
            conditions=tuple(item for item in (value.get("conditions") or ()) if isinstance(item, dict)),
            literature=tuple(item for item in (value.get("literature") or ()) if isinstance(item, dict)),
        )

    def to_dict(self, *, include_evidence: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {"product": self.product, "reactants": list(self.reactants)}
        if self.reaction_class:
            value["reaction_class"] = self.reaction_class
        if include_evidence:
            if self.reaction_id:
                value["reaction_id"] = self.reaction_id
            value["conditions"] = list(self.conditions)
            value["literature"] = list(self.literature)
        return value


@dataclass(frozen=True)
class ReferenceRoute:
    """A hidden known route: ``patent`` routes come from one filing, ``witness`` routes are
    stock-closed recombinations of corpus reactions that prove a constrained task is solvable."""

    route_id: str
    steps: tuple[ReactionStep, ...]
    kind: str = "patent"
    source: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceRoute":
        kind = str(value.get("kind", "patent"))
        if kind not in ROUTE_KINDS:
            raise ValueError(f"unknown reference route kind {kind!r}")
        return cls(
            route_id=str(value.get("route_id", "reference")),
            steps=tuple(ReactionStep.from_dict(step) for step in value.get("steps", [])),
            kind=kind,
            source=tuple(value.get("source") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "kind": self.kind,
            "steps": [step.to_dict() for step in self.steps],
            "source": list(self.source),
        }


@dataclass(frozen=True)
class TaskConstraints:
    forbidden_classes: tuple[str, ...] = ()
    excluded_stock: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "TaskConstraints":
        value = value or {}
        return cls(
            forbidden_classes=tuple(str(item) for item in value.get("forbidden_classes") or ()),
            excluded_stock=tuple(str(item) for item in value.get("excluded_stock") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"forbidden_classes": list(self.forbidden_classes), "excluded_stock": list(self.excluded_stock)}


@dataclass(frozen=True)
class RetroTask:
    task_id: str
    target_smiles: str
    stock_id: str
    split: str
    max_depth: int
    reference_routes: tuple[ReferenceRoute, ...]
    parent_id: str = ""
    variant: str = "standard"
    min_routes: int = 1
    max_routes: int = 5
    constraints: TaskConstraints = field(default_factory=TaskConstraints)
    difficulty: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TASK_SCHEMA

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetroTask":
        task = cls(
            task_id=str(value["task_id"]),
            parent_id=str(value.get("parent_id") or value["task_id"]),
            variant=str(value.get("variant", "standard")),
            target_smiles=str(value["target_smiles"]),
            stock_id=str(value["stock_id"]),
            split=str(value.get("split", "train")),
            max_depth=int(value["max_depth"]),
            min_routes=int(value.get("min_routes", 1)),
            max_routes=int(value.get("max_routes", 5)),
            constraints=TaskConstraints.from_dict(value.get("constraints")),
            reference_routes=tuple(ReferenceRoute.from_dict(route) for route in value.get("reference_routes", [])),
            difficulty=dict(value.get("difficulty") or {}),
            schema_version=str(value.get("schema_version", TASK_SCHEMA)),
        )
        if task.variant not in VARIANTS:
            raise ValueError(f"unknown task variant {task.variant!r}")
        if task.max_depth < 1:
            raise ValueError("max_depth must be positive")
        if not 1 <= task.min_routes <= task.max_routes <= 5:
            raise ValueError("route count bounds must satisfy 1 <= min_routes <= max_routes <= 5")
        if not task.reference_routes:
            raise ValueError(f"task {task.task_id!r} has no private reference routes")
        return task

    def to_dict(self, *, include_hidden: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "parent_id": self.parent_id,
            "variant": self.variant,
            "split": self.split,
            "target_smiles": self.target_smiles,
            "stock_id": self.stock_id,
            "max_depth": self.max_depth,
            "min_routes": self.min_routes,
            "max_routes": self.max_routes,
            "constraints": self.constraints.to_dict(),
        }
        if include_hidden:
            value["difficulty"] = self.difficulty
            value["reference_routes"] = [route.to_dict() for route in self.reference_routes]
        return value


@dataclass(frozen=True)
class ScoreResult:
    reward: float
    valid: bool
    verification_tier: str
    hard_failures: tuple[str, ...]
    components: dict[str, float]
    metrics: dict[str, Any]
    step_results: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "valid": self.valid,
            "verification_tier": self.verification_tier,
            "hard_failures": list(self.hard_failures),
            "components": self.components,
            "metrics": self.metrics,
            "step_results": list(self.step_results),
        }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
