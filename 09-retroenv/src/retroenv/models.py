"""Dependency-light domain models shared by the pipeline and environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReactionStep:
    product: str
    reactants: tuple[str, ...]
    reaction_class: str | None = None
    reaction_id: str | None = None
    reaction_smarts: str | None = None
    mapping_status: str = "unmapped"
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
            reaction_smarts=_optional_text(value.get("reaction_smarts")),
            mapping_status=str(value.get("mapping_status") or "unmapped"),
            conditions=tuple(
                item for item in (value.get("conditions") or ()) if isinstance(item, dict)
            ),
            literature=tuple(
                item for item in (value.get("literature") or ()) if isinstance(item, dict)
            ),
        )

    def to_dict(self, *, include_evidence: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "product": self.product,
            "reactants": list(self.reactants),
        }
        if self.reaction_class:
            value["reaction_class"] = self.reaction_class
        if include_evidence:
            value.update(
                {
                    "reaction_id": self.reaction_id,
                    "reaction_smarts": self.reaction_smarts,
                    "mapping_status": self.mapping_status,
                    "conditions": list(self.conditions),
                    "literature": list(self.literature),
                }
            )
        return {key: item for key, item in value.items() if item is not None}


@dataclass(frozen=True)
class ReferenceRoute:
    route_id: str
    steps: tuple[ReactionStep, ...]
    source: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ReferenceRoute":
        return cls(
            route_id=str(value.get("route_id", "reference")),
            steps=tuple(ReactionStep.from_dict(step) for step in value.get("steps", [])),
            source=tuple(value.get("source") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "steps": [step.to_dict() for step in self.steps],
            "source": list(self.source),
        }


@dataclass(frozen=True)
class RetroTask:
    task_id: str
    mode: str
    target_smiles: str
    max_steps: int
    stock_id: str
    split: str
    reference_routes: tuple[ReferenceRoute, ...]
    min_routes: int = 1
    max_routes: int = 5
    difficulty: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "retro-task-v1"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetroTask":
        mode = str(value.get("mode", "single_step"))
        if mode not in {"single_step", "route_planning"}:
            raise ValueError(f"unsupported task mode: {mode!r}")
        task = cls(
            task_id=str(value["task_id"]),
            mode=mode,
            target_smiles=str(value["target_smiles"]),
            max_steps=int(value.get("max_steps", 1)),
            stock_id=str(value["stock_id"]),
            split=str(value.get("split", "train")),
            reference_routes=tuple(
                ReferenceRoute.from_dict(route)
                for route in value.get("reference_routes", [])
            ),
            min_routes=int(value.get("min_routes", 1)),
            max_routes=int(value.get("max_routes", 5)),
            difficulty=dict(value.get("difficulty") or {}),
            schema_version=str(value.get("schema_version", "retro-task-v1")),
        )
        if task.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 1 <= task.min_routes <= task.max_routes <= 5:
            raise ValueError("route count bounds must satisfy 1 <= min_routes <= max_routes <= 5")
        if not task.reference_routes:
            raise ValueError(f"task {task.task_id!r} has no private reference routes")
        return task

    def to_dict(self, *, include_references: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "mode": self.mode,
            "target_smiles": self.target_smiles,
            "max_steps": self.max_steps,
            "stock_id": self.stock_id,
            "min_routes": self.min_routes,
            "max_routes": self.max_routes,
            "split": self.split,
            "difficulty": self.difficulty,
        }
        if include_references:
            value["reference_routes"] = [
                route.to_dict() for route in self.reference_routes
            ]
        return value


@dataclass(frozen=True)
class StepValidation:
    valid: bool
    product: str | None
    reactants: tuple[str, ...]
    support: str
    matched_reaction_id: str | None
    atom_conservation: str
    checks: dict[str, bool]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreResult:
    reward: float
    valid: bool
    verification_tier: str
    hard_failures: tuple[str, ...]
    components: dict[str, float]
    metrics: dict[str, Any]
    step_results: tuple[StepValidation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "valid": self.valid,
            "verification_tier": self.verification_tier,
            "hard_failures": list(self.hard_failures),
            "components": self.components,
            "metrics": self.metrics,
            "step_results": [result.to_dict() for result in self.step_results],
        }


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
