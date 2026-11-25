"""The v1 reward depends only on a valid probability and a trusted binary outcome."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

CATEGORIES = (
    "requirement_violation",
    "tool_misuse",
    "information_error",
    "false_completion",
    "process_breakdown",
    "communication_failure",
    "external_cause",
    "unsafe_or_unauthorized_action",
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class Assessment:
    p_failed: float | None
    critical_step: int | None = None
    category: str | None = None
    evidence_steps: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()


def parse_action(action: str | dict, step_ids: set[int]) -> Assessment:
    try:
        if isinstance(action, str):
            if len(action) > 65536:
                raise ValueError("Output too long")
            action = json.loads(
                action,
                object_pairs_hook=_unique_object,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        if not isinstance(action, dict):
            raise ValueError("Expected one JSON object")
        probability = action.get("p_failed")
        if (
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("p_failed must be a finite number in [0, 1]")
    except (ValueError, TypeError, OverflowError, RecursionError):
        return Assessment(None, warnings=("invalid_required_output",))
    warnings = []
    critical = action.get("critical_step")
    if critical is not None and (type(critical) is not int or critical not in step_ids):
        warnings.append("invalid_critical_step")
        critical = None
    category = action.get("category")
    if category is not None and (not isinstance(category, str) or category not in CATEGORIES):
        warnings.append("invalid_category")
        category = None
    evidence = action.get("evidence_steps", [])
    if (
        not isinstance(evidence, list)
        or len(evidence) > 3
        or any(type(v) is not int or v not in step_ids for v in evidence)
        or len(set(evidence)) != len(evidence)
    ):
        warnings.append("invalid_evidence_steps")
        evidence = []
    return Assessment(float(probability), critical, category, tuple(evidence), tuple(warnings))


def score_probability(assessment: Assessment, failed: bool) -> float:
    if type(failed) is not bool:
        raise ValueError("Cannot score an unlabelled outcome")
    return 0.0 if assessment.p_failed is None else 1.0 - (assessment.p_failed - int(failed)) ** 2
