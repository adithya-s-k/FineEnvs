"""Import external attribution joins only with explicit trace/step mappings and label provenance.

Input is an export manifest joining a native/converted ATIF trace with the source's annotations.
This deliberately does not guess offsets or pretend annotation-only rows contain a whole trajectory.
"""

from __future__ import annotations

import copy

from .common import DataError, make_record

MAPPING = {
    "Plan Adherence Failure": "requirement_violation",
    "Intent–Plan Misalignment": "requirement_violation",
    "Invalid Invocation": "tool_misuse",
    "Invention of New Information": "information_error",
    "Misinterpretation of Tool Output": "information_error",
    "System Failure": "external_cause",
    "Guardrails Triggered": "external_cause",
    "Intent Not Supported": "external_cause",
    "1.1": "requirement_violation",
    "1.2": "requirement_violation",
    "2.3": "requirement_violation",
    "2.5": "information_error",
    "3.1": "false_completion",
    "3.2": "false_completion",
    "3.3": "false_completion",
    "1.3": "process_breakdown",
    "1.4": "process_breakdown",
    "1.5": "process_breakdown",
    "2.1": "process_breakdown",
    "2.2": "communication_failure",
    "2.4": "communication_failure",
    "Incorrect Problem ID": "requirement_violation",
    "Instruction Non-compliance": "requirement_violation",
    "Formatting Errors": "requirement_violation",
    "Goal Deviation": "requirement_violation",
    "Tool Selection Errors": "tool_misuse",
    "Language-only Hallucination": "information_error",
    "Tool-related Hallucination": "information_error",
    "Poor Information Retrieval": "information_error",
    "Tool Output Misinterpretation": "information_error",
    "Context Handling Failures": "process_breakdown",
    "Resource Abuse": "process_breakdown",
    "Task Orchestration Errors": "process_breakdown",
    "Incorrect Tool Definition": "external_cause",
    "Environment Setup Errors": "external_cause",
    "Rate Limiting, Auth, Service Errors": "external_cause",
    "Resource Exhaustion": "external_cause",
}


def convert(row: dict, *, dataset: str, revision: str, license: str) -> dict:
    if dataset not in {
        "microsoft/AgentRx",
        "PatronusAI/TRAIL",
        "Kevin355/Who_and_When",
        "mcemri/MAST-Data",
    }:
        raise DataError("Unsupported attribution source")
    trajectory = copy.deepcopy(row.get("trajectory"))
    if not isinstance(trajectory, dict) or trajectory.get("schema_version") != "ATIF-v1.7":
        raise DataError("Join the source annotations to a validated ATIF-v1.7 trajectory first")
    mapping = row.get("step_map")
    if not isinstance(mapping, dict):
        raise DataError("Explicit original-step to ATIF-step mapping is required")
    annotations = []
    for failure in row.get("failures", []):
        source_category = failure.get("source_label", failure.get("failure_category"))
        category = failure.get("mapped_category", MAPPING.get(source_category))
        if category is None:
            raise DataError(f"Category needs case-by-case adjudication: {source_category}")
        if "mapped_category" in failure:
            if (
                not isinstance(failure.get("adjudication_reason"), str)
                or not failure["adjudication_reason"].strip()
            ):
                raise DataError("Manual category mapping requires an adjudication_reason")
            trajectory.setdefault("extra", {}).setdefault("taxonomy_adjudications", []).append(
                {
                    "source_label": source_category,
                    "mapped_category": category,
                    "reason": failure["adjudication_reason"],
                    "original_step": failure.get("step_number"),
                }
            )
        original_step = str(failure.get("step_number"))
        if original_step not in mapping:
            raise DataError("Failure step has no explicit normalization mapping")
        annotations.append(
            {
                "category": category,
                "step_id": mapping[original_step],
                "critical": failure.get("critical", False),
                "source_label": source_category,
            }
        )
    human = (
        dataset != "mcemri/MAST-Data"
        and row.get("human_confirmed") is True
        and bool(row.get("reviewer"))
    )
    failed = (
        row.get("verified_failed") if human and type(row.get("verified_failed")) is bool else None
    )
    result = make_record(
        source={
            "dataset": dataset,
            "revision": revision,
            "license": license,
            "original_id": str(row["original_id"]),
            "source_split": row.get("source_split", "unknown"),
            "original_task_family_id": row["task_family_id"],
            "canonical_task_family_id": row.get("canonical_task_family_id"),
        },
        task=row["task"],
        trajectory=trajectory,
        family=row.get("canonical_task_family_id") or f"{dataset}:{row['task_family_id']}",
        split="test" if dataset != "mcemri/MAST-Data" else "train",
        failed=failed,
        label_source="expert",
        checker="published_human_annotation" if human else None,
        reviewer=row.get("reviewer") if human else None,
        failures=annotations,
    )
    result["source"]["source_label"] = row.get("source_label")
    return result
