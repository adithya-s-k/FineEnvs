"""Outcome cohorts and a nondegenerate baseline comparison, separate from release approval."""

from __future__ import annotations

from collections import defaultdict

from .metrics import binary_metrics
from .records import (
    HELDOUT_DATASETS,
    heldout_source,
    scoring_exclusion_reason,
    validate_collection,
)
from .shortcuts import compare_write_baseline

EXPECTED_GROUPS = {"tau2:retail", *HELDOUT_DATASETS.values()}
OUTCOME_GROUPS = {"tau2:retail", "microsoft/AgentRx"}
DIAGNOSTIC_GROUPS = EXPECTED_GROUPS - OUTCOME_GROUPS


def retail_task_key(row: dict) -> str | None:
    """Return an explicitly supplied canonical retail task family."""
    family = row["group"]["task_family_id"]
    return (
        family
        if family.startswith("tau2:retail:") and family.removeprefix("tau2:retail:")
        else None
    )


def matched_retail_pairs(
    rows: list[dict], cases: list[int], controls: list[int]
) -> tuple[list[int], list[int], list[dict]]:
    by_case, by_control = defaultdict(list), defaultdict(list)
    for indexes, target in ((cases, by_case), (controls, by_control)):
        for i in indexes:
            family = retail_task_key(rows[i])
            if family:
                target[family].append(i)
    selected_cases, selected_controls, tasks = [], [], []
    for family in sorted(by_case.keys() & by_control.keys()):
        # Equal class counts within each task make task identity alone non-discriminative.
        # Selection is deterministic and independent of model scores and test thresholds.
        case_ids = sorted(by_case[family], key=lambda i: rows[i]["record_id"])
        control_ids = sorted(by_control[family], key=lambda i: rows[i]["record_id"])
        n = min(len(case_ids), len(control_ids))
        selected_cases.extend(case_ids[:n])
        selected_controls.extend(control_ids[:n])
        tasks.append(
            {
                "task_family_id": family,
                "available_failures": len(case_ids),
                "available_controls": len(control_ids),
                "selected_per_outcome": n,
            }
        )
    return selected_cases, selected_controls, tasks


def evaluate_gate(
    rows: list[dict],
    scores: dict[str, list[float]],
    thresholds: dict[str, float],
    *,
    training_audit: dict | None = None,
    validated: bool = False,
) -> dict:
    if not validated:
        validate_collection(rows)
    if any(len(values) != len(rows) for values in scores.values()) or set(scores) != set(
        thresholds
    ):
        raise ValueError("Gate predictions/thresholds must align with test records")
    included, native, indexes, cohorts = _cohorts(rows)
    coverage = all(cohorts[g]["gate_evaluable"] for g in OUTCOME_GROUPS)
    measured, candidates = _measure_baselines(rows, scores, thresholds, indexes, cohorts, coverage)
    task_only = measured.get("task_only_tfidf")
    task_shortcut = bool(task_only) and any(task_only[g]["meets_minimums"] for g in OUTCOME_GROUPS)
    write_comparison, qualifying, training_blocked, evidence_inconclusive = _write_comparisons(
        rows, scores, indexes, training_audit, candidates
    )
    if task_shortcut or training_blocked:
        qualifying = []
    return {
        "heldout_records": len(included),
        "outcome_comparison_groups": sorted(OUTCOME_GROUPS),
        "diagnostic_groups": sorted(DIAGNOSTIC_GROUPS),
        "comparison_status": _comparison_status(
            coverage, task_shortcut, training_blocked, evidence_inconclusive, qualifying
        ),
        "training_shortcut_audit": training_audit
        or {"status": "not_supplied", "release_allowed": False},
        "write_baseline_comparison": write_comparison,
        "missing_heldout_sources": sorted(g for g in EXPECTED_GROUPS if not native[g]),
        "retail_has_both_outcomes": {rows[i]["outcome"]["failed"] for i in native["tau2:retail"]}
        == {False, True},
        "cohorts": cohorts,
        "heldout_metrics": measured,
        "qualifying_baselines": qualifying,
        "candidate_baselines_meeting_thresholds": candidates,
        "baseline_meets_minimums_all_outcome_cohorts": bool(qualifying),
        "task_only_audit": {
            "available": task_only is not None,
            "shortcut_detected": task_shortcut,
            "metrics": task_only,
        },
    }


def _comparison_status(
    coverage, task_shortcut, training_blocked, evidence_inconclusive, qualifying
):
    if not coverage:
        return "inconclusive"
    if task_shortcut:
        return "inconclusive_task_shortcut"
    if training_blocked:
        return "inconclusive_training_shortcut"
    if evidence_inconclusive:
        return "inconclusive_write_baseline"
    return "baseline_meets_minimums" if qualifying else "baselines_below_minimums"


def _cohorts(rows):
    included = [
        i
        for i, r in enumerate(rows)
        if r["split"] == "test" and heldout_source(r) and scoring_exclusion_reason(r) is None
    ]
    native = {
        group: [i for i in included if heldout_source(rows[i]) == group]
        for group in EXPECTED_GROUPS
    }

    def verified_success(i):
        outcome = rows[i]["outcome"]
        checker = outcome["checker"]
        db_state = (
            checker == "db_state" or isinstance(checker, dict) and checker.get("kind") == "db_state"
        )
        return (
            outcome["failed"] is False
            and outcome["label_source"] == "verified_outcome"
            and db_state
        )

    controls = [i for i in native["tau2:retail"] if verified_success(i)]
    retail_cases = [
        i
        for i in native["microsoft/AgentRx"]
        if rows[i]["source"].get("domain") == "retail"
        or str(rows[i]["source"].get("source_split") or "").casefold() in {"tau_retail", "retail"}
    ]
    failure_cases = [i for i in retail_cases if rows[i]["outcome"]["failed"] is True]
    paired_cases, paired_controls, matched_tasks = matched_retail_pairs(
        rows, failure_cases, controls
    )
    indexes = dict(native)
    indexes["microsoft/AgentRx"] = paired_cases + paired_controls
    cohorts = _cohort_counts(
        rows, indexes, native, retail_cases, controls, matched_tasks, paired_cases, paired_controls
    )
    cohorts["microsoft/AgentRx"]["task_matching"] = {
        "matched_tasks": matched_tasks,
        "eligible_retail_failures": len(failure_cases),
        "available_verified_success_controls": len(controls),
        "unused_verified_success_controls": len(controls) - len(paired_controls),
        "unselected_retail_failures": len(failure_cases) - len(paired_cases),
    }
    return included, native, indexes, cohorts


def _measure_baselines(rows, scores, thresholds, indexes, cohorts, coverage):
    measured, candidates = {}, []
    for name, probabilities in scores.items():
        measured[name] = {}
        for group in sorted(EXPECTED_GROUPS):
            ii = indexes[group]
            metric = binary_metrics(
                [rows[i]["outcome"]["failed"] for i in ii],
                [probabilities[i] for i in ii],
                thresholds[name],
            )
            metric["meets_minimums"] = (
                (
                    cohorts[group]["gate_evaluable"]
                    and all(metric[key] >= 0.8 for key in ("precision", "recall", "specificity"))
                    and metric["auroc"] > 0.5
                )
                if group in OUTCOME_GROUPS
                else None
            )
            measured[name][group] = metric
        if (
            name != "task_only_tfidf"
            and coverage
            and all(measured[name][g]["meets_minimums"] for g in OUTCOME_GROUPS)
        ):
            candidates.append(name)
    return measured, candidates


def _write_comparisons(rows, scores, indexes, training_audit, candidates):
    qualifying = candidates
    write_comparison = {}
    if training_audit and training_audit.get("comparison_required") is True:
        _write_comparison_metrics(rows, scores, indexes, training_audit, write_comparison)
        qualifying = [
            name
            for name in qualifying
            if all(write_comparison[name][group]["passes"] for group in OUTCOME_GROUPS)
        ]
        training_blocked = training_audit.get("comparison_ready") is not True
        evidence_inconclusive = bool(candidates) and not qualifying
    else:
        training_blocked = (
            training_audit is None or training_audit.get("release_allowed") is not True
        )
        evidence_inconclusive = False
    return write_comparison, qualifying, training_blocked, evidence_inconclusive


def _cohort_counts(
    rows, indexes, native, retail_cases, controls, matched_tasks, paired_cases, paired_controls
):
    cohorts = {}
    for group in sorted(EXPECTED_GROUPS):
        labels = {rows[i]["outcome"]["failed"] for i in indexes[group]}
        reason = _cohort_reason(group, labels, native, retail_cases, controls, matched_tasks)
        cohorts[group] = {
            "gate_evaluable": reason is None,
            "reason": reason,
            "required_for_outcome_comparison": group in OUTCOME_GROUPS,
            "role": "outcome_comparison" if reason is None else "diagnostic_only",
            "native_records": len(native[group]),
            "outcome_records": len(indexes[group]),
            "positive_records": sum(rows[i]["outcome"]["failed"] for i in indexes[group]),
            "negative_records": sum(not rows[i]["outcome"]["failed"] for i in indexes[group]),
            "success_controls": len(paired_controls) if group == "microsoft/AgentRx" else 0,
            "unpaired_native_records": len(native[group]) - len(paired_cases)
            if group == "microsoft/AgentRx"
            else 0,
        }
    return cohorts


def _write_comparison_metrics(rows, scores, indexes, training_audit, write_comparison):
    for name, probabilities in scores.items():
        write_comparison[name] = {}
        for group in sorted(OUTCOME_GROUPS):
            ii = indexes[group]
            write_comparison[name][group] = compare_write_baseline(
                [rows[i] for i in ii],
                [probabilities[i] for i in ii],
                training_audit.get("write_baseline"),
                min_pairs=training_audit.get("minimum_pooled_pairs", 10),
                confidence=training_audit.get("confidence", 0.95),
                min_task_clusters=training_audit.get("minimum_task_clusters", 8),
            )


def _cohort_reason(group, labels, native, retail_cases, controls, matched_tasks):
    reason = None
    if group in DIAGNOSTIC_GROUPS:
        if not native[group]:
            reason = "missing_diagnostic_source"
        elif labels != {False, True}:
            reason = "single_outcome_diagnostic"
        else:
            reason = "diagnostic_source"
    elif not native[group]:
        reason = "missing_source"
    elif group == "microsoft/AgentRx" and not retail_cases:
        reason = "missing_agentrx_retail_cases"
    elif group == "microsoft/AgentRx" and not controls:
        reason = "missing_verified_retail_success_controls"
    elif group == "microsoft/AgentRx" and not matched_tasks:
        reason = "no_matched_task_families"
    elif labels != {False, True}:
        reason = "single_outcome_cohort"
    return reason
