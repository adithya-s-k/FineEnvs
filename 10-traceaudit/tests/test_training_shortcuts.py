import copy
import itertools
from collections import Counter

import pytest
from test_adapters import tau_row
from test_evaluation import full_cohorts
from test_reward_hacks import production
from trace_audit.core import DataError, render_record
from trace_audit.core.gate import evaluate_gate
from trace_audit.core.provenance import generator_key
from trace_audit.core.sampling import _default_pair_count, balanced_batch
from trace_audit.core.shortcuts import model_label_probe, training_shortcut_audit

from adapters.swe_rebench import convert as swe
from adapters.tau2 import convert, convert_results

"""Model-style confounds, finite pair capacity and within-task release blockers."""


def runs(
    template,
    *,
    source="source",
    task="task",
    agent="generator",
    user="simulator",
    successes=1,
    failures=1,
    write_success=True,
    write_failure=True,
):
    rows = []
    for failed, size, write in ((False, successes, write_success), (True, failures, write_failure)):
        for i in range(size):
            row = copy.deepcopy(template)
            key = f"{source}:{task}:{agent}:{user}:{failed}:{i}"
            row["record_id"] = key
            row["source"].update(
                dataset=source,
                domain="airline",
                original_id=key,
                data_kind="unit_test_case",
                agent_model=agent,
                user_model=user,
            )
            row["group"]["task_family_id"] = f"{source}:{task}"
            row["split"] = "train"
            row["outcome"] = {
                "failed": failed,
                "label_source": "expert",
                "reviewer": "unit reviewer",
                "checker": "unit evidence",
            }
            calls = [c for s in row["trajectory"]["steps"] for c in s.get("tool_calls") or []]
            for call in calls:
                call["function_name"] = "get_reservation_details"
            if write:
                calls[0]["function_name"] = "book_reservation"
            rows.append(row)
    return rows


def test_model_provenance_is_retained_conflict_checked_and_redacted():
    raw = tau_row()
    raw.update(agent_model="SecretAgentModel", user_model="SecretSimulatorModel")
    raw["simulation"]["messages"][-1]["content"] = "SecretAgentModel used SecretSimulatorModel."
    result = convert(raw, revision="a" * 40)
    assert generator_key(result) == ("SecretAgentModel", "SecretSimulatorModel")
    assert "SecretAgentModel" not in render_record(result)
    assert "SecretSimulatorModel" not in render_record(result)
    raw["simulation"]["agent_model"] = "ConflictingModel"
    with pytest.raises(DataError, match="Conflicting"):
        convert(raw, revision="a" * 40)
    native = {
        "tasks": [raw["task"]],
        "simulations": [{**tau_row()["simulation"], "task_id": "7"}],
        "info": {"agent_info": {"llm": "model-A"}, "user_info": {"llm": "model-U"}},
    }
    converted = convert_results(
        native, revision="a" * 40, domain="airline", tools=[], policy="policy"
    )
    assert generator_key(converted[0]) == ("model-A", "model-U")
    with pytest.raises(DataError, match="Conflicting"):
        convert_results(
            native,
            revision="a" * 40,
            domain="airline",
            tools=[],
            policy="policy",
            agent_model="different",
        )
    result = swe(
        {
            "instance_id": "repo-1",
            "trajectory_id": "run",
            "resolved": 1,
            "source_model": "swe-model",
            "trajectory": [{"role": "user", "content": "Fix bug"}],
        },
        revision="a" * 40,
    )
    assert generator_key(result) == ("swe-model", "not_applicable")


def test_missing_models_are_excluded_without_cross_model_fallback(rows):
    records = production(rows)
    extra = copy.deepcopy(records[0])
    extra["record_id"] = "unknown-generator"
    extra["source"]["agent_model"] = None
    batch, report = balanced_batch([*records, extra])
    assert report["excluded_records"]["missing_model_provenance"] == 1
    assert extra["record_id"] not in {r["record_id"] for r in batch}
    for row in records:
        row["source"]["agent_model"] = None
    with pytest.raises(DataError, match="task/model"):
        balanced_batch(records)


@pytest.mark.parametrize("field", ["agent_model", "user_model"])
def test_pure_outcomes_from_different_models_do_not_form_a_pair(rows, field):
    records = runs(rows[0])
    for row in records:
        row["source"][field] = "strong" if not row["outcome"]["failed"] else "weak"
    with pytest.raises(DataError, match="task/model"):
        balanced_batch(records, source_cap=1)


def test_model_label_oracle_loses_signal_after_pairing(rows):
    records = runs(rows[0], agent="strong", successes=9, failures=1)
    records += runs(rows[0], agent="weak", successes=1, failures=9)
    raw = model_label_probe(records)
    assert raw["metrics"]["auroc"] == 0.9
    batch, report = balanced_batch(records, source_cap=1)
    assert len(batch) == 4 and report["max_record_reuse"] == 1
    assert model_label_probe(batch)["metrics"]["auroc"] == 0.5
    counts = Counter((r["source"]["agent_model"], r["outcome"]["failed"]) for r in batch)
    assert counts[("strong", False)] == counts[("strong", True)] == 1
    assert counts[("weak", False)] == counts[("weak", True)] == 1


def test_rare_success_is_not_repeated_to_fill_batch(rows):
    records = runs(rows[0], task="rare", successes=1, failures=9)
    records += runs(rows[0], task="abundant", successes=9, failures=9)
    batch, report = balanced_batch(records, source_cap=1)
    assert len(batch) == 20 and report["unique_records"] == 20
    assert report["max_record_reuse"] == 1
    assert sum(r["group"]["task_family_id"].endswith(":rare") for r in batch) == 2
    with pytest.raises(DataError, match="reuse limit"):
        balanced_batch(records, 32, source_cap=1)
    batch, report = balanced_batch(records, source_cap=1, max_record_reuse=2)
    assert len(batch) == 40 and report["max_record_reuse"] == 2
    assert max(Counter(r["record_id"] for r in batch).values()) == 2


def test_default_capacity_matches_exhaustive_oracle():
    for capacities in itertools.product(range(1, 7), repeat=3):
        for cap in (0.25, 0.4, 0.5, 0.6, 1.0):
            oracle = max(
                [
                    0,
                    *(
                        p
                        for p in range(1, sum(capacities) + 1)
                        if sum(min(n, int(2 * p * cap) // 2) for n in capacities) >= p
                    ),
                ]
            )
            assert _default_pair_count(dict(enumerate(capacities)), cap) == oracle


def test_sparse_opposite_write_signals_pool_signed_without_false_block(rows):
    source = "sierra-research/tau2-bench"
    records = runs(rows[0], source=source, task="a", write_success=True, write_failure=False)
    records += runs(rows[0], source=source, task="b", write_success=False, write_failure=True)
    batch, _ = balanced_batch(records, source_cap=1)
    audit = training_shortcut_audit(batch)
    assert audit["status"] == "inconclusive" and not audit["release_allowed"]
    probe = audit["pooled_probes"]["any_write"]
    assert probe["auroc"] == 0.5
    assert len(probe["per_stratum"]) == 2
    assert all(r["metrics"]["predictive_strength_auroc"] == 1 for r in probe["per_stratum"])


def test_sparse_write_count_is_diagnostic_when_presence_is_constant(rows):
    records = runs(rows[0], source="sierra-research/tau2-bench")
    for row in records:
        if row["outcome"]["failed"]:
            for step in row["trajectory"]["steps"]:
                for call in step.get("tool_calls") or []:
                    call["function_name"] = "book_reservation"
    audit = training_shortcut_audit(records)
    assert audit["status"] == "inconclusive"
    assert audit["pooled_probes"]["any_write"]["auroc"] == 0.5
    assert audit["pooled_probes"]["write_count"]["auroc"] == 1


def test_unknown_tools_low_support_and_reused_pairs_do_not_clear_release(rows):
    records = runs(rows[0], source="sierra-research/tau2-bench")
    assert training_shortcut_audit(records)["status"] == "inconclusive"
    assert training_shortcut_audit(records * 20)["status"] == "inconclusive"
    assert (
        training_shortcut_audit(records * 20)["pooled_probes"]["any_write"]["unique_outcome_pairs"]
        == 1
    )
    for row in records:
        row["trajectory"]["steps"][1]["tool_calls"][0]["function_name"] = "unclassified"
    audit = training_shortcut_audit(records)
    assert "no_write_baseline_coverage" in audit["pending_reasons"]
    assert not audit["release_allowed"]


def test_training_probes_alone_cannot_approve_auditor_comparison(rows):
    records = runs(rows[0], source="sierra-research/tau2-bench", successes=10, failures=10)
    audit = training_shortcut_audit(records)
    assert audit["status"] == "pending_auditor_comparison" and not audit["release_allowed"]
    assert audit["comparison_ready"]


@pytest.mark.parametrize(
    "audit",
    [
        None,
        {"status": "blocked", "release_allowed": False},
        {"status": "inconclusive", "release_allowed": False},
    ],
)
def test_missing_or_predictive_training_audit_blocks_otherwise_perfect_comparison(rows, audit):
    data = full_cohorts(rows)
    perfect = [float(row["outcome"]["failed"]) for row in data]
    result = evaluate_gate(data, {"judge": perfect}, {"judge": 0.5}, training_audit=audit)
    assert result["comparison_status"] == "inconclusive_training_shortcut"
    assert result["qualifying_baselines"] == []
    assert result["candidate_baselines_meeting_thresholds"] == ["judge"]
