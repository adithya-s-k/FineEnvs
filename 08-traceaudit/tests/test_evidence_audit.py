import copy
import json

import pytest
from test_reward_hacks import port, production
from test_training_shortcuts import runs
from trace_audit.core import DataError, TraceAuditEnv
from trace_audit.core.baselines import write_tool_call_count
from trace_audit.core.gate import evaluate_gate
from trace_audit.core.sampling import balanced_batch
from trace_audit.core.shortcuts import compare_write_baseline, training_shortcut_audit

from tools.tau2 import main as tau_main

"""Sparse noise, legitimate write evidence, and actual held-out discrimination."""


TAU = "sierra-research/tau2-bench"


def make_read_only(row):
    for step in row["trajectory"]["steps"]:
        for call in step.get("tool_calls") or []:
            call["function_name"] = "get_reservation_details"


def test_one_perfect_pair_is_noise_not_a_rejection(rows):
    records = runs(rows[0], source=TAU, write_failure=False)
    audit = training_shortcut_audit(records)
    probe = audit["pooled_probes"]["any_write"]
    assert probe["predictive_strength_auroc"] == 1
    assert probe["confidence_interval"]["lower"] is None
    assert probe["confidence_interval"]["upper"] is None
    assert probe["confidence_interval"]["status"] == "insufficient_task_clusters"
    assert probe["support_status"] == "insufficient_pairs"
    assert audit["status"] == "inconclusive"


def test_genuine_no_action_failures_create_baseline_not_rejection(rows):
    records = runs(rows[0], source=TAU, successes=12, failures=12)
    failures = [r for r in records if r["outcome"]["failed"]]
    for row in failures[:8]:
        make_read_only(row)
    audit = training_shortcut_audit(records)
    assert audit["pooled_probes"]["any_write"]["predictive_strength_auroc"] == pytest.approx(5 / 6)
    assert audit["status"] == "pending_auditor_comparison"
    assert audit["comparison_ready"]
    assert audit["write_baseline"]["buckets"]["no_write"] == {"n": 8, "p_failed": 0.9}
    assert not audit[
        "release_allowed"
    ]  # Training diagnostics alone cannot establish held-out performance.


def test_length_tool_count_and_closing_language_are_measured_on_sample(rows):
    records = []
    for i in range(12):
        members = runs(rows[0], task=str(i))
        for row in members:
            final = row["trajectory"]["steps"][-1]
            final["message"] = "All done " + "x" * 2000 if row["outcome"]["failed"] else "Pending."
            if row["outcome"]["failed"]:
                step = row["trajectory"]["steps"][1]
                call = copy.deepcopy(step["tool_calls"][0])
                call["tool_call_id"] = "additional-call"
                step["tool_calls"].append(call)
        records += members
    audit = training_shortcut_audit(records)
    for key in ("trace_length", "tool_calls", "closing_claim"):
        probe = audit["pooled_probes"][key]
        assert probe["auroc"] == 1
        assert probe["unique_outcome_pairs"] == 12
        assert probe["confidence_interval"]["task_clusters"] == 12
        assert probe["decision_role"] == "diagnostic_only"
    assert audit["model_label_probe"]["role"] == "balance_invariant_check"


def test_model_variants_and_replays_do_not_inflate_independent_task_support(rows):
    records = runs(rows[0], source=TAU, agent="a", write_failure=False)
    records += runs(rows[0], source=TAU, agent="b", write_failure=False)
    probe = training_shortcut_audit(records * 20)["pooled_probes"]["any_write"]
    assert probe["unique_outcome_pairs"] == 2
    assert probe["confidence_interval"]["task_clusters"] == 1
    assert probe["confidence_interval"]["lower"] is None


def test_model_invariant_checks_weighted_sample_while_support_deduplicates(rows):
    records = runs(rows[0], source=TAU, agent="a", successes=1, failures=9)
    records += runs(rows[0], source=TAU, agent="b", successes=9, failures=1)
    batch, _ = balanced_batch(records, source_cap=1, max_record_reuse=3)
    audit = training_shortcut_audit(batch)
    assert audit["model_label_probe"]["metrics"]["auroc"] == 0.5
    assert all(
        r["failure_rate"] == 0.5 for r in audit["model_label_probe"]["conditional_failure_rates"]
    )
    assert audit["unique_records"] < audit["sampled_records"]


def test_core_explicit_reset_cannot_access_excluded_admitted_record(rows):
    records = production(rows)
    excluded = copy.deepcopy(records[0])
    excluded["record_id"] = "single-outcome-task"
    excluded["group"]["task_family_id"] = "single-outcome-task"
    env = TraceAuditEnv([*records, excluded])
    assert excluded["record_id"] in env._records
    with pytest.raises(DataError, match="paired training sample"):
        env.reset(record_id=excluded["record_id"])
    env.reset(record_id=env._training_ids[0])
    assert env.step({"p_failed": 0.5}).reward == 0.75


def test_toolkit_and_core_share_the_same_custom_training_sample(rows):
    kit = port().TraceAuditToolkit(
        rows, allow_fixtures=True, training_size=4, source_cap=1, sampling_seed=7
    )
    batch, _ = kit._env.training_sample()
    assert len(batch) == 4
    ids = {row["record_id"] for row in batch}
    for key in ids:
        kit.reset(record_id=key)
        assert kit.submit_analysis('{"p_failed":0.5}')
    excluded = next(row["record_id"] for row in kit._env.records if row["record_id"] not in ids)
    with pytest.raises(DataError, match="paired training sample"):
        kit.reset(record_id=excluded)


def heldout_cohorts(template, *, no_write_failures=False, families=8):
    data = []
    # Eight independent families can support a strong empirical gain; repeated
    # trials are useful class support but do not count as extra task families.
    for i in range(families):
        family = f"tau2:retail:{i}"
        members = runs(template, source=TAU, task=str(i), successes=2, failures=2)
        members += runs(template, source="microsoft/AgentRx", task=str(i), successes=0, failures=2)
        if no_write_failures:
            members += runs(template, source=TAU, task=f"extra-{i}", successes=0, failures=8)
        for row in members:
            _heldout_record(row, family, no_write_failures)
            data.append(row)
    return data


def test_real_auditor_can_beat_write_baseline_and_discriminate_written_runs(rows):
    audit = training_shortcut_audit(runs(rows[0], source=TAU, successes=10, failures=10))
    data = heldout_cohorts(rows[0])
    p = [float(r["outcome"]["failed"]) for r in data]
    result = evaluate_gate(data, {"auditor": p}, {"auditor": 0.5}, training_audit=audit)
    assert result["qualifying_baselines"] == ["auditor"]
    for comparison in result["write_baseline_comparison"]["auditor"].values():
        assert comparison["passes"]
        assert comparison["brier_gain_task_cluster_interval"]["lower"] > 0
        assert comparison["written_runs"]["confidence_interval"]["lower"] > 0.5


def test_write_only_predictor_cannot_qualify_via_real_no_action_failures(rows):
    audit = training_shortcut_audit(runs(rows[0], source=TAU, successes=10, failures=10))
    data = heldout_cohorts(rows[0], no_write_failures=True)
    p = [0.9 if write_tool_call_count(r) == 0 else 0.1 for r in data]
    result = evaluate_gate(data, {"write_only": p}, {"write_only": 0.5}, training_audit=audit)
    assert result["candidate_baselines_meeting_thresholds"] == ["write_only"]
    assert result["qualifying_baselines"] == []
    assert result["comparison_status"] == "inconclusive_write_baseline"
    assert all(
        "written_discrimination_not_established" in c["reasons"]
        for c in result["write_baseline_comparison"]["write_only"].values()
    )


def test_baseline_parameters_are_not_refitted_on_heldout_labels(rows):
    baseline = training_shortcut_audit(runs(rows[0], source=TAU, successes=10, failures=10))[
        "write_baseline"
    ]
    data = runs(rows[0], source=TAU, successes=1, failures=9)
    before = copy.deepcopy(baseline)
    result = compare_write_baseline(data, [0.5] * len(data), baseline)
    assert baseline == before
    assert result["brier_gain_task_cluster_interval"]["estimate"] == 0
    assert not result["passes"]


def test_collection_plan_documents_pairing_losses_and_written_outcome_targets(
    tmp_path, monkeypatch
):
    output = tmp_path / "plan.json"
    monkeypatch.setattr("sys.argv", ["tau2.py", "plan", "--output", str(output)])
    tau_main()
    plan = json.loads(output.read_text())
    requirements = plan["collection_requirements"]
    assert requirements["unit"] == "task/agent_model/user_model"
    assert requirements["repeated_trials"]
    assert requirements["target_written_failures_per_unit"] == 10
    assert "retention fraction" in " ".join(requirements["coverage_report"])
    assert plan["status"] == "not_launched"


def _heldout_record(row, family, no_write_failures):
    row["split"] = "test"
    row["group"]["task_family_id"] = family
    row["source"].update(domain="retail", source_split="tau_retail")
    for step in row["trajectory"]["steps"]:
        for call in step.get("tool_calls") or []:
            call["function_name"] = "modify_user_address"
    if row["source"]["dataset"] == TAU:
        row["outcome"].update(
            label_source="verified_outcome",
            checker={"kind": "db_state", "task_requires_db_change": True},
            reviewer=None,
        )
    if (
        no_write_failures
        and row["outcome"]["failed"]
        and (row["source"]["dataset"] != TAU or "extra-" in row["record_id"])
    ):
        for step in row["trajectory"]["steps"]:
            for call in step.get("tool_calls") or []:
                call["function_name"] = "get_order_details"
