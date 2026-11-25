import copy
from collections import Counter, defaultdict
from importlib.util import module_from_spec, spec_from_file_location

import pytest
from conftest import ROOT
from test_adapters import tau_row
from test_evaluation import full_cohorts
from trace_audit.core import DataError, TraceAuditEnv, render_record
from trace_audit.core.baselines import write_tool_call_count
from trace_audit.core.gate import evaluate_gate
from trace_audit.core.metrics import binary_metrics
from trace_audit.core.records import scoring_exclusion_reason
from trace_audit.core.review import apply_reviews, assert_training_audits, make_queue
from trace_audit.core.sampling import balanced_batch

from adapters.tau2 import convert, db_change_task
from tools.data import admission_report

"""Adversarial checks for source/task rewards, DB no-ops and trainer integration."""


def production(rows):
    records = copy.deepcopy(
        [r for r in rows if r["split"] == "train" and r["outcome"]["failed"] is not None]
    )
    for row in records:
        row["source"]["data_kind"] = "unit_test_case"
    return records


@pytest.mark.parametrize("sources", [2, 3])
@pytest.mark.parametrize("seed", [42, 7, 8, 0, 999])
def test_source_task_oracle_cannot_beat_constant_half(rows, sources, seed):
    records = production(rows)
    if sources == 3:
        extra = copy.deepcopy(records[:6])
        for row in extra:
            row["record_id"] += "third"
            row["source"]["dataset"] = "third"
            row["group"]["task_family_id"] += "third"
        records += extra
    batch, report = balanced_batch(records, 32, seed=seed, max_record_reuse=4)
    strata = defaultdict(Counter)
    for row in batch:
        strata[(row["source"]["dataset"], row["group"]["task_family_id"])][
            row["outcome"]["failed"]
        ] += 1
    assert all(c[False] == c[True] > 0 for c in strata.values())
    assert max(report["source_counts"].values()) <= 16
    # Any probability constant within each task/source gets no more than .75.
    probabilities = {key: 0.1 + 0.8 * (i % 2) for i, key in enumerate(strata)}
    y = [r["outcome"]["failed"] for r in batch]
    p = [probabilities[(r["source"]["dataset"], r["group"]["task_family_id"])] for r in batch]
    assert binary_metrics(y, p)["auroc"] == 0.5
    assert 1 - binary_metrics(y, p)["brier"] <= 0.75
    assert balanced_batch(list(reversed(records)), 32, seed=seed, max_record_reuse=4) == (
        batch,
        report,
    )


def test_single_outcome_tasks_and_domains_cannot_fill_other_sources(rows):
    records = production(rows)
    for row in records:
        row["group"]["task_family_id"] += str(row["outcome"]["failed"])
    with pytest.raises(DataError, match="paired task"):
        balanced_batch(records, 32)
    records = production(rows)
    for row in records:
        row["source"]["domain"] = str(row["outcome"]["failed"])
    with pytest.raises(DataError, match="paired task"):
        balanced_batch(records, 32)


def test_single_class_source_excluded_and_reported(rows):
    records = production(rows)
    extra = copy.deepcopy(records[0])
    extra["record_id"] = "only-success"
    extra["source"]["dataset"] = "only-success"
    records.append(extra)
    _, report = balanced_batch(records, 8)
    assert "only-success" not in report["source_counts"]
    assert report["excluded_records"]["task_model_missing_paired_outcome"] == 1
    with pytest.raises(DataError):
        balanced_batch(production(rows), 6)  # cap=3 cannot hold two equal-class pairs per source


@pytest.mark.parametrize(
    "actions", [[], [{"name": "get_reservation_details"}], [{"name": "unknown_write_tool"}]]
)
@pytest.mark.parametrize("db_match", [True, False])
def test_read_only_db_labels_need_full_outcome_review(actions, db_match):
    raw = tau_row()
    raw["task"]["evaluation_criteria"]["actions"] = actions
    raw["simulation"]["reward_info"]["db_check"]["db_match"] = db_match
    result = convert(raw, revision="a" * 40)
    assert result["outcome"]["failed"] is None
    assert result["source"]["label_exclusion_reason"] == "db_only_no_verified_required_change"
    assert not db_change_task(raw["task"], "airline")
    assert admission_report([result])["eligible"] == {}


def test_unchanged_expected_hash_overrides_mutating_reference_and_native_success():
    raw = tau_row()
    raw["simulation"]["reward_info"]["db_check"]["db_match"] = True
    raw["db_verification"] = {
        "completed": True,
        "initial_db_hash": "a" * 64,
        "expected_db_hash": "A" * 64,
        "actual_db_hash": "a" * 64,
    }
    assert convert(raw, revision="a" * 40)["outcome"]["failed"] is None
    # Independent initial/target hashes can establish a required state transition
    # without a reference route. The actual trace need not call a specific tool.
    raw["task"]["evaluation_criteria"]["actions"] = []
    raw["db_verification"]["initial_db_hash"] = "b" * 64
    result = convert(raw, revision="a" * 40)
    assert result["outcome"]["failed"] is False
    assert scoring_exclusion_reason(result) is None
    assert "expected_vs_initial_hash" not in render_record(result)


@pytest.mark.parametrize("checker", ["db_state", {"kind": "db_state"}])
def test_legacy_db_exports_cannot_bypass_admission(rows, checker):
    row = convert(tau_row(), revision="a" * 40)
    row["outcome"]["checker"] = checker
    reason = "unverified_db_task_eligibility"
    assert admission_report([row])["exclusions"][0]["reason"] == reason
    control = copy.deepcopy(rows[0])
    control["split"] = row["split"]
    env = TraceAuditEnv([row, control], split=row["split"], allow_fixtures=True)
    assert row["record_id"] not in env._records
    assert env.exclusions[(row["source"]["dataset"], reason)] == 1
    row["split"] = "train"
    _, report = balanced_batch([*production(rows), row], 8)
    assert report["excluded_records"][reason] == 1
    data = full_cohorts(rows)
    for item in data:
        if item["source"]["dataset"] == "sierra-research/tau2-bench":
            item["outcome"]["checker"] = checker
    gate = evaluate_gate(
        data, {"perfect": [float(r["outcome"]["failed"]) for r in data]}, {"perfect": 0.5}
    )
    assert gate["comparison_status"] == "inconclusive"


def test_human_can_label_read_only_task_with_evidence():
    raw = tau_row()
    raw["task"]["evaluation_criteria"]["actions"] = []
    row = convert(raw, revision="a" * 40)
    queue = make_queue([row])
    queue[0].update(
        confirmed=True,
        failed=True,
        reviewer="reviewer",
        checker="The agent gave incorrect reservation information.",
    )
    reviewed, _ = apply_reviews([row], queue)
    assert scoring_exclusion_reason(reviewed[0]) is None


def test_write_probe_separates_reads_writes_and_unknowns(rows):
    row = copy.deepcopy(rows[0])
    row["source"].update(dataset="sierra-research/tau2-bench", domain="airline")
    for step in row["trajectory"]["steps"]:
        for item in step.get("tool_calls") or []:
            item["function_name"] = "get_reservation_details"
    call = row["trajectory"]["steps"][1]["tool_calls"][0]
    call["function_name"] = "get_reservation_details"
    assert write_tool_call_count(row) == 0
    call["function_name"] = "book_reservation"
    assert write_tool_call_count(row) == 1
    child = copy.deepcopy(row["trajectory"])
    row["trajectory"]["subagent_trajectories"] = [child]
    assert write_tool_call_count(row) == 2
    call["function_name"] = "unknown"
    assert write_tool_call_count(row) is None
    assert write_tool_call_count(rows[0]) is None


def port():
    spec = spec_from_file_location("reward_hacks_port", ROOT / "envs/trace_audit/verifiers/env.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trainer_hook_missing_repeated_and_valid_submission(rows):
    module = port()
    kit = module.TraceAuditToolkit(rows, allow_fixtures=True)
    assert module.consume_toolkit_reward(kit) == 0
    kit.reset(seed=1)
    assert module.consume_toolkit_reward(kit) == 0
    kit.submit_analysis('{"p_failed":0.5}')
    assert module.consume_toolkit_reward(kit) == 0
    kit.reset(seed=1)
    kit.submit_analysis('{"p_failed":0.5}')
    assert module.consume_toolkit_reward(kit) == 0.75
    assert module.consume_toolkit_reward(kit) == 0
    kit.reset(seed=1)
    kit.submit_analysis('{"p_failed":0.5}')
    kit.submit_analysis('{"p_failed":0.5}')
    assert module.consume_toolkit_reward(kit) == 0


def test_training_pairing_happens_after_budget_admission(rows):
    module = port()
    records = production(rows)
    # Exclude every failure from a single source through the real token budget.
    source = records[0]["source"]["dataset"]
    for row in records:
        if row["source"]["dataset"] == source and row["outcome"]["failed"]:
            row["trajectory"]["steps"][-1]["message"] = "x" * 40000
    with pytest.raises(DataError, match="paired task"):
        module.TraceAuditToolkit(records)


def test_native_uses_paired_training_and_original_eval():
    pytest.importorskip("verifiers")
    module = port()
    env, report = module.create_verifiers_env(allow_fixtures=True, return_report=True)
    assert len(env.dataset) == 8
    assert report["max_record_reuse"] == 1
    assert report["shortcut_audit"]["sampled_records"] == len(env.dataset)
    assert len(env.eval_dataset) == 12
    assert all(c["succeeded"] == c["failed"] for c in report["source_outcome_counts"].values())


def test_swe_audit_is_revision_bound_and_disagreements_block_grader_only_training(rows):
    records = production(rows)
    template = records[0]
    audited = []
    for i in range(50):
        row = copy.deepcopy(template)
        row["record_id"] = f"audit-{i}"
        row["source"].update(
            dataset="nebius/SWE-rebench-openhands-trajectories", original_id=f"audit-{i}"
        )
        queue = make_queue([row])
        queue[0].update(
            confirmed=True,
            failed=False,
            reviewer="human",
            checker="Reviewed patch and task requirements",
        )
        reviewed, _ = apply_reviews([row], queue)
        audited.extend(reviewed)
    unreviewed = copy.deepcopy(template)
    unreviewed["source"]["dataset"] = "nebius/SWE-rebench-openhands-trajectories"
    with pytest.raises(DataError, match="50 human"):
        assert_training_audits([*audited[:49], unreviewed])
    assert_training_audits([*audited, unreviewed])
    unreviewed["source"]["revision"] = "new-revision"
    with pytest.raises(DataError, match="new-revision"):
        assert_training_audits([*audited, unreviewed])
    unreviewed["source"]["revision"] = template["source"]["revision"]
    audited[0]["outcome"]["failed"] = True
    with pytest.raises(DataError, match="disagreements"):
        assert_training_audits([*audited, unreviewed])
    assert_training_audits(audited)  # All labels independently reviewed: no grader-only reward.


def test_core_default_training_reset_rejects_unpaired_tasks(rows):
    records = production(rows)
    for row in records:
        row["group"]["task_family_id"] += str(row["outcome"]["failed"])
    env = TraceAuditEnv(records)
    with pytest.raises(DataError, match="paired task"):
        env.reset()


def test_direct_sampler_cannot_bypass_swe_audit(rows):
    records = production(rows)
    source = records[0]["source"]["dataset"]
    for row in records:
        if row["source"]["dataset"] == source:
            row["source"]["dataset"] = "nebius/SWE-rebench-openhands-trajectories"
    with pytest.raises(DataError, match="human label audits"):
        balanced_batch(records, 8)
