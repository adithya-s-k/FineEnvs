import asyncio
import copy
from importlib.util import module_from_spec, spec_from_file_location

import pytest
from conftest import ROOT
from trace_audit.core.gate import evaluate_gate
from trace_audit.core.metrics import binary_metrics, choose_threshold, shortcut_metrics
from trace_audit.core.records import validate_collection

from adapters.external import convert as external


def test_auc_ties_and_single_class():
    assert binary_metrics([False, True], [0.5, 0.5])["auroc"] == 0.5
    assert binary_metrics([False, True], [0, 1])["auroc"] == 1
    assert binary_metrics([False, True], [1, 0])["auroc"] == 0
    assert binary_metrics([True], [0.2])["auroc"] is None
    assert binary_metrics([False, True], [0, 1])["brier"] == 0
    assert choose_threshold([False, True], [0.1, 0.4])["threshold"] == 0.4
    assert not choose_threshold([False, True], [0.5, 0.5])["minimums_feasible"]


def test_offline_rollout_reports_pending_gates(rows):
    spec = spec_from_file_location("audit_rollout", ROOT / "envs/trace_audit/verifiers/rollout.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.evaluate(rows, allow_fixtures=True)
    assert report["fixture_only"] is True
    assert "pending" in report["prompted_judge"]
    assert report["baselines"]["training_prior"]["test"]["auroc"] == 0.5
    assert set(report["shortcut_audit"]["write_calls"]) == {"train", "dev", "test"}
    assert report["episode"]["reward"] == report["episode_smoke"]["reward"]
    assert report["episode"]["response"] == {"p_failed": 1 / 3}
    assert report["rl_gate"]["comparison_status"] == "inconclusive"


def test_evaluation_validates_each_input_record_once(rows, monkeypatch):
    import trace_audit.core.records as records
    from trace_audit.core.evaluation import evaluate

    checked = []
    validate = records.validate_record

    def track(row):
        checked.append(row["record_id"])
        validate(row)

    monkeypatch.setattr(records, "validate_record", track)
    evaluate(rows, allow_fixtures=True)
    assert checked == [row["record_id"] for row in rows]


def test_native_verifiers_offline_rollout(monkeypatch):
    vf = pytest.importorskip("verifiers")
    if not hasattr(vf, "SingleTurnEnv"):
        pytest.skip("Install the native extra for Verifiers integration")
    from openai.types.chat import ChatCompletion

    spec = spec_from_file_location("native_port", ROOT / "envs/trace_audit/verifiers/env.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    env = module.create_verifiers_env(allow_fixtures=True)
    assert isinstance(env, vf.SingleTurnEnv) and not env.oai_tools
    calls = []

    async def fake_response(state, prompt):
        calls.append(prompt)
        return ChatCompletion(
            id="offline",
            model="fixture",
            object="chat.completion",
            created=0,
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"p_failed":0.5}'},
                }
            ],
            usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        )

    monkeypatch.setattr(env, "get_model_response", fake_response)

    async def run():
        state = await env.run_rollout(
            env.eval_dataset[0],
            client=None,
            model="offline",
            gen_sampling_args={},
            gen_sem=asyncio.Semaphore(1),
            score_sem=asyncio.Semaphore(1),
            score=True,
        )
        assert state["reward"] == 0.75
        assert len(calls) == 1
        assert len(state["completion"]) == 1
        await env.rubric.score_rollout(state, score_sem=asyncio.Semaphore(1))
        assert state["reward"] == 0

    asyncio.run(run())


def test_reverse_shortcuts_and_single_class():
    metrics = shortcut_metrics([False, False, True, True], [0.8, 0.4, 0.6, 0.2])
    assert metrics["auroc"] == 0.25
    assert metrics["predictive_strength_auroc"] == 0.75
    assert metrics["signal_direction"] == "decreasing"
    assert shortcut_metrics([True], [0.4])["predictive_strength_auroc"] is None


def test_gate_does_not_count_fixture_or_training_source_results(rows):
    spec = spec_from_file_location("gate_rollout", ROOT / "envs/trace_audit/verifiers/rollout.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.evaluate(rows, allow_fixtures=True)
    assert report["rl_gate"]["heldout_records"] == 0
    assert not report["rl_gate"]["baseline_meets_minimums_all_outcome_cohorts"]
    assert len(report["rl_gate"]["missing_heldout_sources"]) == 4


def cohort_row(template, dataset, failed, index, source_split=None):
    row = copy.deepcopy(template)
    row["record_id"] = f"{dataset}:{failed}:{index}"
    row["source"].update(
        dataset=dataset,
        original_id=f"{failed}:{index}",
        data_kind="unit_test_case",
        source_split=source_split,
    )
    row["group"]["task_family_id"] = f"test-family:{dataset}:{index}"
    row["split"] = "test"
    if dataset == "sierra-research/tau2-bench":
        row["source"]["domain"] = "retail"
        row["group"]["task_family_id"] = "tau2:retail:33"
        row["outcome"] = {
            "failed": failed,
            "label_source": "verified_outcome",
            "checker": {"kind": "db_state", "task_requires_db_change": True},
            "reviewer": None,
        }
    else:
        row["outcome"] = {
            "failed": failed,
            "label_source": "expert",
            "checker": "unit test evidence",
            "reviewer": "unit reviewer",
        }
        if dataset == "microsoft/AgentRx" and source_split == "tau_retail":
            row["group"]["task_family_id"] = "tau2:retail:33"
    return row


def full_cohorts(rows):
    data = []
    for dataset in (
        "sierra-research/tau2-bench",
        "microsoft/AgentRx",
        "PatronusAI/TRAIL",
        "Kevin355/Who_and_When",
    ):
        for i in range(10):
            data.append(
                cohort_row(
                    rows[0],
                    dataset,
                    True,
                    i,
                    "tau_retail" if dataset == "microsoft/AgentRx" else None,
                )
            )
        if dataset != "microsoft/AgentRx":
            data.append(cohort_row(rows[0], dataset, False, 100))
    validate_collection(data)
    return data


def test_always_failed_cannot_satisfy_gate_even_with_high_precision(rows):
    data = full_cohorts(rows)
    result = evaluate_gate(
        data,
        {"always_failed": [1.0] * len(data)},
        {"always_failed": 0.5},
        training_audit={"status": "clear", "release_allowed": True},
    )
    assert result["comparison_status"] == "baselines_below_minimums"
    assert not result["baseline_meets_minimums_all_outcome_cohorts"]
    assert result["heldout_metrics"]["always_failed"]["tau2:retail"]["precision"] >= 0.8
    for group in result["outcome_comparison_groups"]:
        metric = result["heldout_metrics"]["always_failed"][group]
        assert metric["recall"] == 1
        assert metric["specificity"] == 0 and metric["auroc"] == 0.5
        assert not metric["meets_minimums"]


@pytest.mark.parametrize("missing", ["controls", "retail_cases"])
def test_pairing_requires_verified_controls_and_explicit_retail_provenance(rows, missing):
    data = full_cohorts(rows)
    for row in data:
        if (
            missing == "controls"
            and row["source"]["dataset"] == "sierra-research/tau2-bench"
            and row["outcome"]["failed"] is False
        ):
            row["outcome"].update(
                label_source="expert", checker="human label", reviewer="unit reviewer"
            )
        if missing == "retail_cases" and row["source"]["dataset"] == "microsoft/AgentRx":
            row["source"]["source_split"] = "unknown"
    result = evaluate_gate(data, {"constant": [1.0] * len(data)}, {"constant": 0.5})
    assert result["comparison_status"] == "inconclusive"
    assert not result["cohorts"]["microsoft/AgentRx"]["gate_evaluable"]
    assert result["cohorts"]["microsoft/AgentRx"]["success_controls"] == 0
    assert result["cohorts"]["microsoft/AgentRx"]["reason"] == (
        "missing_verified_retail_success_controls"
        if missing == "controls"
        else "missing_agentrx_retail_cases"
    )


def test_dev_selection_rejects_a_majority_only_feasible_threshold():
    # Precision and recall alone accept all-failed at a 90% failure prior.
    choice = choose_threshold([True] * 9 + [False], [0.9] * 10)
    assert not choice["minimums_feasible"]
    assert "specificity" in choice["selection"]


def decide(data, scores, name="perfect"):
    # Isolate outcome-comparison behavior; shortcut blockers have dedicated tests.
    return evaluate_gate(
        data,
        {name: scores},
        {name: 0.5},
        training_audit={"status": "clear", "release_allowed": True},
    )


def test_perfect_comparison_can_conclude_with_failure_only_who_when(rows):
    data = [
        r
        for r in full_cohorts(rows)
        if not (
            r["source"]["dataset"] == "Kevin355/Who_and_When" and r["outcome"]["failed"] is False
        )
    ]
    result = decide(data, [0.9 if r["outcome"]["failed"] else 0.1 for r in data])
    assert result["comparison_status"] == "baseline_meets_minimums"
    assert result["outcome_comparison_groups"] == ["microsoft/AgentRx", "tau2:retail"]
    assert result["cohorts"]["Kevin355/Who_and_When"]["role"] == "diagnostic_only"
    assert result["qualifying_baselines"] == ["perfect"]


def test_diagnostic_predictions_cannot_pass_or_fail_retail_comparison(rows):
    data = full_cohorts(rows)
    # Deliberately wrong predictions for both diagnostic sources are still reported.
    scores = [
        (0.9 if not r["outcome"]["failed"] else 0.1)
        if r["source"]["dataset"] in {"PatronusAI/TRAIL", "Kevin355/Who_and_When"}
        else (0.9 if r["outcome"]["failed"] else 0.1)
        for r in data
    ]
    result = decide(data, scores)
    assert result["comparison_status"] == "baseline_meets_minimums"
    assert result["heldout_metrics"]["perfect"]["PatronusAI/TRAIL"]["auroc"] == 0
    assert result["heldout_metrics"]["perfect"]["PatronusAI/TRAIL"]["meets_minimums"] is None


def test_missing_diagnostics_are_reported_but_missing_outcome_groups_block(rows):
    data = [
        r
        for r in full_cohorts(rows)
        if r["source"]["dataset"] not in {"PatronusAI/TRAIL", "Kevin355/Who_and_When"}
    ]
    result = decide(data, [0.9 if r["outcome"]["failed"] else 0.1 for r in data])
    assert result["comparison_status"] == "baseline_meets_minimums"
    assert set(result["missing_heldout_sources"]) == {"PatronusAI/TRAIL", "Kevin355/Who_and_When"}
    without_agentrx = [r for r in data if r["source"]["dataset"] != "microsoft/AgentRx"]
    assert (
        decide(without_agentrx, [0.9 if r["outcome"]["failed"] else 0.1 for r in without_agentrx])[
            "comparison_status"
        ]
        == "inconclusive"
    )


def test_no_matching_task_controls_is_inconclusive_not_fallback_to_other_tasks(rows):
    data = full_cohorts(rows)
    for row in data:
        if row["source"]["dataset"] == "microsoft/AgentRx":
            row["group"]["task_family_id"] = "tau2:retail:99"
    result = decide(data, [0.9 if r["outcome"]["failed"] else 0.1 for r in data])
    assert result["comparison_status"] == "inconclusive"
    pair = result["cohorts"]["microsoft/AgentRx"]
    assert pair["reason"] == "no_matched_task_families"
    assert pair["outcome_records"] == 0 and pair["success_controls"] == 0


def test_local_source_ids_do_not_imply_canonical_task_equivalence(rows):
    data = full_cohorts(rows)
    for row in data:
        if row["source"]["dataset"] == "microsoft/AgentRx":
            row["source"]["original_task_family_id"] = "33"
            row["group"]["task_family_id"] = "microsoft/AgentRx:33"
    result = decide(data, [0.9 if r["outcome"]["failed"] else 0.1 for r in data])
    assert result["cohorts"]["microsoft/AgentRx"]["reason"] == "no_matched_task_families"


def two_task_cases(rows):
    data = []
    for task, case_count, control_count in (("33", 8, 2), ("34", 2, 5)):
        for i in range(case_count):
            row = cohort_row(rows[0], "microsoft/AgentRx", True, int(task) * 100 + i, "tau_retail")
            row["group"]["task_family_id"] = f"tau2:retail:{task}"
            data.append(row)
        for i in range(control_count):
            row = cohort_row(rows[0], "sierra-research/tau2-bench", False, int(task) * 100 + i)
            row["group"]["task_family_id"] = f"tau2:retail:{task}"
            data.append(row)
        # Native retail also has failures, keeping that required cohort evaluable.
        row = cohort_row(rows[0], "sierra-research/tau2-bench", True, int(task) * 100)
        row["group"]["task_family_id"] = f"tau2:retail:{task}"
        data.append(row)
    unmatched = cohort_row(rows[0], "microsoft/AgentRx", True, 9999, "tau_retail")
    unmatched["group"]["task_family_id"] = "tau2:retail:99"
    data.append(unmatched)
    validate_collection(data)
    return data


def test_task_identity_oracle_cannot_separate_balanced_same_task_pairs(rows):
    data = two_task_cases(rows)
    # Scores know the task but observe no trace: one score shared by both outcomes per task.
    scores = [0.9 if r["group"]["task_family_id"] == "tau2:retail:33" else 0.1 for r in data]
    result = decide(data, scores, "task_identity_oracle")
    pair = result["cohorts"]["microsoft/AgentRx"]
    assert pair["positive_records"] == pair["negative_records"] == 4
    assert len(pair["task_matching"]["matched_tasks"]) == 2
    assert {t["task_family_id"] for t in pair["task_matching"]["matched_tasks"]} == {
        "tau2:retail:33",
        "tau2:retail:34",
    }
    assert result["heldout_metrics"]["task_identity_oracle"]["microsoft/AgentRx"]["auroc"] == 0.5
    assert not result["baseline_meets_minimums_all_outcome_cohorts"]


def test_pair_selection_does_not_depend_on_prediction_scores_or_input_order(rows):
    data = two_task_cases(rows)
    first = decide(data, [0.0] * len(data))
    reversed_data = list(reversed(data))
    second = decide(reversed_data, [1.0] * len(data))
    assert first["cohorts"]["microsoft/AgentRx"] == second["cohorts"]["microsoft/AgentRx"]
    # Outcomes do not change source input records or their original labels.
    assert len(data) == 20 and sum(r["outcome"]["failed"] for r in data) == 13


def test_task_only_success_blocks_a_trace_baseline_interpretation(rows):
    data = full_cohorts(rows)
    perfect = [0.9 if r["outcome"]["failed"] else 0.1 for r in data]
    result = evaluate_gate(
        data,
        {"trace_judge": perfect, "task_only_tfidf": perfect},
        {"trace_judge": 0.5, "task_only_tfidf": 0.5},
    )
    assert result["comparison_status"] == "inconclusive_task_shortcut"
    assert result["task_only_audit"]["shortcut_detected"]
    assert result["candidate_baselines_meeting_thresholds"] == ["trace_judge"]
    assert result["qualifying_baselines"] == []
    assert not result["baseline_meets_minimums_all_outcome_cohorts"]


def test_canonical_join_provenance_preserved_by_adapter(rows):
    raw = {
        "original_id": "case",
        "task_family_id": "33",
        "canonical_task_family_id": "tau2:retail:33",
        "source_split": "tau_retail",
        "task": rows[0]["task"],
        "trajectory": rows[0]["trajectory"],
        "step_map": {},
        "failures": [],
    }
    result = external(raw, dataset="microsoft/AgentRx", revision="a" * 40, license="CC-BY-4.0")
    assert (
        result["group"]["task_family_id"]
        == result["source"]["canonical_task_family_id"]
        == "tau2:retail:33"
    )
    assert result["source"]["original_task_family_id"] == "33"


@pytest.mark.parametrize("source", ["PatronusAI/TRAIL", "Kevin355/Who_and_When"])
def test_failure_only_diagnostics_do_not_block_outcome_comparison(rows, source):
    data = [
        r
        for r in full_cohorts(rows)
        if not (r["source"]["dataset"] == source and r["outcome"]["failed"] is False)
    ]
    scores = [0.9 if r["outcome"]["failed"] else 0.1 for r in data]
    result = evaluate_gate(
        data,
        {"perfect": scores},
        {"perfect": 0.5},
        training_audit={"status": "clear", "release_allowed": True},
    )
    assert result["comparison_status"] == "baseline_meets_minimums"
    assert result["cohorts"][source]["role"] == "diagnostic_only"
    assert result["cohorts"][source]["reason"] == "single_outcome_diagnostic"
    assert result["heldout_metrics"]["perfect"][source]["meets_minimums"] is None
    assert result["baseline_meets_minimums_all_outcome_cohorts"]


@pytest.mark.parametrize("checker", [{"kind": "db_state", "task_requires_db_change": True}])
def test_agentrx_retail_has_real_controls_and_other_domains_are_unpaired(rows, checker):
    data = full_cohorts(rows)
    for row in data:
        if (
            row["source"]["dataset"] == "sierra-research/tau2-bench"
            and row["outcome"]["failed"] is False
        ):
            row["outcome"]["checker"] = checker
    data.append(cohort_row(rows[0], "microsoft/AgentRx", True, 1000, "magentic_one"))
    scores = [0.9 if r["outcome"]["failed"] else 0.1 for r in data]
    result = evaluate_gate(
        data,
        {"perfect": scores},
        {"perfect": 0.5},
        training_audit={"status": "clear", "release_allowed": True},
    )
    pair = result["cohorts"]["microsoft/AgentRx"]
    assert pair["success_controls"] == 1 and pair["outcome_records"] == 2
    assert pair["native_records"] == 11 and pair["unpaired_native_records"] == 10
    assert pair["task_matching"]["matched_tasks"][0]["selected_per_outcome"] == 1
    assert result["baseline_meets_minimums_all_outcome_cohorts"]
