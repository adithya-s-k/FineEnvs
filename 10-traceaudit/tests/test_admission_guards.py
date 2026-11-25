"""Guards on audit evidence, data admission, conversion output and rendering."""

import copy
import json
from collections import Counter

import pytest
from test_adapters import tau_row
from test_reward_hacks import port
from trace_audit.core import (
    DataError,
    TraceAuditEnv,
    baselines,
    environment,
    evaluation,
    render_record,
    shortcuts,
)
from trace_audit.core.baselines import last_tool_error
from trace_audit.core.records import training_split, validate_record
from trace_audit.core.review import apply_reviews, make_queue
from trace_audit.core.sampling import balanced_batch

from tools.tau2 import main as tau_main


@pytest.fixture
def reviewed_swe(rows):
    records = []
    for task in range(100):
        family = f"swe-task-{task}"
        for failed in (False, True):
            row = copy.deepcopy(rows[0])
            key = f"swe-{task}-{failed}"
            row["record_id"] = key
            row["source"].update(
                dataset="nebius/SWE-rebench-openhands-trajectories",
                data_kind="test_inputs",
                original_id=key,
            )
            row["group"]["task_family_id"] = family
            row["split"] = training_split(family)
            row["outcome"]["failed"] = failed
            records.append(row)
    queue = make_queue(records)
    for item in queue:
        item.update(
            confirmed=True,
            failed=item["existing_outcome"]["failed"],
            reviewer="human",
            checker="Compared the task and final state",
        )
    reviewed, _ = apply_reviews(records, queue)
    return reviewed


@pytest.mark.parametrize("exclude_reviews", [False, True])
def test_completed_swe_audit_survives_split_and_budget_filtering(reviewed_swe, exclude_reviews):
    audited = [row for row in reviewed_swe if row.get("review_history")]
    assert len(audited) == 50
    assert 0 < sum(row["split"] == "train" for row in audited) < 50
    batch, _ = balanced_batch(reviewed_swe, 2, source_cap=1)
    assert len(batch) == 2 and all(row["split"] == "train" for row in batch)
    if exclude_reviews:
        for row in audited:
            row["trajectory"]["steps"][-1]["message"] = "REVIEW_ONLY"
    env = TraceAuditEnv(
        reviewed_swe,
        source_cap=1,
        token_counter=lambda text: 32001 if "REVIEW_ONLY" in text else 1,
    )
    assert env.reset(seed=42)
    if exclude_reviews:
        assert not {row["record_id"] for row in audited} & {
            row["record_id"] for row in env.training_sample()[0]
        }


def test_offline_evaluation_uses_complete_revision_audit(rows, reviewed_swe):
    report = evaluation.evaluate([*rows, *reviewed_swe], allow_fixtures=True, source_cap=1)
    audit = report["shortcut_audit"]["paired_training"]
    assert audit.get("reason") != "training_sample_unavailable"
    assert audit["sampled_records"] > 0


@pytest.mark.parametrize("problem", ["disagreement", "different_revision", "insufficient"])
def test_dev_audit_evidence_keeps_revision_and_agreement_checks(reviewed_swe, problem):
    audited = [row for row in reviewed_swe if row.get("review_history")]
    if problem == "disagreement":
        dev = next(row for row in audited if row["split"] == "dev")
        dev["outcome"]["failed"] = not dev["outcome"]["failed"]
    elif problem == "different_revision":
        for row in audited:
            row["source"]["revision"] = "other-revision"
    else:
        audited[0]["review_history"] = []
    with pytest.raises(DataError, match="50 human label audits"):
        TraceAuditEnv(reviewed_swe, source_cap=1).reset(seed=42)


@pytest.mark.parametrize("existing", [False, True])
def test_invalid_tau_conversion_does_not_create_or_overwrite_outputs(
    tmp_path, monkeypatch, existing
):
    raw = tau_row()
    simulation = {**raw["simulation"], "task_id": raw["task"]["id"]}
    source, output = tmp_path / "input.json", tmp_path / "output.jsonl"
    policy, tools = tmp_path / "policy.txt", tmp_path / "tools.json"
    report = output.with_suffix(".jsonl.report.json")
    source.write_text(json.dumps({"tasks": [raw["task"]], "simulations": [simulation] * 2}))
    policy.write_text("Policy")
    tools.write_text("[]")
    if existing:
        output.write_text("previous records")
        report.write_text("previous report")
    monkeypatch.setattr(
        "sys.argv",
        [
            "tau2.py",
            "convert",
            str(source),
            str(output),
            "--revision",
            "a" * 40,
            "--domain",
            "airline",
            "--policy",
            str(policy),
            "--tools",
            str(tools),
        ],
    )
    with pytest.raises(DataError, match="Duplicate record_id"):
        tau_main()
    if existing:
        assert output.read_text() == "previous records"
        assert report.read_text() == "previous report"
    else:
        assert not output.exists() and not report.exists()


def test_null_tool_error_is_valid_and_evaluates(rows):
    for row in rows:
        for step in row["trajectory"]["steps"]:
            for result in (step.get("observation") or {}).get("results", []):
                result.setdefault("extra", {})["error"] = None
        validate_record(row)
        assert not last_tool_error(row)
    report = evaluation.evaluate(rows, allow_fixtures=True)
    assert "last_tool_error" in report["baselines"]


def test_internal_training_audit_value_error_is_visible(rows, monkeypatch):
    def broken_audit(*args, **kwargs):
        raise ValueError("internal audit defect")

    monkeypatch.setattr(evaluation, "training_shortcut_audit", broken_audit)
    with pytest.raises(ValueError, match="internal audit defect"):
        evaluation.evaluate(rows, allow_fixtures=True)


def test_unavailable_training_sample_remains_a_reported_data_error(rows):
    report = evaluation.evaluate(rows, allow_fixtures=True, training_size=4000)
    assert report["shortcut_audit"]["paired_training"]["reason"] == "training_sample_unavailable"


@pytest.mark.parametrize(
    "dataset",
    [
        "example/trail-analysis",
        "example/agentrx-v2",
        "example/who_and_when-v2",
        "example/tau-helper",
    ],
)
def test_unrelated_source_substrings_do_not_force_test_split(rows, dataset):
    row = copy.deepcopy(rows[0])
    row["source"].update(dataset=dataset, domain="retail", data_kind="test_inputs")
    validate_record(row)


def test_evaluation_renders_each_input_once(rows, monkeypatch):
    calls = Counter()

    def track(row):
        calls[row["record_id"]] += 1
        return render_record(row)

    for module in (environment, baselines, shortcuts):
        monkeypatch.setattr(module, "render_record", track)
    evaluation.evaluate(rows, allow_fixtures=True)
    assert calls == Counter({row["record_id"]: 1 for row in rows})


def test_toolkit_skips_statistical_audit_and_reports_do_not_mutate_core(rows, monkeypatch):
    module = port()
    audits = []

    def track(batch, **kwargs):
        audits.append(batch)
        return {"sampled_records": len(batch)}

    monkeypatch.setattr(module, "training_shortcut_audit", track)
    kit = module.TraceAuditToolkit(rows, allow_fixtures=True)
    assert not audits
    kit.reset(seed=7)
    kit.submit_analysis('{"p_failed":0.5}')
    assert module.consume_toolkit_reward(kit) == 0.75
    _, core_report = kit._env.training_sample()
    before = copy.deepcopy(core_report)
    _, report = module._training_records(kit._env)
    assert len(audits) == 1 and "shortcut_audit" in report
    assert core_report == before
    assert kit._sampling_report is not core_report


def test_identity_keys_are_redacted_without_losing_colliding_values(rows):
    row = copy.deepcopy(rows[0])
    row["trajectory"]["agent"]["name"] = "task"
    row["trajectory"]["agent"]["tool_definitions"] = [
        {"gpt-private-1": {"claude-private-2": "schema evidence"}}
    ]
    call = next(c for s in row["trajectory"]["steps"] for c in (s.get("tool_calls") or []))
    call["arguments"] = {
        "gpt-private-1": "first evidence",
        "claude-private-2": "second evidence",
        "[model removed]": "third evidence",
        "nested": {"task": "task"},
    }
    validate_record(row)
    rendered = render_record(row)
    assert "gpt-private-1" not in rendered and "claude-private-2" not in rendered
    public = json.loads(rendered)
    assert "task" in public
    arguments = next(
        s["tool_calls"][0]["arguments"]
        for s in public["trajectory"]["steps"]
        if s.get("tool_calls")
    )
    assert list(arguments.values())[:3] == ["first evidence", "second evidence", "third evidence"]
    assert arguments["nested"] == {"[identity removed]": "[identity removed]"}
