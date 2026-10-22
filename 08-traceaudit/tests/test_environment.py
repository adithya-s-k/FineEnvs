import copy
import inspect
import json
from importlib.util import module_from_spec, spec_from_file_location

import pytest
from conftest import ROOT
from trace_audit.core import (
    DataError,
    TraceAuditEnv,
    parse_action,
    render_record,
    score_probability,
    validate_record,
)
from trace_audit.core.render import task_text


@pytest.mark.parametrize(
    "action",
    [
        None,
        "",
        "null",
        "[]",
        "{}",
        '{"p_failed":true}',
        '{"p_failed":"0.9"}',
        '{"p_failed":NaN}',
        '{"p_failed":Infinity}',
        '{"p_failed":1e999}',
        '{"p_failed":-0.1}',
        '{"p_failed":1.1}',
        '{"p_failed":0.5,"p_failed":1}',
        '```json\n{"p_failed":1}\n```',
        '{"p_failed":0.5} trailing',
        {"p_failed": 10**400},
    ],
)
def test_malformed_scores_zero(action, rows):
    env = TraceAuditEnv(rows, allow_fixtures=True)
    env.reset(seed=1)
    result = env.step(action)
    assert result.reward == 0 and result.done


def test_optional_fields_do_not_change_reward(rows):
    env = TraceAuditEnv(rows, allow_fixtures=True)
    key = next(iter(env._records))
    env.reset(record_id=key)
    expected = env.step({"p_failed": 0.8}).reward
    env.reset(record_id=key)
    result = env.step(
        {"p_failed": 0.8, "category": {}, "critical_step": True, "evidence_steps": [{}, 999]}
    )
    assert result.reward == expected and len(result.info["optional_warnings"]) == 3


def test_once_only_and_isolation(rows):
    a, b = TraceAuditEnv(rows, allow_fixtures=True), TraceAuditEnv(rows, allow_fixtures=True)
    prompt = a.reset(seed=42)
    assert b.reset(seed=42) == prompt
    assert a.step({"p_failed": 0.5}).reward == 0.75
    assert a.step({"p_failed": 1}).reward == 0
    assert b.step({"p_failed": 0.5}).reward == 0.75
    assert a.reset(seed=42) == prompt
    assert a.step(None).reward == 0


def test_unlabelled_and_fixtures_are_not_production(rows):
    with pytest.raises(DataError):
        TraceAuditEnv(rows)
    env = TraceAuditEnv(rows, allow_fixtures=True)
    assert "unlabelled-fixture" not in env._records
    assert env.exclusions[(rows[0]["source"]["dataset"], "unlabelled")] == 1
    with pytest.raises(ValueError):
        score_probability(parse_action({"p_failed": 0.5}, {1}), None)


def test_budget_excludes_whole_trace(rows):
    large = copy.deepcopy(rows[0])
    large["record_id"] = "long"
    large["source"]["original_id"] = "long"
    large["trajectory"]["steps"][-1]["message"] = "界" * 32001
    env = TraceAuditEnv([*rows, large], allow_fixtures=True)
    assert "long" not in env._records
    assert env.exclusions[(large["source"]["dataset"], "over_render_budget")] == 1


def test_private_metadata_not_rendered(rows):
    row = copy.deepcopy(rows[0])
    row["trajectory"]["extra"] = {"label": "SECRET_FAILURE", "path": "PRIVATE_TRACE_FILE"}
    row["trajectory"]["agent"]["model_name"] = "gpt-secret-123"
    row["trajectory"]["steps"][1]["model_name"] = "my-private-model"
    row["trajectory"]["steps"][1]["message"] = "OpenHands using gpt-secret-123 and my-private-model"
    row["outcome"]["checker"] = "SECRET_CHECKER"
    row["task"]["success_criteria"] = "SECRET_GOLD"
    row["trajectory"]["steps"][1]["tool_calls"][0]["tool_call_id"] = "call_private_failure_label"
    row["trajectory"]["steps"][1]["observation"]["results"][0]["source_call_id"] = (
        "call_private_failure_label"
    )
    visible = render_record(row)
    for private in [
        "SECRET_FAILURE",
        "PRIVATE_TRACE_FILE",
        "SECRET_CHECKER",
        "SECRET_GOLD",
        "OpenHands",
        "gpt-secret-123",
        "my-private-model",
        row["record_id"],
        row["source"]["dataset"],
        "call_private_failure_label",
    ]:
        assert private not in visible
    assert '"step_id": 2' in visible
    assert '"call_1"' in visible


def test_family_split_isolation_even_direct_constructor(rows):
    wrong = copy.deepcopy(rows[0])
    wrong["record_id"] = "different"
    wrong["source"]["original_id"] = "different"
    wrong["split"] = "test"
    with pytest.raises(DataError, match="family"):
        TraceAuditEnv([*rows, wrong], allow_fixtures=True)


def test_required_outcome_and_atif_invariants(rows):
    row = copy.deepcopy(rows[0])
    del row["outcome"]["failed"]
    with pytest.raises(DataError):
        validate_record(row)
    row = copy.deepcopy(rows[0])
    row["trajectory"]["steps"][1]["observation"]["results"][0]["source_call_id"] = "orphan"
    with pytest.raises(DataError):
        validate_record(row)


def test_toolkit_methods_do_not_reveal_labels_or_reward(rows):
    spec = spec_from_file_location("traceaudit_port", ROOT / "envs/trace_audit/verifiers/env.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TOOL_FUNCTIONS == []
    kit = module.TraceAuditToolkit(rows, allow_fixtures=True)
    assert {
        name
        for name, method in inspect.getmembers(kit, inspect.ismethod)
        if not name.startswith("_")
    } == {"reset", "submit_analysis"}
    kit.reset(seed=1)
    assert kit.submit_analysis('{"p_failed":0.5}') == "Analysis submitted. Episode complete."
    assert kit._env._last_transition.reward == 0.75
    assert kit.submit_analysis('{"p_failed":1}') == "Analysis submitted. Episode complete."
    assert kit._env._last_transition.reward == 0


def test_explicit_token_counter_and_multimodal_exclusions(rows):
    row = copy.deepcopy(rows[0])
    row["record_id"] = "multimodal"
    row["source"]["original_id"] = "multimodal"
    row["trajectory"]["steps"][0]["message"] = [{"type": "text", "text": "Visible part"}]
    validate_record(row)
    env = TraceAuditEnv(
        [*rows, row], allow_fixtures=True, max_tokens=1, token_counter=lambda text: 1
    )
    assert len(env._records) == 12
    assert env.exclusions[(row["source"]["dataset"], "unsupported_multimodal")] == 1


def test_nested_delegates_and_quoted_identities_preserve_json(rows):
    row = copy.deepcopy(rows[0])
    child = copy.deepcopy(row["trajectory"])
    child["trajectory_id"] = "source-name-private-id"
    child["agent"]["name"] = 'Agent "Nested"'
    child["steps"][1]["message"] = 'Agent "Nested" checked the file.'
    row["trajectory"]["subagent_trajectories"] = [child]
    row["trajectory"]["steps"][1]["observation"]["results"][0]["subagent_trajectory_ref"] = [
        {"trajectory_id": child["trajectory_id"]}
    ]
    validate_record(row)
    visible = render_record(row)
    parsed = json.loads(visible)
    assert "Nested" not in visible and "source-name-private-id" not in visible
    assert parsed["trajectory"]["delegates"][0]["id"] == "delegate_1"
    assert parsed["trajectory"]["steps"][1]["observations"][0]["delegated_to"] == ["delegate_1"]


def test_unjoined_continuation_rejected(rows):
    row = copy.deepcopy(rows[0])
    row["trajectory"]["continued_trajectory_ref"] = {"trajectory_path": "earlier.json"}
    with pytest.raises(DataError, match="continued"):
        validate_record(row)


def test_trail_common_word_and_task_dedup(rows):
    raw = copy.deepcopy(rows[0])
    text = "Keep an audit trail; trailing abcdef foo3 are ordinary text. TRAIL is the dataset."
    raw["task"]["instruction"] = text
    raw["trajectory"]["steps"][0]["message"] = text
    rendered = render_record(raw)
    assert rendered.count("audit trail") == 1
    assert "trailing abcdef foo3" in rendered and "TRAIL" not in rendered
    public = json.loads(rendered)
    assert public["task"] == {"instruction_step_id": 1}
    assert task_text(public).startswith("Keep an audit trail")
    raw["task"]["instruction"] = "Different task context"
    assert json.loads(render_record(raw))["task"] == "Different task context"
