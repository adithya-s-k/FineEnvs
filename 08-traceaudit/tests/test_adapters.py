import copy
import json

import pytest
from trace_audit.core import DataError, TraceAuditEnv, render_record, validate_record
from trace_audit.core.baselines import last_shell_nonzero_exit, last_tool_error, tool_call_count
from trace_audit.core.records import validate_collection

from adapters.common import chat_to_atif
from adapters.external import convert as external
from adapters.swe_rebench import convert as swe
from adapters.tau2 import convert as tau
from adapters.tau2 import convert_results, db_only_task
from tools.data import admission_report


def test_tool_result_mapping_preserves_malformed_calls_and_recovery():
    trajectory = chat_to_atif(
        [
            {"role": "user", "content": "Task"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "bad", "function": {"name": "run", "arguments": "{broken"}}],
            },
            {"role": "tool", "tool_call_id": "bad", "content": "Invalid args", "is_error": True},
            {"role": "assistant", "content": "Recovered."},
        ]
    )
    step = trajectory["steps"][1]
    assert step["tool_calls"][0]["arguments"] == {"raw_arguments": "{broken"}
    assert step["observation"]["results"][0]["extra"]["error"]["is_error"]
    assert trajectory["extra"]["original_message_index_to_step_id"]["2"] == 2
    with pytest.raises(DataError):
        chat_to_atif([{"role": "tool", "content": "x", "tool_call_id": "orphan"}])


def tau_row():
    return {
        "domain": "airline",
        "task": {
            "id": "7",
            "evaluation_criteria": {
                "reward_basis": ["DB"],
                "actions": [{"name": "book_reservation"}],
            },
        },
        "simulation": {
            "id": "trial",
            "messages": [
                {"role": "user", "content": "Update it."},
                {"role": "assistant", "content": "Done."},
            ],
            "reward_info": {"reward": 1, "db_check": {"db_match": False, "db_reward": 1}},
        },
        "instruction": "Update it.",
        "policy": "Confirm before updating.",
    }


def test_tau_uses_end_state_not_grade_or_path():
    source = tau_row()
    row = tau(source, revision="a" * 40)
    assert row["outcome"]["failed"] is True
    source["db_verification"] = {
        "completed": True,
        "initial_db_hash": "b" * 64,
        "expected_db_hash": "a" * 64,
        "actual_db_hash": "a" * 64,
    }
    source["simulation"]["reward_info"]["reward"] = 0
    source["task"]["evaluation_criteria"]["actions"] = [{"name": "different_valid_route"}]
    assert tau(source, revision="a" * 40)["outcome"]["failed"] is False
    source["task"]["evaluation_criteria"]["reward_basis"] = ["DB", "NL_ASSERTION"]
    assert tau(source, revision="a" * 40)["outcome"]["failed"] is None
    assert not db_only_task({})
    assert db_only_task(
        {"evaluation_criteria": {"reward_basis": ["DB", "COMMUNICATE"], "communicate_info": []}}
    )
    assert not db_only_task(
        {
            "evaluation_criteria": {
                "reward_basis": ["DB", "COMMUNICATE"],
                "communicate_info": ["say this"],
            }
        }
    )


def test_tau_native_tool_ids_and_retail_holdout():
    row = tau_row()
    row["domain"] = "retail"
    row["simulation"]["task_id"] = "7"
    row["simulation"]["messages"] = [
        {"role": "user", "content": "Update it."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c", "name": "update", "arguments": {}}],
        },
        {"role": "tool", "id": "c", "content": "Unavailable", "error": True},
    ]
    result = convert_results(
        {"tasks": [row["task"]], "simulations": [row["simulation"]]},
        revision="a" * 40,
        domain="retail",
        tools=[],
        policy="Policy",
    )[0]
    assert result["split"] == "test"
    assert result["trajectory"]["steps"][1]["observation"]["results"][0]["extra"]["error"][
        "is_error"
    ]


def test_swe_resolved_not_exit_or_generated_tests():
    raw = {
        "instance_id": "repo-123",
        "trajectory_id": "t",
        "trajectory": [{"role": "user", "content": "Fix bug"}],
        "resolved": 0,
        "exit_status": "submit",
        "pred_passes_gen_test": 100,
    }
    assert swe(raw, revision="a" * 40)["outcome"]["failed"] is True
    raw["resolved"] = None
    assert swe(raw, revision="a" * 40)["outcome"]["failed"] is None
    raw["resolved"] = 1
    assert swe(raw, revision="a" * 40)["outcome"]["failed"] is False


def test_external_reviews_and_mast_remain_analysis_only(rows):
    raw = {
        "original_id": "x",
        "task": rows[0]["task"],
        "trajectory": rows[0]["trajectory"],
        "task_family_id": "ext:x",
        "step_map": {"14": 2},
        "failures": [
            {"step_number": 14, "failure_category": "Invalid Invocation", "critical": True}
        ],
        "verified_failed": True,
    }
    assert (
        external(raw, dataset="microsoft/AgentRx", revision="a" * 40, license="CC-BY-4.0")[
            "outcome"
        ]["failed"]
        is None
    )
    raw.update(human_confirmed=True, reviewer="Rahul")
    row = external(raw, dataset="microsoft/AgentRx", revision="a" * 40, license="CC-BY-4.0")
    assert (
        row["split"] == "test"
        and row["failures"][0]["step_id"] == 2
        and row["outcome"]["failed"] is True
    )
    assert (
        external(raw, dataset="mcemri/MAST-Data", revision="a" * 40, license="CC-BY-4.0")[
            "outcome"
        ]["failed"]
        is None
    )
    del raw["step_map"]["14"]
    with pytest.raises(DataError):
        external(raw, dataset="microsoft/AgentRx", revision="a" * 40, license="CC-BY-4.0")


def test_ambiguous_taxonomy_decision_is_recorded_privately(rows):
    raw = {
        "original_id": "x",
        "task": rows[0]["task"],
        "trajectory": rows[0]["trajectory"],
        "task_family_id": "ext:x",
        "step_map": {"0": 2},
        "failures": [
            {
                "step_number": 0,
                "source_label": "Resource Not Found",
                "mapped_category": "external_cause",
            }
        ],
    }
    with pytest.raises(DataError, match="adjudication_reason"):
        external(raw, dataset="PatronusAI/TRAIL", revision="a" * 40, license="MIT")
    raw["failures"][0]["adjudication_reason"] = (
        "Service resource was unavailable independently of the agent's action."
    )
    result = external(raw, dataset="PatronusAI/TRAIL", revision="a" * 40, license="MIT")
    assert result["trajectory"]["extra"]["taxonomy_adjudications"]
    assert "taxonomy_adjudications" not in raw["trajectory"]["extra"]


def swe_row(messages=None):
    return {
        "instance_id": "repo-1",
        "trajectory_id": "run-1",
        "resolved": 0,
        "trajectory": messages or [{"role": "user", "content": "Fix this issue."}],
    }


@pytest.mark.parametrize("code,error", [(0, False), (1, True), (2, True), (-9, True)])
def test_swe_status_footer_normalized(code, error):
    messages = [
        {"role": "user", "content": "Fix this issue."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c", "function": {"name": "execute_bash", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": "c",
            "content": f"Output\n[Command finished with exit code {code}]",
        },
    ]
    raw = swe_row(messages)
    result = swe(raw, revision="a" * 40)
    assert last_shell_nonzero_exit(result) is error
    assert not last_tool_error(result)
    status = result["trajectory"]["steps"][1]["observation"]["results"][0]["extra"][
        "command_status"
    ]
    assert status == {"exit_code": code, "nonzero_exit": error}
    assert "is_error" not in raw["trajectory"][2]  # Converter does not mutate caller input.


@pytest.mark.parametrize(
    "name,content",
    [
        ("str_replace_editor", "[Command finished with exit code 2]"),
        ("execute_bash", "Example: [Command finished with exit code 2]"),
        ("execute_bash", "[Command finished with exit code 2]\nThis is a quoted example."),
        ("execute_bash", "an error occurred"),
    ],
)
def test_swe_does_not_guess_errors_from_arbitrary_text(name, content):
    result = swe(
        swe_row(
            [
                {"role": "user", "content": "Fix it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c", "function": {"name": name, "arguments": {}}}],
                },
                {"role": "tool", "tool_call_id": "c", "content": content},
            ]
        ),
        revision="a" * 40,
    )
    assert not last_tool_error(result)


def test_swe_tool_failure_is_not_an_outcome_label():
    raw = swe_row(
        [
            {"role": "user", "content": "Fix it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c", "function": {"name": "execute_bash", "arguments": {}}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": "Long output\n[Command finished with exit code 2]",
            },
        ]
    )
    raw["resolved"] = 1
    result = swe(raw, revision="a" * 40)
    assert result["outcome"]["failed"] is False
    assert last_shell_nonzero_exit(result) and not last_tool_error(result)


def external_row(rows):
    return {
        "original_id": "1",
        "task_family_id": "1",
        "task": rows[0]["task"],
        "trajectory": rows[0]["trajectory"],
        "step_map": {},
        "failures": [],
    }


def test_external_local_ids_namespaced_but_explicit_shared_tasks_still_checked(rows):
    raw = external_row(rows)
    agent = external(raw, dataset="microsoft/AgentRx", revision="a" * 40, license="CC-BY-4.0")
    mast = external(raw, dataset="mcemri/MAST-Data", revision="a" * 40, license="CC-BY-4.0")
    validate_collection([agent, mast])
    assert agent["group"] != mast["group"]
    raw["canonical_task_family_id"] = "shared-task:123"
    agent = external(raw, dataset="microsoft/AgentRx", revision="a" * 40, license="CC-BY-4.0")
    mast = external(raw, dataset="mcemri/MAST-Data", revision="a" * 40, license="CC-BY-4.0")
    with pytest.raises(DataError, match="family"):
        validate_collection([agent, mast])


def test_telecom_user_tools_preserve_actor_and_links():
    raw = tau_row()
    raw.update(domain="telecom")
    raw["task"]["evaluation_criteria"]["reward_basis"] = ["ENV_ASSERTION"]
    raw["simulation"]["messages"] = [
        {
            "role": "user",
            "content": "Check my device.",
            "tool_calls": [{"id": "same", "name": "restart_device", "arguments": {}}],
        },
        {"role": "tool", "id": "same", "requestor": "user", "content": "Restarted"},
        {
            "role": "assistant",
            "content": "Check service",
            "tool_calls": [{"id": "same", "name": "get_status", "arguments": {}}],
        },
        {
            "role": "tool",
            "id": "same",
            "requestor": "assistant",
            "content": "Unavailable",
            "error": True,
        },
    ]
    result = tau(raw, revision="a" * 40)
    assert result["outcome"]["failed"] is None
    public = json.loads(render_record(result))["trajectory"]["steps"]
    user = public[0]
    assert user["role"] == "user" and "tool_calls" not in user
    assert user["participant_tool_calls"][0]["requestor"] == "user"
    assert user["observations"][0]["call_id"] == user["participant_tool_calls"][0]["id"]
    assert public[1]["observations"][0]["call_id"] == public[1]["tool_calls"][0]["id"]
    from tools.validate_harbor import prepare

    prepare()
    from harbor.utils.trajectory_validator import TrajectoryValidator

    assert TrajectoryValidator().validate(result["trajectory"], validate_images=False)


@pytest.mark.parametrize(
    "stop", ["max_steps", "too_many_errors", "timeout", "infrastructure_error"]
)
def test_early_stops_remain_unlabelled_and_are_counted(stop, rows):
    raw = tau_row()
    raw["simulation"].update(termination_reason=stop, reward_info={"reward": 0})
    row = tau(raw, revision="a" * 40)
    assert row["outcome"]["failed"] is None
    reason = "unverified_early_stop:" + stop
    report = admission_report([row])
    assert report["exclusions"][0]["reason"] == reason and report["exclusions"][0]["n"] == 1
    # Runtime uses the same exclusion reasons as offline preparation.
    eligible = copy.deepcopy(rows[0])
    eligible["split"] = row["split"]
    env = TraceAuditEnv([row, eligible], split=row["split"], allow_fixtures=True)
    assert env.exclusions[(row["source"]["dataset"], reason)] == 1
    # Independent final-state evidence can establish success even after an early stop.
    raw["db_verification"] = {
        "completed": True,
        "expected_db_hash": "a" * 64,
        "actual_db_hash": "a" * 64,
    }
    verified = tau(raw, revision="a" * 40)
    assert verified["outcome"]["failed"] is False


@pytest.mark.parametrize(
    "command,code",
    [("grep missing file.txt", 1), ("pytest tests/test_new_case.py", 1), ("python broken.py", 2)],
)
def test_nonzero_status_is_evidence_not_a_confirmed_tool_error(command, code):
    row = swe(
        {
            "trajectory_id": "t",
            "instance_id": "repo-1",
            "resolved": 1,
            "trajectory": [
                {"role": "user", "content": "Investigate"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c",
                            "function": {"name": "execute_bash", "arguments": {"command": command}},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c",
                    "content": f"Output\n[Command finished with exit code {code}]",
                },
            ],
        },
        revision="a" * 40,
    )
    assert row["outcome"]["failed"] is False
    assert not last_tool_error(row) and last_shell_nonzero_exit(row)
    status = json.loads(render_record(row))["trajectory"]["steps"][1]["observations"][0][
        "command_status"
    ]
    assert status == {"exit_code": code, "nonzero_exit": True}
    row["trajectory"]["steps"][1]["observation"]["results"][0]["extra"]["command_status"][
        "nonzero_exit"
    ] = False
    with pytest.raises(DataError, match="Command status"):
        validate_record(row)


def test_explicit_tool_errors_still_normalize_independently_of_exit_code():
    raw = {
        "trajectory_id": "t",
        "instance_id": "repo-1",
        "resolved": 0,
        "trajectory": [
            {"role": "user", "content": "Task"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c", "function": {"name": "execute_bash", "arguments": {}}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": "Invocation rejected",
                "is_error": True,
            },
        ],
    }
    assert last_tool_error(swe(raw, revision="a" * 40))


def test_call_count_includes_user_and_embedded_calls():
    raw = tau_row()
    raw["domain"] = "telecom"
    raw["task"]["evaluation_criteria"]["reward_basis"] = ["ENV_ASSERTION"]
    raw["simulation"]["messages"] = [
        {
            "role": "user",
            "content": "Task",
            "tool_calls": [{"id": "u", "name": "restart", "arguments": {}}],
        },
        {"role": "tool", "id": "u", "requestor": "user", "content": "OK"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "a", "name": "check", "arguments": {}}],
        },
        {"role": "tool", "id": "a", "content": "OK"},
    ]
    row = tau(raw, revision="a" * 40)
    assert tool_call_count(row) == 2
    child = copy.deepcopy(row["trajectory"])
    child["trajectory_id"] = "delegate"
    row["trajectory"]["subagent_trajectories"] = [child]
    validate_record(row)
    assert tool_call_count(row) == 4
