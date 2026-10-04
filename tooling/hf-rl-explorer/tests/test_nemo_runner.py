import json

import pytest

from app import nemo_runner as runner
from app.envs import nemogym


def row():
    return {"responses_create_params": {"input": [{"role": "user", "content": "Look up A."},
                 {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Previous reasoning"}]}],
                "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]},
            "expected_action": {"type": "function_call", "name": "lookup", "arguments": '{"id":"PRIVATE-ANSWER"}'}}


def test_responses_input_preserved_and_answer_withheld():
    data = row()
    out = runner.responses_request(data)
    assert out["input"] == data["responses_create_params"]["input"]
    assert "PRIVATE-ANSWER" not in json.dumps(out)


def test_provider_hosted_tools_cannot_execute():
    data = row()
    data["responses_create_params"]["tools"] = [{"type": "mcp", "server_url": "https://example.com", "require_approval": "never"}]
    with pytest.raises(ValueError, match="Only function"):
        runner.responses_request(data)


def test_native_comparator_scores_arguments_and_calls_before_text():
    data = row()
    agent = nemogym.agent_of({}, "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1")
    expected = runner.validate(data, agent)
    response = {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Looking it up"}]},
                           {"type": "function_call", "name": "lookup", "arguments": '{"id":"PRIVATE-ANSWER"}'}]}
    assert runner.score(expected, runner.prediction(response))[0] == 1
    response["output"][1]["arguments"] = '{"id":"wrong"}'
    assert runner.score(expected, runner.prediction(response))[0] == 0
    response["output"][1]["arguments"] = 'not json'
    assert runner.score(expected, runner.prediction(response))[0] == 0


def test_invalid_reference_does_not_leak_through_errors():
    data = row()
    data["expected_action"] = {"type": "PRIVATE-ANSWER"}
    with pytest.raises(ValueError) as e:
        runner.validate(data, nemogym.agent_of({}, "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1"))
    assert "PRIVATE-ANSWER" not in str(e.value)


def test_unknown_nemo_server_stays_native():
    with pytest.raises(ValueError, match="not hosted"):
        runner.validate(row(), {"known": True, "resources_server": "another_server"})


def test_provider_failure_inside_http_200_is_not_a_zero_reward():
    with pytest.raises(ValueError, match="did not produce a prediction"):
        runner.prediction({"status": "failed", "error": {"message": "invalid schema"}, "output": []})
    with pytest.raises(ValueError, match="invalid Responses"):
        runner.prediction({"message": "not a response"})


def test_invalid_strict_declaration_is_relaxed_without_changing_parameters():
    data = row()
    tool = data["responses_create_params"]["tools"][0]
    tool["strict"] = True
    original = json.dumps(data)
    request = runner.responses_request(data)
    assert request["tools"][0]["strict"] is False
    assert request["tools"][0]["parameters"] == tool["parameters"]
    assert json.dumps(data) == original
    tool["parameters"].update(properties={}, required=[], additionalProperties=False)
    assert runner.responses_request(data)["tools"][0]["strict"] is True
