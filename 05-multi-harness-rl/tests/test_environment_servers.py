"""Verify the deployment boundary without paying for a sandbox."""

import json
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.contract


def exchange(ws, kind, data):
    ws.send_json({"type": kind, "data": data})
    message = ws.receive_json()
    assert message["type"] == "observation", message
    return message["data"]["observation"]


def tool(ws, name, **arguments):
    observation = exchange(
        ws, "step", {"type": "call_tool", "tool_name": name, "arguments": arguments}
    )
    assert not observation.get("error"), observation
    result = observation["result"]
    if isinstance(result, str):
        return json.loads(result)
    return result.get("data", result)


def test_whitebox_websocket_retains_episode_and_closes_sandbox(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "false")
    from fastapi.testclient import TestClient
    from smoldataenv_whitebox import server

    monkeypatch.setattr(
        server, "task_by_name", lambda split, name: {"folder": "/test/tasks/demo"}
    )
    monkeypatch.setattr(
        server.BashEnvironment, "reset", lambda self, folder: "Solve demo"
    )
    monkeypatch.setattr(
        server.WhiteboxEnvironment,
        "get_task",
        lambda self, split, index: {"name": "demo"},
    )
    sandbox = Mock()

    def grade(self):
        self._calls, self._correctness = 15, 1.0
        self._sandbox = sandbox
        self._close()
        return 1.05

    monkeypatch.setattr(server.BashEnvironment, "get_reward", grade)
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
        with client.websocket_connect("/ws") as ws:
            initial = exchange(ws, "reset", {"split": "test", "task_name": "demo"})
            assert initial["metadata"]["instruction"] == "Solve demo"
            selected = tool(ws, "start_task", split="test", index=0)
            assert selected["instruction"] == "Solve demo"
            grade = tool(ws, "grade")
            assert grade == {"reward": 1.05, "correctness": 1.0, "tool_calls": 15}
    sandbox.kill.assert_called_once()


def test_native_server_exports_typed_trace_and_closes_session(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "false")
    from fastapi.testclient import TestClient
    from openenv.core.harness import TrainingTrace, TrainingTurn
    from smoldataenv_opencode import server

    trace = TrainingTrace(
        turns=[
            TrainingTurn(
                node_id="agent",
                prompt_token_ids=[1],
                completion_token_ids=[2, 3],
                per_token_logps=[-0.1, -0.2],
                loss_mask=[0, 1, 0],
            )
        ]
    )
    session = Mock()
    session.fetch_training_trace.return_value = trace
    session.verify.return_value.env_reward = 1.0
    monkeypatch.setattr(
        server,
        "TaskFactory",
        Mock(return_value=Mock(create=Mock(return_value=session))),
    )
    monkeypatch.setattr(
        server, "task_by_name", lambda split, name: {"instruction": "demo"}
    )
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
        with client.websocket_connect("/ws") as ws:
            exchange(ws, "reset", {})
            result = tool(
                ws,
                "run_rollout",
                split="test",
                task_name="demo",
                model="model",
                sampling={"temperature": 0.8},
            )
            exported = TrainingTrace.model_validate(result["training_trace"])
            assert exported.turns[0].loss_mask == [0, 1, 0]
            assert result["correctness"] == 1.0
    session.close.assert_called_once()


def test_remote_native_capture_failure_stays_a_contract_error():
    from smoldataenv_opencode.client import RemoteSession

    session = RemoteSession.__new__(RemoteSession)
    session.result = {"capture_error": "mask does not match sampled tokens"}
    with pytest.raises(ValueError, match="mask does not match"):
        session.fetch_training_trace()


def test_harbor_metadata_routes_precede_root_ui(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "false")
    monkeypatch.setenv("SPACE_HOST", "smoke.hf.space")
    monkeypatch.setenv("OPENENV_DATASETS", "")
    monkeypatch.setenv("OPENENV_LLM_URL", "")
    monkeypatch.setenv("OPENENV_HARBOR_TRIALS_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    from smoldataenv_harbor import server

    trajectory = tmp_path / "trial" / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.0",
                "steps": [
                    {"source": "agent", "tool_calls": [{"tool_call_id": "call-1"}]}
                ],
            }
        )
    )
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
        assert "/{env_name}/task" in client.get("/openapi.json").json()["paths"]
        splits = client.get("/smoldataenv/splits")
        assert splits.status_code == 200
        assert set(splits.json()) == {"train", "test"}
        counted = client.get("/smoldataenv/trials/trial/tool-count")
        assert counted.status_code == 200
        assert counted.json() == {"native_tool_calls": 1}
