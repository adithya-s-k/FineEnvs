"""Run with --contract after installing the tutorial dependencies."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytestmark = pytest.mark.contract


@pytest.fixture
def trace():
    from openenv.core.harness import TrainingTrace, TrainingTurn

    return TrainingTrace(
        turns=[
            TrainingTurn(
                node_id="agent",
                prompt_token_ids=[1, 2],
                completion_token_ids=[3, 4, 5],
                per_token_logps=[-0.1, -0.2, -0.3],
                loss_mask=[0, 0, 1, 0, 1],
            ),
            TrainingTurn(
                node_id="context",
                prompt_token_ids=[1],
                completion_token_ids=[2],
                per_token_logps=[-0.3],
                loss_mask=[0, 0],
            ),
        ]
    )


def test_native_trace_preserves_masks_and_passes_native_count_to_trl(trace):
    from smoldataenv_opencode.environment import TaskSession
    from trl.experimental.async_grpo.openenv_harness import _turns_from_training_trace

    from train.opencode import reward

    events = [{"type": "step_finish"}] + [
        {
            "type": "tool_use",
            "part": {"callID": str(i), "state": {"status": "completed"}},
        }
        for i in range(6)
    ]
    native = Mock()
    native.fetch_training_trace.return_value = trace
    native.fetch_trace.return_value = "\n".join(json.dumps(e) for e in events)
    result = TaskSession(native).fetch_training_trace()
    assert [t.loss_mask for t in result.turns] == [[0, 0, 1, 0, 1], [0, 0]]
    # Exercise the real typed consumer, including diagnostic metadata used by reward().
    records = _turns_from_training_trace(result)
    entries = result.to_trace_entries()
    assert records[0].output_mask == [1, 0, 1]
    assert records[1].output_mask == [0]
    assert reward(SimpleNamespace(env_reward=1.0, trace=entries)) == pytest.approx(
        1 + 1.5 / 21
    )


@pytest.mark.parametrize(
    "grade,calls,expected",
    [(None, 4, None), (0, 4, 0), (1, None, 1), (1, 0, 1), (1, 15, 1.05)],
)
def test_both_async_rewards_match(grade, calls, expected):
    from train.multi_harness import reward as harbor
    from train.opencode import reward as native

    outcome = SimpleNamespace(
        env_reward=grade, trace=[{"metadata": {"native_tool_calls": calls}}]
    )
    assert native(outcome) == expected
    assert harbor(outcome) == expected


def test_harbor_assignment_depends_on_prompt_not_seed(monkeypatch):
    import smoldataenv_harbor.environment as tutorial

    rows = [{"instruction": str(i)} for i in range(8)]
    args = SimpleNamespace(
        server="unused",
        data="prepared",
        vllm_url="unused",
        model="test",
        trials="trials",
    )
    monkeypatch.setattr(tutorial, "HarborEnv", Mock())
    constructor = Mock()
    monkeypatch.setattr(tutorial, "TaskSession", constructor)
    monkeypatch.setattr(
        "httpx.get",
        Mock(return_value=Mock(json=lambda: {"train": "prepared/datasets/train"})),
    )
    factory = tutorial.TaskFactory(args, rows, sampling={"temperature": 0.8})
    for seed in (0, 17, 9000):
        factory.create([{"role": "user", "content": "5"}], seed=seed)
        assert constructor.call_args.kwargs["task_index"] == 5
        assert constructor.call_args.kwargs["harness"] == "claude-code"
        assert constructor.call_args.kwargs["sampling"] == {"temperature": 0.8}
        assert constructor.call_args.kwargs["reward_key"] == "correctness,reward"


@pytest.mark.parametrize("grade", [0.0, 1.0])
def test_harbor_reads_both_published_grader_formats(grade):
    from openenv.harbor.rollout import _pick_reward

    assert _pick_reward({"reward": grade}, "correctness,reward")[0] == grade
    assert (
        _pick_reward({"correctness": grade, "submission": 1.0}, "correctness,reward")[0]
        == grade
    )


def test_native_setup_failure_closes_sandbox(monkeypatch):
    import smoldataenv_opencode.environment as tutorial

    monkeypatch.setenv("SANDBOX_VLLM_URL", "https://example.test")
    monkeypatch.setenv("SANDBOX_VLLM_KEY", "test-key")
    factory = tutorial.TaskFactory(
        SimpleNamespace(model="test"),
        [{"instruction": "task", "folder": "unused"}],
        sampling={"temperature": 0.8},
    )
    session = Mock()
    factory.factory = Mock()
    factory.factory.create.return_value = session
    monkeypatch.setattr(
        tutorial, "stage_task", Mock(side_effect=RuntimeError("staging"))
    )
    with pytest.raises(RuntimeError, match="staging"):
        factory.create([{"role": "user", "content": "task"}], seed=999)
    session.close.assert_called_once()
    session.start_agent.assert_not_called()


def test_whitebox_reward_and_cleanup(monkeypatch):
    import smoldataenv_whitebox.environment as tutorial

    env = tutorial.BashEnvironment()
    sandbox = Mock()
    env._sandbox, env._folder, env._calls = sandbox, Path("unused"), 15
    monkeypatch.setattr(tutorial, "grade_answer", lambda *args: 1.0)
    assert env.get_reward() == 1.05
    sandbox.kill.assert_called_once()
    assert env._sandbox is None


@pytest.mark.parametrize(
    "module", ["train.opencode", "train.multi_harness", "train.whitebox"]
)
@pytest.mark.parametrize("model", ["LiquidAI/LFM2.5-2.6B", "Qwen/Qwen3.5-2B"])
def test_nonthinking_prompt(module, model):
    import importlib

    tokenizer = importlib.import_module(module).tokenizer_for(model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say OK"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tail = prompt.rsplit("<|im_start|>assistant", 1)[-1]
    assert "</think>" in tail and tail.count("<think>") == tail.count("</think>")


def test_whitebox_public_tool_schemas_are_valid():
    import inspect

    from transformers.utils import get_json_schema

    from train.whitebox import BashEnvironment

    env = BashEnvironment()
    tools = [
        get_json_schema(method)["function"]["name"]
        for name, method in inspect.getmembers(env, inspect.ismethod)
        if not name.startswith("_") and name not in {"reset", "get_reward"}
    ]
    assert set(tools) == {
        "bash",
        "read",
        "write",
        "edit",
        "ls",
        "grep",
        "glob",
        "submit_solution",
    }


def test_inference_proxy_blocks_unauthorized_and_admin_routes():
    from fastapi.testclient import TestClient

    from jobs.inference_proxy import create_app

    with TestClient(create_app("test-secret")) as client:
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get(
                "/v1/models", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/v1/update_weights", headers={"Authorization": "Bearer test-secret"}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/update_weights", headers={"Authorization": "Bearer test-secret"}
            ).status_code
            == 404
        )


def test_public_config_signatures_accept_tutorial_keywords():
    import ast
    import inspect

    from trl import GRPOConfig
    from trl.experimental.async_grpo import AsyncGRPOConfig

    for name, cls in [
        ("whitebox", GRPOConfig),
        ("opencode", AsyncGRPOConfig),
        ("multi_harness", AsyncGRPOConfig),
    ]:
        tree = ast.parse((ROOT / "train" / f"{name}.py").read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == cls.__name__
        ]
        assert len(calls) == 1
        assert {kw.arg for kw in calls[0].keywords} <= set(
            inspect.signature(cls).parameters
        )


def test_whitebox_does_not_count_submission(monkeypatch):
    import time

    from train.whitebox import BashEnvironment

    env = BashEnvironment()
    env._sandbox, env._deadline, env._calls = Mock(), time.monotonic() + 60, 3
    env.submit_solution("42")
    assert env._calls == 3
    with pytest.raises(RuntimeError, match="finished"):
        env.bash("echo retry")
