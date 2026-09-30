"""CPU integration checks against the fetched runtime and prepared task files."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pytestmark = pytest.mark.skipif(not (ROOT / ".runtime/trl/trl").exists() or not (ROOT / "prepared/ready.json").exists(),
                                reason="Run bootstrap.py and prepare.py for runtime checks")


def test_corrected_grader_returns_only_numeric_fields(monkeypatch):
    from prepare import GRADER_SHA256
    import hashlib
    path = next((ROOT / "prepared/datasets/test/tasks").glob("*/tests/grader.py"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == GRADER_SHA256
    module = runpy.run_path(str(path))
    monkeypatch.setenv("EXPECTED_ANSWER", "42")
    monkeypatch.setenv("REWARD_MODE", "numeric")
    for answer, expected in [("42", 1.0), ("17", 0.0), ("", 0.0)]:
        monkeypatch.setattr(sys, "stdin", io.StringIO(answer))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            module["main_json"]()
        result = json.loads(stream.getvalue())
        assert result["correctness"] == expected
        assert all(type(v) in {float, int} for v in result.values())


def test_native_task_apis_share_ids_and_do_not_expose_answers(monkeypatch):
    from runtime.bootstrap import activate
    activate()
    monkeypatch.setenv("DATA_AGENT_FROZEN_TASKS_DIR", str(ROOT / "prepared/datasets"))
    monkeypatch.setenv("DAYTONA_COMPARISON_RUN", str(ROOT / "prepared"))
    monkeypatch.setenv("WHITE_BOX_BASH_TASK_SOURCE", "harbor-frozen")
    from data_agent_env.tasks import rows_for
    from whitebox_bash.tasks import _load
    for split, count in (("train", 1000), ("test", 250)):
        native, whitebox = rows_for(split), _load(split)
        assert len(native) == len(whitebox) == count
        assert [t["task_id"] for t in native] == [t.metadata["source_name"] for t in whitebox]
        assert all("answer" not in t.public() and "metadata" not in t.public() for t in whitebox)


def test_worker_options_match_pinned_runtime():
    import ast
    import inspect
    from runtime.bootstrap import activate
    activate()
    from trl.experimental.async_grpo.async_rollout_worker import _AsyncRolloutLoop
    from trl.experimental.async_grpo.openenv_harness import _HarnessRolloutLoop
    from hard_curriculum_train import FiniteLoop
    names = set()
    for cls in (_AsyncRolloutLoop, _HarnessRolloutLoop, FiniteLoop):
        names.update(inspect.signature(cls.__init__).parameters)
    names.add("max_outstanding_rollouts")
    tree = ast.parse((ROOT / "train/blackbox.py").read_text())
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Worker":
            assert not {kw.arg for kw in call.keywords} - names


@pytest.mark.parametrize("mask", [[0, 0, 1, 0], [0, 0, 0, 0]])
def test_native_openenv_partial_mask_reaches_trl(mask):
    from runtime.bootstrap import activate
    activate()
    from data_agent_env.models import DataAgentRolloutResult, DataAgentTurn
    from runtime.native_capture import training_trace
    from trl.experimental.async_grpo.openenv_harness import _turns_from_training_trace
    result = DataAgentRolloutResult(rollout_type="train", turns=[DataAgentTurn(
        trainable=bool(any(mask)), prompt_token_ids=[1, 2], completion_token_ids=[3, 4],
        per_token_logps=[-.2, -.4], loss_mask=mask, capture_metadata={"node_id": "agent-0"})])
    trace = training_trace(result)
    assert len(trace.turns) == 1
    assert _turns_from_training_trace(trace)[0].output_mask == mask[2:]


@pytest.mark.parametrize("broken", ["eval", "mask", "duplicate", "fatal", "identity"])
def test_native_training_capture_rejects_invalid_records(broken):
    from runtime.bootstrap import activate
    activate()
    from data_agent_env.models import DataAgentRolloutResult, DataAgentTurn
    from runtime.native_capture import training_trace
    turn = DataAgentTurn(trainable=True, prompt_token_ids=[1], completion_token_ids=[2],
                        per_token_logps=[-.2], loss_mask=[0, 1], capture_metadata={"node_id": "a"})
    result = DataAgentRolloutResult(rollout_type="train", turns=[turn])
    if broken == "eval":
        result.rollout_type = "eval"
    elif broken == "mask":
        turn.loss_mask = []
    elif broken == "identity":
        turn.capture_metadata = {}
    elif broken == "duplicate":
        result.turns.append(turn)
    else:
        result.metadata["capture_findings"] = ["[FATAL] incomplete capture"]
    with pytest.raises((ValueError, KeyError)):
        training_trace(result)


@pytest.mark.parametrize("correctness,actions,expected", [(1.0, 5, 1.075), (0.0, 5, 0.0), (1.0, None, 1.0), (None, 5, None)])
def test_native_session_worker_preserves_reward_after_typed_copy(tmp_path, correctness, actions, expected):
    from runtime.bootstrap import activate
    activate()
    import queue
    import threading
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from accelerate import PartialState
    from data_agent_env.models import DataAgentRolloutResult, DataAgentTurn
    from data_agent_env.harness import DataAgentSession
    from openenv.core.harness import TrainableSession
    from trl.experimental.async_grpo.openenv_harness import _HarnessRolloutLoop
    from train.adapters import Session, rollout_reward

    PartialState()
    result = DataAgentRolloutResult(rollout_type="train", correctness=correctness,
        reward=correctness, metadata={"verified_native_actions": actions}, turns=[DataAgentTurn(
            trainable=True, prompt_token_ids=[1], completion_token_ids=[2, 3],
            per_token_logps=[-.2, -.4], loss_mask=[0, 1, 0], capture_metadata={"node_id": "a"})])
    client = MagicMock()
    client.run_rollout.return_value = result
    cfg = {"mode": "opencode", "efficiency_weight": .1, "tool_budget": 15, "output": str(tmp_path)}
    wrapped = Session(DataAgentSession(client, "train", 0, "task"), cfg, {"episode_id": "smoke"}, tmp_path)
    assert isinstance(wrapped, TrainableSession)
    policy = {}
    def build(*, sampling):
        policy.update(sampling)
        return SimpleNamespace(create=lambda *args, **kwargs: wrapped)
    loop = _HarnessRolloutLoop(harness_session_factory=build, rollout_reward_fn=rollout_reward,
        model_name="test", dataset=[{"prompt": [{"role": "user", "content": "task"}]}], reward_funcs=[],
        processing_class=MagicMock(eos_token_id=0, pad_token_id=0), rollout_buffer=queue.Queue(),
        model_version_value=SimpleNamespace(value=0), heartbeat_value=SimpleNamespace(value=0.),
        failed_event=threading.Event(), exception_info_queue=queue.Queue(), metrics_queue=queue.Queue(),
        max_inflight_tasks=2, temperature=.8)
    try:
        actual, metrics = loop._run_session([])
        if expected is None:
            assert actual[-1] is None
        else:
            assert actual[-1] == pytest.approx(expected)
        assert actual[2][0].completion_mask == [0, 1, 0]
        assert actual[2][0].old_log_probs == [0., -.2, -.4]
        assert policy["temperature"] == .8 and policy["top_p"] == 1.
        record = json.loads((tmp_path / "rollouts/smoke.json").read_text())
        assert record["reward"] == actual[-1]
        client.close.assert_called_once()
    finally:
        loop._session_pool.shutdown(wait=True)
        loop._loop.close()


@pytest.mark.parametrize("tasks,requested", [(2000, 1000), (2000, 2000), (5, 1000)])
def test_whitebox_update_limit_matches_trl_sampler(tasks, requested):
    from runtime.bootstrap import activate
    activate()
    from recipe import config, whitebox_step_limit
    from trl.trainer.utils import RepeatSampler
    cfg = config(mode="whitebox", max_steps=requested)
    accumulation = cfg["batch_size"] * cfg["gradient_accumulation_steps"]
    sampler = RepeatSampler(range(tasks), mini_repeat_count=cfg["num_generations"],
        batch_size=accumulation // cfg["num_generations"], repeat_count=accumulation, shuffle=False)
    available = len(list(sampler)) // (accumulation * accumulation)
    assert whitebox_step_limit(cfg, tasks) == min(requested, available)
    if tasks == 2000:
        assert available == 1000
