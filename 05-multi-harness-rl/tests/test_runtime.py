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


def test_native_openenv_partial_mask_reaches_trl():
    from runtime.bootstrap import activate
    activate()
    from data_agent_env.models import DataAgentRolloutResult, DataAgentTurn
    from data_agent_env.harness import to_trace_entries
    from trl.experimental.async_grpo.openenv_harness import _turns_from_trace
    result = DataAgentRolloutResult(rollout_type="train", turns=[DataAgentTurn(
        trainable=True, prompt_token_ids=[1, 2], completion_token_ids=[3, 4],
        per_token_logps=[-.2, -.4], loss_mask=[0, 0, 1, 0])])
    assert _turns_from_trace(to_trace_entries(result))[0].output_mask == [1, 0]
