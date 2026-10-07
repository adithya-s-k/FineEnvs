from __future__ import annotations

import json

import pytest
from conftest import FIXTURE_RELEASE
from retroenv.training import RetroRouteTrainingEnv
from retroenv.verifier import known_routes_submission


@pytest.mark.skipif(not FIXTURE_RELEASE.exists(), reason="tests/fixtures/mini-release is not built")
def test_training_factory_adapter_uses_the_same_terminal_score(tmp_path):
    trace = tmp_path / "episodes.jsonl"
    env = RetroRouteTrainingEnv(FIXTURE_RELEASE, trace_path=trace)
    task = next(t for t in env.store.tasks("train") if t.variant == "standard")
    index = env.store.tasks("train").index(task)
    opening = env.reset(split="train", index=index)
    assert "reference_routes" not in opening[0]["text"]
    result = env.emit_routes(known_routes_submission(task, env.store.stock(task.stock_id)))
    assert json.loads(result[0]["text"])["score"]["valid"] is True
    assert env.get_reward() == 1.0
    assert trace.exists()
