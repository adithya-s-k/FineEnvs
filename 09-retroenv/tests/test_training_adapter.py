from __future__ import annotations

import json

from retroenv.graph import routes_to_submission
from retroenv.training import RetroRouteTrainingEnv

from conftest import make_task


def test_training_factory_adapter_uses_same_terminal_score(tmp_path):
    task = make_task()
    tasks_dir = tmp_path / "tasks"
    stocks_dir = tmp_path / "stocks"
    tasks_dir.mkdir()
    stocks_dir.mkdir()
    (tasks_dir / "train.jsonl").write_text(json.dumps(task.to_dict()) + "\n")
    (stocks_dir / "test_stock.smi").write_text("CCO\nCC(=O)O\n")
    trace = tmp_path / "episodes.jsonl"
    env = RetroRouteTrainingEnv(tasks_dir, stocks_dir, trace_path=trace)
    opening = env.reset(split="train", index=0)
    assert "reference_routes" not in opening[0]["text"]
    submission = routes_to_submission(
        task.target_smiles,
        task.reference_routes,
        {"CCO", "CC(=O)O"},
    )
    result = env.emit_routes(submission)
    assert json.loads(result[0]["text"])["score"]["valid"] is True
    assert env.get_reward() == 1.0
    assert trace.exists()
