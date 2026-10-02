"""Check complete evaluations and retry behavior without model or sandbox services."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import evaluate


@pytest.mark.parametrize("mode", ["whitebox", "blackbox"])
def test_evaluation_keeps_grades_and_retries_only_missing_pairs(
    mode, tmp_path, monkeypatch
):
    tasks = [
        {"name": "a", "difficulty": "medium", "instruction": "first task"},
        {"name": "b", "difficulty": "hard", "instruction": "second task"},
    ]
    monkeypatch.setattr(evaluate, "load_tasks", lambda *args: tasks)
    logger = Mock()
    monkeypatch.setitem(
        sys.modules, "trackio", SimpleNamespace(init=lambda **kw: logger)
    )
    harnesses = ["whitebox"] if mode == "whitebox" else list(evaluate.HARNESSES)
    missing = ("b", harnesses[-1])
    calls = []
    fail = True

    def episode(args, index, task, harness):
        pair = (task["name"], harness)
        calls.append(pair)
        if fail and pair == missing:
            raise RuntimeError("Sandbox temporarily unavailable")
        return {
            "correctness": float(index),
            "tool_calls": 15,
            "generated_tokens": 10,
            "prompt_tokens": 20,
        }

    selected = "whitebox_episode" if mode == "whitebox" else "harbor_episode"
    other = "harbor_episode" if mode == "whitebox" else "whitebox_episode"
    monkeypatch.setattr(evaluate, selected, episode)
    monkeypatch.setattr(
        evaluate, other, Mock(side_effect=AssertionError("Wrong evaluation interface"))
    )
    argv = [
        "evaluate.py",
        "--mode",
        mode,
        "--checkpoint",
        "checkpoint-100",
        "--tasks",
        "2",
        "--concurrency",
        "4",
        "--output",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2
    partial = json.loads((tmp_path / "summary.json").read_text())
    expected = 2 * len(harnesses)
    assert partial["graded"] == expected - 1
    assert partial["expected"] == expected
    assert set(partial["harnesses"]) == set(harnesses)
    saved_wrong = tmp_path / "pairs" / f"a--{harnesses[0]}.json"
    original = saved_wrong.read_bytes()

    fail = False
    calls.clear()
    evaluate.main()
    assert calls == [missing]
    assert saved_wrong.read_bytes() == original
    complete = json.loads((tmp_path / "summary.json").read_text())
    assert complete["graded"] == expected
    assert complete["coverage"] == 1.0
    assert complete["pass_at_1_observed"] == 0.5
    assert complete["combined_reward"] == pytest.approx(0.525)
    assert complete["mean_generated_tokens"] == 10
    assert complete["mean_prompt_tokens"] == 20
    assert complete["difficulty"]["medium"]["pass_at_1_observed"] == 0
    assert complete["difficulty"]["hard"]["pass_at_1_observed"] == 1
    logger.log.assert_called()

    calls.clear()
    evaluate.main()
    assert not calls
    argv[argv.index("checkpoint-100")] = "checkpoint-200"
    with pytest.raises(ValueError, match="different evaluation"):
        evaluate.main()
    assert not calls
