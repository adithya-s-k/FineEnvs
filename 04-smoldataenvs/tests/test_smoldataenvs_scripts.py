from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "04-smoldataenvs" / "scripts"


def load_script(name: str):
    """Load one script module without relying on package layout."""
    path = SCRIPTS / f"{name}.py"
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(f"test_{name}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(SCRIPTS))
        except ValueError:
            pass


def test_leading_greater_than_answers_are_values_but_commands_and_pipes_are_flagged(monkeypatch):
    rollout = load_script("rollout")

    class Runner:
        def __init__(self, stdout: str):
            self.stdout = stdout

        def run(self, bucket_prefix, code):
            return self.stdout, ""

    row = {"answer": ">50K", "reward_mode": "exact", "atol": 0.0, "rtol": 0.0, "bucket_prefix": "bucket/prefix"}
    seen_predictions = []

    def fake_grade(got_row, prediction):
        seen_predictions.append(prediction)
        return 1.0

    monkeypatch.setattr(rollout, "grade", fake_grade)

    for answer in [">50K", "> 2 Years", ">40hrs"]:
        result = rollout.rollout(Runner(f"notes\n{answer}\n"), row, "```python\nprint('ok')\n```")
        assert result["reward"] == 1.0
        assert result["prediction"] == answer
        assert result["stderr"] == ""

    assert seen_predictions == [">50K", "> 2 Years", ">40hrs"]

    for command_line in ["echo >50K", "printf '2'", "2 > answer.txt", "cat answer.txt", "$(cat answer.txt)",
                         "| 2.14", "| /tmp/2.14", "2.14 | tee answer.txt"]:
        result = rollout.rollout(Runner(command_line), row, "```python\nprint('ok')\n```")
        assert result == {
            "reward": 0.0,
            "ran": 1.0,
            "prediction": command_line,
            "stderr": "answer is a command, not a value",
        }


def test_reward_cache_is_tied_to_completion_list_object_not_reused_id(monkeypatch):
    train_grpo = load_script("train_grpo")
    monkeypatch.setattr(train_grpo, "id", lambda _obj: 7, raising=False)

    calls = []

    def fake_grade_batch(completions, **columns):
        calls.append((list(completions), list(columns["answer"])))
        return [
            {"reward": float(len(calls)), "ran": 1.0, "prediction": columns["answer"][i], "stderr": ""}
            for i in range(len(completions))
        ]

    monkeypatch.setattr(train_grpo, "_grade_batch", fake_grade_batch)

    first = ["same text"]
    second = ["same text"]

    assert train_grpo.reward_correct(first, answer=["first"], reward_mode=["exact"], atol=[0.0], rtol=[0.0], bucket_prefix=["p1"]) == [1.0]
    assert train_grpo.reward_ran(first, answer=["first"], reward_mode=["exact"], atol=[0.0], rtol=[0.0], bucket_prefix=["p1"]) == [1.0]
    assert train_grpo.reward_correct(second, answer=["second"], reward_mode=["exact"], atol=[0.0], rtol=[0.0], bucket_prefix=["p2"]) == [2.0]

    assert calls == [(["same text"], ["first"]), (["same text"], ["second"])]


def test_eval_summary_writes_none_rewards_as_ungraded(tmp_path, monkeypatch):
    eval_pass1 = load_script("eval_pass1")

    rows = [
        {"task_id": "none", "difficulty_tier": "easy", "answer": "A", "reward_mode": "exact", "atol": 0.0, "rtol": 0.0, "bucket_prefix": "p", "files": [], "question": "q"},
        {"task_id": "right", "difficulty_tier": "easy", "answer": "B", "reward_mode": "exact", "atol": 0.0, "rtol": 0.0, "bucket_prefix": "p", "files": [], "question": "q"},
        {"task_id": "wrong", "difficulty_tier": "medium", "answer": "C", "reward_mode": "exact", "atol": 0.0, "rtol": 0.0, "bucket_prefix": "p", "files": [], "question": "q"},
    ]

    class FakeDatasetModule:
        @staticmethod
        def load_dataset(dataset, split):
            return rows

    class Runner:
        def close(self):
            pass

    rewards = iter([None, 1.0, 0.0])

    def fake_rollout(runner, row, completion):
        reward = next(rewards)
        return {"reward": reward, "ran": 0.0 if reward is None else 1.0, "prediction": f"pred-{row['task_id']}", "stderr": ""}

    out_path = tmp_path / "eval_results.json"
    monkeypatch.setitem(sys.modules, "datasets", FakeDatasetModule)
    monkeypatch.setattr(eval_pass1, "SandboxRunner", Runner)
    monkeypatch.setattr(eval_pass1, "rollout", fake_rollout)
    monkeypatch.setattr(eval_pass1.sys, "argv", ["eval_pass1.py", "--dry-run"])
    monkeypatch.setenv("DRY_RUN_TASKS", "3")
    monkeypatch.setattr(eval_pass1, "OUT", str(out_path))

    eval_pass1.main()

    payload = json.loads(out_path.read_text())
    assert payload["summary"] == {
        "model": eval_pass1.MODEL,
        "split": eval_pass1.SPLIT,
        "n": 2,
        "ungraded": 1,
        "pass@1": 0.5,
        "by_tier": {"easy": 1.0, "medium": 0.0},
    }
    assert [record["reward"] for record in payload["records"]] == [None, 1.0, 0.0]


def test_eval_summary_with_nothing_graded_has_no_score(tmp_path, monkeypatch):
    eval_pass1 = load_script("eval_pass1")

    rows = [
        {"task_id": f"t{i}", "difficulty_tier": "easy", "answer": "A", "reward_mode": "exact", "atol": 0.0, "rtol": 0.0,
         "bucket_prefix": "p", "files": [], "question": "q"}
        for i in range(3)
    ]

    class FakeDatasetModule:
        @staticmethod
        def load_dataset(dataset, split):
            return rows

    class Runner:
        def close(self):
            pass

    def fake_rollout(runner, row, completion):
        return {"reward": None, "ran": 0.0, "prediction": "", "stderr": "sandbox unavailable"}

    out_path = tmp_path / "eval_results.json"
    monkeypatch.setitem(sys.modules, "datasets", FakeDatasetModule)
    monkeypatch.setattr(eval_pass1, "SandboxRunner", Runner)
    monkeypatch.setattr(eval_pass1, "rollout", fake_rollout)
    monkeypatch.setattr(eval_pass1.sys, "argv", ["eval_pass1.py", "--dry-run"])
    monkeypatch.setenv("DRY_RUN_TASKS", "3")
    monkeypatch.setattr(eval_pass1, "OUT", str(out_path))

    eval_pass1.main()

    summary = json.loads(out_path.read_text())["summary"]
    assert summary["n"] == 0
    assert summary["ungraded"] == 3
    assert summary["pass@1"] is None
    assert summary["by_tier"] == {}
