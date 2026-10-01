import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoldataenv_whitebox.tasks import stage_task

from eval.evaluate import summarize, write_json


def test_fixed_split_is_disjoint_and_keeps_the_training_mix():
    train = json.loads((ROOT / "data/train.json").read_text())
    test = json.loads((ROOT / "data/test.json").read_text())
    assert len(train) == 1000 and len(test) == 250
    assert sum(row["difficulty"] == "medium" for row in train) == 400
    assert sum(row["difficulty"] == "hard" for row in train) == 600
    for key in ("source", "notebook", "instruction_sha256"):
        assert not {t[key] for t in train} & {t[key] for t in test}


def test_ungraded_is_excluded_from_pass_at_one_but_visible_in_coverage():
    result = summarize(
        [
            {"correctness": 1, "tool_calls": 6},
            {"correctness": 0},
            {"correctness": None},
        ],
        4,
    )
    assert result["pass_at_1_observed"] == 0.5
    assert result["coverage"] == 0.5
    assert result["mean_tool_calls"] == 6
    assert result["tool_count_coverage"] == 1
    assert summarize([], 4)["pass_at_1_observed"] is None


def test_staging_does_not_upload_gold_or_leave_tokens_in_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-download-token")
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment/pull_bucket.py").write_text("print('download')")
    (tmp_path / "task.toml").write_text(
        '[environment.env]\nHF_TOKEN="${HF_TOKEN}"\n[verifier.env]\nEXPECTED_ANSWER="hidden-gold"\n'
    )
    writes, commands = [], []
    sandbox = SimpleNamespace(
        write_text=lambda *x: writes.append(x),
        exec=lambda *x, **kw: commands.append((x, kw)) or SimpleNamespace(exit_code=0),
    )
    stage_task(sandbox, tmp_path)
    assert "hidden-gold" not in str(writes) + str(commands)
    assert "test-download-token" not in str(writes)
    assert commands[0][1]["envs"] == {"HF_TOKEN": "test-download-token"}


def test_atomic_results_are_readable_after_replacement(tmp_path):
    target = tmp_path / "pairs" / "a.json"
    write_json(target, {"correctness": None})
    write_json(target, {"correctness": 1})
    assert json.loads(target.read_text())["correctness"] == 1
    assert not target.with_suffix(".tmp").exists()
