import copy
import hashlib
import json
import sqlite3

import pytest
from build_comparison import build, merge_scalars, training_segments


def record(log_step, optimizer_step, timestamp, reward=0.5):
    return {
        "id": log_step,
        "step": log_step,
        "timestamp": timestamp,
        "metrics": {"train/global_step": optimizer_step, "train/reward": reward},
    }


def test_async_timestamps_do_not_create_restarts_but_step_resets_do():
    rows = [
        record(12, 3, "2026-07-23T00:00:02"),
        record(10, 1, "2026-07-23T00:00:03"),
        record(20, 1, "2026-07-23T00:00:04"),
        record(11, 2, "2026-07-23T00:00:01"),
    ]
    segments = training_segments(rows)
    assert [[r["metrics"]["train/global_step"] for r in s] for s in segments] == [
        [1, 2, 3],
        [1],
    ]


def test_optimizer_alignment_and_repeated_baselines_preserve_raw_values():
    training = [record(501, 500, "2026-07-23T01:00:00", 0.7)]
    baseline = {
        "step": 0,
        "timestamp": "2026-07-23T00:00:00",
        "metrics": {"eval/test_reward": 0.4},
    }
    evaluation = [
        baseline,
        copy.deepcopy(baseline),
        {
            "step": 500,
            "timestamp": "2026-07-23T01:00:01",
            "metrics": {"eval/test_reward": 0.6},
        },
    ]
    original = copy.deepcopy((training, evaluation))
    merged = merge_scalars(training, evaluation)
    assert sorted(merged) == [0, 500]
    assert merged[500]["metrics"]["train/reward"] == 0.7
    assert merged[500]["metrics"]["eval/test_reward"] == 0.6
    assert merged[500]["metrics"]["eval/reward_gain"] == pytest.approx(0.2)
    assert (training, evaluation) == original
    with pytest.raises(ValueError, match="Conflicting"):
        merge_scalars(
            [], [baseline, {**baseline, "metrics": {"eval/test_reward": 0.8}}]
        )


def test_build_keeps_model_identities_and_does_not_mutate_sources(tmp_path):
    source_dir = tmp_path / "sources"
    path = source_dir / "latex-ocr-colocate-5k-bucket" / "latex-ocr-colocate-5k.db"
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE configs(id INTEGER PRIMARY KEY, run_id TEXT, run_name TEXT, config TEXT);
        CREATE TABLE metrics(id INTEGER PRIMARY KEY, run_id TEXT, run_name TEXT, timestamp TEXT, step INTEGER, metrics TEXT);
        CREATE TABLE system_metrics(run_id TEXT,run_name TEXT,timestamp TEXT,metrics TEXT);
    """)
    # Full Trackio configs require created_at in the generated view.
    conn.execute("ALTER TABLE configs ADD COLUMN created_at TEXT")
    for name in ["qwen3-vl-2b", "glm-ocr", "gemma4-e2b"]:
        conn.execute(
            "INSERT INTO configs(run_id,run_name,config,created_at) VALUES (?,?,?,?)",
            (
                name,
                name,
                json.dumps(
                    {
                        "max_steps": 5000,
                        "max_grad_norm": 1.0,
                        "hub_token": "private-placeholder",
                    }
                ),
                "2026-07-23",
            ),
        )
        conn.execute(
            "INSERT INTO metrics(run_id,run_name,timestamp,step,metrics) VALUES (?,?,?,?,?)",
            (
                name,
                name,
                "2026-07-23",
                7,
                json.dumps({"train/global_step": 1, "train/reward": 0.5}),
            ),
        )
    conn.commit()
    conn.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = build(source_dir, tmp_path / "derived")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert {r["model"] for r in report["runs"]} == {
        "Qwen3-VL-2B",
        "GLM-OCR",
        "Gemma4-E2B",
    }
    assert len({r["run_id"] for r in report["runs"]}) == 3
    gemma = next(r for r in report["runs"] if r["model"] == "Gemma4-E2B")
    assert gemma["project"] == "latex-ocr-comparison"
    assert gemma["name"] == "Gemma4-E2B / unstable"
    assert gemma["recorded_config"]["max_grad_norm"] == 1.0
    assert all("hub_token" not in r["recorded_config"] for r in report["runs"])
    derived = sqlite3.connect(tmp_path / "derived" / "latex-ocr-comparison.db")
    assert derived.execute("SELECT DISTINCT step FROM metrics").fetchall() == [(1,)]
    assert all(
        "hub_token" not in json.loads(row[0])
        for row in derived.execute("SELECT config FROM configs")
    )
    derived.close()
