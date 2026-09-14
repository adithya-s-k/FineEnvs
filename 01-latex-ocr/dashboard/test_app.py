"""Regression for the missing reward/loss charts in the retained archive."""

import json
import sqlite3

from trackio.sqlite_storage import SQLiteStorage


def test_interleaved_training_metrics_survive_dashboard_sampling():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE metrics (id INTEGER PRIMARY KEY, run_id TEXT, "
        "timestamp TEXT, step INTEGER, metrics BLOB)"
    )
    rows = []
    for step in range(3000):
        for offset in range(8):
            metrics = {"profiling/generate": 0.1}
            if offset == 1:
                metrics = {"train/reward": step / 3000, "train/loss": -0.01}
            elif offset == 2 and step % 500 == 0:
                metrics = {"eval/test_reward": 0.5 + step / 10000}
            elif offset > 1:
                metrics = {}
            rows.append(("run", f"{step * 8 + offset:06}", step, json.dumps(metrics)))
    conn.executemany(
        "INSERT INTO metrics (run_id, timestamp, step, metrics) VALUES (?, ?, ?, ?)",
        rows,
    )
    cursor = conn.cursor()
    displayed = SQLiteStorage._fetch_metric_logs_with_cursor(
        cursor, ("run_id", "run"), 3000, scalar_only=True
    )
    assert len(displayed) <= 3000
    rewards = [row for row in displayed if "train/reward" in row]
    assert len(rewards) > 2  # Include intermediate training values, not just endpoints.
    assert rewards[0]["step"] == 0
    assert rewards[-1]["step"] == 2999
    assert rewards[-1]["train/reward"] == 2999 / 3000
    assert all(row["train/loss"] == -0.01 for row in rewards)
    assert all(row["train/reward"] == row["step"] / 3000 for row in rewards)
    evaluations = [row for row in displayed if "eval/test_reward" in row]
    assert [row["step"] for row in evaluations] == list(range(0, 3000, 500))
    assert all(
        row["eval/test_reward"] == 0.5 + row["step"] / 10000 for row in evaluations
    )
    assert conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 24000
    conn.close()


def test_hosted_landing_view_keeps_comparison_first_without_hiding_other_projects(
    monkeypatch,
):
    import app

    available = [
        "latex-ocr-2b-overnight",
        "latex-ocr-eval",
        "latex-ocr-history",
        "latex-ocr-comparison",
        "future-results",
    ]
    monkeypatch.setattr(app, "_get_projects", lambda: available)
    assert app.get_projects()[0] == "latex-ocr-comparison"
    assert set(app.get_projects()) == set(available)
    assert available[0] == "latex-ocr-2b-overnight"
