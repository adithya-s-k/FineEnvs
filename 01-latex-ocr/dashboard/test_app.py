"""Regression for the missing reward/loss charts in the retained archive."""

import json
import sqlite3

from app import _fetch_metric_logs, fetch_metric_logs


def test_interleaved_training_metrics_survive_dashboard_sampling():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE metrics (run_id TEXT, timestamp TEXT, step INTEGER, metrics BLOB)"
    )
    rows = []
    for step in range(3000):
        for offset in range(8):
            metrics = {"profiling/generate": 0.1}
            if offset == 1:
                metrics = {"train/reward": step / 3000, "train/loss": -0.01}
            elif offset > 1:
                metrics = {}
            rows.append(("run", f"{step * 8 + offset:06}", step, json.dumps(metrics)))
    conn.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?)", rows)
    cursor = conn.cursor()
    old = _fetch_metric_logs(cursor, ("run_id", "run"), 3000, scalar_only=True)
    assert not any("train/reward" in row for row in old)
    displayed = fetch_metric_logs(cursor, ("run_id", "run"), 3000, scalar_only=True)
    rewards = [row for row in displayed if "train/reward" in row]
    assert len(rewards) == 3000
    assert rewards[0]["step"] == 0
    assert rewards[-1]["step"] == 2999
    assert rewards[-1]["train/reward"] == 2999 / 3000
    assert all(row["train/loss"] == -0.01 for row in rewards)
    assert conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 24000
    conn.close()
