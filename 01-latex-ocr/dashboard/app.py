"""Serve the curated Trackio archive without dropping interleaved scalars."""

import os

import trackio
from trackio.sqlite_storage import SQLiteStorage

_fetch_metric_logs = SQLiteStorage._fetch_metric_logs_with_cursor
_get_projects = SQLiteStorage.get_projects
PROJECT_ORDER = [
    "latex-ocr-comparison",
    "latex-ocr-history",
    "latex-ocr-2b-overnight",
    "latex-ocr-eval",
]


def get_projects():
    # The hosted frontend opens the first project when no URL selection exists.
    # show(project=...) only changes the URL printed/opened by the Python client.
    priority = {name: index for index, name in enumerate(PROJECT_ORDER)}
    return sorted(
        _get_projects(), key=lambda name: (priority.get(name, len(priority)), name)
    )


def fetch_metric_logs(cursor, run_identity, max_points, *, scalar_only=False):
    # Trackio 0.32.2 samples every eighth row of the 24,000-row archive,
    # selecting profiling entries and skipping every reward/loss entry.
    # Read complete scalar history for this bounded archive. Media requests
    # retain Trackio's normal limits, and the underlying databases are unchanged.
    logs = _fetch_metric_logs(
        cursor,
        run_identity,
        None if scalar_only else max_points,
        scalar_only=scalar_only,
    )
    if scalar_only:
        return [row for row in logs if row.keys() - {"step", "timestamp"}]
    return logs


if __name__ == "__main__":
    SQLiteStorage._fetch_metric_logs_with_cursor = staticmethod(fetch_metric_logs)
    SQLiteStorage.get_projects = staticmethod(get_projects)
    os.environ.setdefault(
        "TRACKIO_PLOT_ORDER",
        "eval/test_reward,eval/reward_gain,train/reward,train/loss",
    )
    trackio.show(project="latex-ocr-comparison", open_browser=False)
