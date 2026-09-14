"""Serve the curated Trackio archive with the combined comparison first."""

import os

import trackio
from trackio.sqlite_storage import SQLiteStorage

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


if __name__ == "__main__":
    SQLiteStorage.get_projects = staticmethod(get_projects)
    os.environ.setdefault(
        "TRACKIO_PLOT_ORDER",
        "eval/test_reward,eval/reward_gain,train/reward,train/loss",
    )
    trackio.show(project="latex-ocr-comparison", open_browser=False)
