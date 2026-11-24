"""Where the weeks live. The live weeks are berth_core's dock-v1 eval and train packs (BERTH_TASKS_DIR, else its own
tasks/); the marine, wind and lines packs and the references sit under PORT_WEEK_TASKS_DIR, else port_week/tasks/, so a
PORT_WEEK_TASKS_DIR holds the committed dock-v1-marine/, live/ and marine/ as well as the built wind and lines weeks.
They are never on BERTH_TASKS_DIR: the marine pack shares dock-v1-eval's task ids, and the lines pack the wind
pack's."""

import json
import os
from pathlib import Path

from berth_core import TaskPack, load_pack

TASKS_ROOT = Path(__file__).resolve().parents[2] / "tasks"
MARINE_PACK = "dock-v1-marine"
WIND_PACK = "dock-v1-wind"
LINES_PACK = "dock-v1-lines"


def tasks_root() -> Path:
    return Path(root) if (root := os.environ.get("PORT_WEEK_TASKS_DIR")) else TASKS_ROOT


def references(kind: str) -> Path:
    return tasks_root() / kind / "references.jsonl"


def live_pack() -> TaskPack:
    return load_pack()


def live_references() -> list[dict]:
    return [json.loads(line) for line in references("live").read_text().splitlines()]


def live_task_ids() -> list[str]:
    """The weeks a rolling re-planner on the week as known brings to the hindsight optimum, so 1.0 is reachable live."""
    return [r["task_id"] for r in live_references() if r["qualifies"]]
