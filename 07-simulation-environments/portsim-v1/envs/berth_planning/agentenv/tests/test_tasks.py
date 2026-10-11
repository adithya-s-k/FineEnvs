"""The task steps equal the agentenv-portsim plugin's at v0.4.5: every dock-v1-eval task as its `tasks generate` wrote
it (fixtures/eval_tasks.json, sha256 per file; one file in full), and the three wiring tasks of its portsim bundle."""

import hashlib
import json
from pathlib import Path

import pytest
from berth_core import load_pack

from berth_agentenv import tasks

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GENERATED = json.loads((FIXTURES / "eval_tasks.json").read_text())
PACK = load_pack()
EVAL = [t for t in PACK.tasks if t.split == "eval"]


@pytest.mark.parametrize("variant", GENERATED.values(), ids=GENERATED.keys())
def test_each_eval_task_is_the_plugins(variant):
    files = {t.task_id: tasks.encode(tasks.steps(t, variant["episode_cap_usd"], variant["hf_bill_to"])) for t in EVAL}
    assert {task_id: hashlib.sha256(body).hexdigest() for task_id, body in files.items()} == variant["sha256"]


def test_the_first_eval_task_in_full():
    task = EVAL[0]
    assert task.task_id == "dock-24B-w06x1-busy-0"
    assert tasks.encode(tasks.steps(task, 5.0)) == (FIXTURES / f"{task.task_id}.json").read_bytes()


def test_the_wiring_tasks_are_the_plugins():
    wiring = tasks.wiring_tasks(PACK.get(tasks.WIRING_TASK))
    assert wiring == {name: json.loads((FIXTURES / "bundle" / f"{name}.json").read_text())
                      for name in ("smoke", "wiring-infeasible", "wiring-nosubmit")}
