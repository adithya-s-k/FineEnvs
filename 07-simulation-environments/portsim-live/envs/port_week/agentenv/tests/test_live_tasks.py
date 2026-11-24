"""The task steps equal the agentenv-portsim plugin's: every qualifying week as its `tasks generate --live`, `--marine`,
`--wind` and `--lines` write it, with and without a cost cap (fixtures/plugin_tasks.json, sha256 per file, at v0.4.5
and, for the lines weeks and the uncapped files, at 83eca5c; the wind and lines weeks once they are built, and the lines
weeks as GPT-6 Astra's published lines sweep played them), the synthetic wind and lines weeks' steps, and the week and
wiring-noplay tasks of its portsim-live and portsim-marine bundles."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from berth_core import TaskPack
from lines_fixture import real_lines
from port_week_core import lines, marine, packs, wind
from wind_fixture import LINES_PACK_DIR, PACK_DIR, STORM_TASK

from port_week_agentenv import tasks
from port_week_agentenv.contract import LINES_ENV, LIVE_ENV, MARINE_ENV, WIND_ENV

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GENERATED = json.loads((FIXTURES / "plugin_tasks.json").read_text())
WIRING_TASK = "dock-24B-w07x1-busy-0"
WIND_BUILT = all(p.exists() for p in (packs.tasks_root() / packs.WIND_PACK / "tasks.jsonl", packs.references("wind")))
real_wind = pytest.mark.skipif(not WIND_BUILT and os.environ.get("PORTSIM_REQUIRE_WIND") != "1",
                               reason="wind weeks not built: run tools/build_wind.py")


def digests(env: str, weeks, variant: dict) -> dict[str, str]:
    return {t.task_id: hashlib.sha256(tasks.encode(tasks.steps(env, t, variant["episode_cap_usd"],
                                                                variant["hf_bill_to"]))).hexdigest() for t in weeks}


def qualifying(pack: TaskPack, task_ids: list[str]) -> list:
    return [t for t in pack.tasks if t.task_id in task_ids]


@pytest.mark.parametrize("variant", GENERATED["live"].values(), ids=GENERATED["live"].keys())
def test_each_live_week_is_the_plugins(variant):
    weeks = qualifying(packs.live_pack(), packs.live_task_ids())
    assert [t.split for t in weeks] == ["eval"] * 15
    assert digests(LIVE_ENV, weeks, variant) == variant["sha256"]


@pytest.mark.parametrize("variant", GENERATED["marine"].values(), ids=GENERATED["marine"].keys())
def test_each_marine_week_is_the_plugins(variant):
    assert digests(MARINE_ENV, qualifying(marine.pack(), marine.task_ids()), variant) == variant["sha256"]


@real_wind
@pytest.mark.parametrize("variant", GENERATED["wind"].values(), ids=GENERATED["wind"].keys())
def test_each_wind_week_is_the_plugins(variant):
    assert digests(WIND_ENV, qualifying(wind.pack(), wind.task_ids()), variant) == variant["sha256"]


def test_the_synthetic_wind_weeks_are_the_plugins():
    pack = TaskPack(PACK_DIR)
    [variant] = GENERATED["wind-fixtures"].values()
    assert digests(WIND_ENV, pack.tasks, variant) == variant["sha256"]
    assert tasks.encode(tasks.steps(WIND_ENV, pack.get(STORM_TASK), 5.0)) == (FIXTURES / "wind-storm.json").read_bytes()


@real_lines
@pytest.mark.parametrize("variant", GENERATED["lines"].values(), ids=GENERATED["lines"].keys())
def test_each_lines_week_is_the_plugins(variant):
    assert digests(LINES_ENV, qualifying(lines.pack(), lines.task_ids()), variant) == variant["sha256"]


def test_the_synthetic_lines_weeks_are_the_plugins():
    pack = TaskPack(LINES_PACK_DIR)
    [variant] = GENERATED["lines-fixtures"].values()
    assert digests(LINES_ENV, pack.tasks, variant) == variant["sha256"]
    steps = tasks.wiring_noplay(LINES_ENV, pack.get(STORM_TASK))
    assert [s["id"] for s in steps] == ["deploy", "load-task", "hide-port-notice", "agents-only", "agents-tools",
                                        "watches", "clock", "end-week", "grade"]


@pytest.mark.parametrize(("env", "pack"), [(LIVE_ENV, packs.live_pack), (MARINE_ENV, marine.pack)])
def test_the_bundle_tasks_are_the_plugins(env, pack):
    task = pack().get(WIRING_TASK)
    bundle = FIXTURES / "bundle" / env
    assert tasks.encode(tasks.steps(env, task, 5.0)) == (bundle / "week.json").read_bytes()
    assert tasks.encode(tasks.wiring_noplay(env, task)) == (bundle / "wiring-noplay.json").read_bytes()
    assert [s["id"] for s in tasks.wiring_noplay(env, task)] == [
        "deploy", "load-task", "hide-port-notice", "watches", "clock", "end-week", "grade"]
