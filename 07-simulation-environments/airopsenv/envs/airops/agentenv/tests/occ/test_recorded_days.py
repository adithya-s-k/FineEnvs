"""The recorded days in fixtures/runs, the v2 board's one-agent run of 2025-01-11 (GPT-6.1 Sol, reward 0.819) and the
v3 board's three-desk run of the same date (reward 0.853), played back through a fresh OccDay (OccDay3) on the fixture
pack (airops_core.occ.replay): every call the env logged, in the order it received them, each result checked against
its recorded digest, the advisories at their places and the rest of the day as end-day ran it; then the whole record
the env served, grade, audit, digest and v3's passengers included, against the recording."""

import gzip
import json

import pytest
from airops_core.occ.day3 import OccDay3
from airops_core.occ.pack import Pack, Pack3
from airops_core.occ.replay import replay
from paths import OCC3_PACK, OCC_PACK, RUNS

BOARD = RUNS / "occ-board-sol"
BOARD3 = RUNS / "occ3-board-sol"


@pytest.mark.slow
def test_the_recorded_day_replays_exactly_to_the_record_it_served():
    row = json.loads((BOARD / "results.jsonl").read_text())
    recorded = json.loads(gzip.decompress((BOARD / f"{row['day']}.gz").read_bytes()))
    again = replay(Pack(OCC_PACK).world(recorded["task_id"]), recorded)
    assert again.exact and len(recorded["calls"]) == 109
    assert json.loads(json.dumps(again.day.data(), default=str)) == recorded
    assert (again.day.grade["reward"], again.day.grade["reward_L"]) == (row["reward"], row["reward_L"]) == (
        0.818584, 0.751209)


@pytest.mark.slow
def test_the_recorded_three_desk_day_with_passengers_replays_exactly_to_the_record_it_served():
    row = json.loads((BOARD3 / "results.jsonl").read_text())
    recorded = json.loads(gzip.decompress((BOARD3 / f"{row['day']}.gz").read_bytes()))
    again = replay(Pack3(OCC3_PACK).world(recorded["task_id"]), recorded, OccDay3)
    assert again.exact and len(recorded["calls"]) == 581
    assert {c["desk"] for c in recorded["calls"]} == {"crew", "customer", "ops"}
    assert json.loads(json.dumps(again.day.data(), default=str)) == recorded
    assert (again.day.grade["reward"], again.day.grade["reward_L"]) == (row["reward"], row["reward_L"]) == (
        0.852972, 0.492579)
