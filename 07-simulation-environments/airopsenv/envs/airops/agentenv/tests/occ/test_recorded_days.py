"""The recorded day in fixtures/runs, the board's one-agent run of 2025-01-11 (GPT-6.1 Sol, reward 0.819), played
back through a fresh OccDay on the fixture pack (airops_core.occ.replay): every call the env logged, in the order it
received them, each result checked against its recorded digest, the advisories at their places and the rest of the day
as end-day ran it; then the whole record the env served, grade, audit and digest included, against the recording."""

import gzip
import json

import pytest
from airops_core.occ.pack import Pack
from airops_core.occ.replay import replay
from paths import OCC_PACK, RUNS

BOARD = RUNS / "occ-board-sol"


@pytest.mark.slow
def test_the_recorded_day_replays_exactly_to_the_record_it_served():
    row = json.loads((BOARD / "results.jsonl").read_text())
    recorded = json.loads(gzip.decompress((BOARD / f"{row['day']}.gz").read_bytes()))
    again = replay(Pack(OCC_PACK).world(recorded["task_id"]), recorded)
    assert again.exact and len(recorded["calls"]) == 109
    assert json.loads(json.dumps(again.day.data(), default=str)) == recorded
    assert (again.day.grade["reward"], again.day.grade["reward_L"]) == (row["reward"], row["reward_L"]) == (
        0.818584, 0.751209)
