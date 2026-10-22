import json

import pytest

from airops_core.core.model import Task, hhmm, load_pack, parse_hhmm


def test_every_fixture_round_trips_byte_for_byte(pack, lines):
    assert len(pack) == len(lines) == 8
    for task, line in zip(pack, lines, strict=True):
        assert json.dumps(task.to_dict(), ensure_ascii=False, separators=(",", ":")) == line


def test_a_pack_loads_from_its_directory_or_its_file(tmp_path, fixtures_path):
    (tmp_path / "tasks.jsonl").write_text(fixtures_path.read_text())
    assert [t.task_id for t in load_pack(tmp_path)] == [t.task_id for t in load_pack(fixtures_path)]


def test_the_task_reads_as_designed(pack):
    task = next(t for t in pack if t.task_id == "b6-jfk-20230626-1400")
    assert (task.window, task.start, task.hub, task.tz) == ((840, 1200), 780, "JFK", "America/New_York")
    assert task.turn_for("E190") == 37 and task.turn_for("B738") == task.turn_all
    f = task.flight(5)
    assert f.id == 5 and f.flight.startswith("B6 ") and f.block > 0 and f.pax > 0
    assert task.caps["JFK:arr"][14] >= 0 and all(isinstance(h, int) for c in task.caps.values() for h in c)
    assert len({t.tail for t in task.tails}) == len(task.tails) == 50
    assert {f.family for f in task.flights} <= set(task.turn)


def test_earliest_never_precedes_the_start(pack):
    for task in pack:
        for f in task.flights:
            assert task.earliest(f, 0) >= task.start
            assert task.deadline(f, f.lb) >= task.earliest(f, f.lb)


@pytest.mark.parametrize(("minute", "text"), [(0, "00:00"), (845, "14:05"), (1439, "23:59"), (1440, "00:00+1"),
                                              (1510, "01:10+1"), (2880, "00:00+2")])
def test_times_read_as_local_clock_with_the_day_past_midnight(minute, text):
    assert hhmm(minute) == text
    assert parse_hhmm(text) == minute


@pytest.mark.parametrize("text", ["24:00", "14:5", "1405", "14:05+", "ab:cd", "14:60", "14:05+x"])
def test_bad_times_are_refused(text):
    with pytest.raises(ValueError):
        parse_hhmm(text)


def test_tasks_are_immutable(pack):
    with pytest.raises(AttributeError):
        pack[0].start = 0
    assert isinstance(pack[0], Task)
