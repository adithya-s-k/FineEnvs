"""The weather weeks' format, in the synthetic fixture and in the built tasks/wind/weather.jsonl, and the window
arithmetic that the build and the env share."""

import json
from datetime import datetime, timedelta

import pytest
import synthetic
from wind_fixtures import REAL_BUSTS, REAL_WEATHER, REAL_WEEKS, WEATHER, real, real_weeks

from port_week_core import wind

FILES = [pytest.param(WEATHER, id="fixture"), pytest.param(REAL_WEATHER, id="data", marks=real)]
WEEKS = [json.loads(line) for line in WEATHER.read_text().splitlines()]
ALL_WEEKS = ([pytest.param(("fixture", i), id=f"fixture-{i:02d}") for i in range(len(WEEKS))]
             + [pytest.param(("data", i), id=f"data-{i:02d}", marks=real) for i in range(len(REAL_WEEKS))])
HOURS = range(0, 264, 6)


@pytest.fixture
def week(request) -> dict:
    source, i = request.param
    return (WEEKS if source == "fixture" else real_weeks())[i]


def stamp(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def test_the_fixture_is_what_synthetic_builds():
    assert WEATHER.read_text() == synthetic.lines(synthetic.WEEKS)


def test_merge_joins_runs_of_hours_two_hours_or_less_apart():
    assert wind.merge([8, 3, 4, 7, 12, 4], 300) == [{"start": 3, "end": 9, "min_length": 300},
                                                   {"start": 12, "end": 13, "min_length": 300}]
    assert wind.merge([], 0) == []


def test_a_runs_windows_come_from_its_knots_exactly_in_thirds():
    calm = [12] * 25
    assert wind.run_windows(-12, calm) == []
    assert wind.run_windows(-12, [25] * 25) == []
    assert wind.run_windows(6, calm[:24] + [31]) == [{"start": 78, "end": 79, "min_length": 300},
                                                      {"start": 78, "end": 79, "min_length": 0}]
    peak = calm[:3] + [27] + calm[4:]
    assert wind.run_windows(0, peak) == [{"start": 9, "end": 10, "min_length": 300}]
    ramp = calm[:3] + [24, 27] + calm[5:]
    assert wind.run_windows(0, ramp) == [{"start": 11, "end": 13, "min_length": 300}]


@pytest.mark.parametrize("path", FILES)
def test_the_weeks_are_in_order_and_named_by_position(path):
    lines = path.read_text().splitlines()
    weeks = [json.loads(line) for line in lines]
    assert [w["id"] for w in weeks] == [f"e{i:02d}" for i in range(len(weeks))]
    assert [w["monday"] for w in weeks] == sorted(w["monday"] for w in weeks)
    for w in weeks:
        monday = stamp(w["monday"])
        assert monday.weekday() == 0 and monday.hour == 0
        assert w["iso_week"] == f"{monday.isocalendar()[0]}-W{monday.isocalendar()[1]:02d}"
        assert w["kind"] in ("storm", "bust")
        assert list(w) == ["id", "iso_week", "monday", "kind", "windows", "observed", "runs"]
    assert lines == [json.dumps(w, ensure_ascii=False, separators=(",", ":")) for w in weeks]


@real
def test_the_built_weeks_are_the_expected_storms_and_busts():
    assert [(w["iso_week"], w["kind"]) for w in real_weeks()] == [
        (w, "bust" if w in REAL_BUSTS else "storm") for w in REAL_WEEKS]


def by_length(windows: list[dict]) -> bool:
    return windows == sorted(windows, key=lambda w: (-w["min_length"], w["start"]))


@pytest.mark.parametrize("week", ALL_WEEKS, indirect=True)
def test_the_truth_is_the_observed_hours_merged(week):
    observed = week["observed"]
    assert by_length(observed) and all(0 <= o["start"] < o["end"] <= 264 for o in observed)
    for min_length in (300, 0):
        runs = [o for o in observed if o["min_length"] == min_length]
        assert all(a["end"] < b["start"] for a, b in zip(runs, runs[1:], strict=False))
    flagged = {m: [h for o in observed if o["min_length"] == m for h in range(o["start"], o["end"])] for m in (300, 0)}
    assert set(flagged[0]) <= set(flagged[300])
    assert [{k: w[k] for k in ("start", "end", "min_length")} for w in week["windows"]] == (
        wind.merge(flagged[300], 300) + wind.merge(flagged[0], 0))
    assert by_length(week["windows"])
    assert all(0 <= w["unvalidated"] <= 1 and round(w["unvalidated"], 3) == w["unvalidated"] for w in week["windows"])


@pytest.mark.parametrize("week", ALL_WEEKS, indirect=True)
def test_the_runs_are_those_in_force_at_some_watch_hour_each_with_its_windows(week):
    runs = week["runs"]
    assert [r["init"] for r in runs] == sorted({r["init"] for r in runs})
    monday = stamp(week["monday"])
    for r in runs:
        assert list(r) == ["run", "init", "issued", "kn", "windows"]
        assert r["init"] % 6 == 0 and r["issued"] == r["init"] + wind.PUBLISHED_HOURS
        assert stamp(r["run"]) == monday + timedelta(hours=r["init"])
        assert len(r["kn"]) == 25 and all(isinstance(x, int) and x >= 0 for x in r["kn"])
        assert r["windows"] == wind.run_windows(r["init"], r["kn"]) and by_length(r["windows"])
    in_force = [wind.delivered(week, h) for h in HOURS]
    assert {r["init"] for r in in_force} == {r["init"] for r in runs}
    for h, r in zip(HOURS, in_force, strict=True):
        assert r["issued"] <= h < r["init"] + wind.HORIZON_HOURS - 6
        assert not any(r["init"] < q["init"] and q["issued"] <= h for q in runs)


def test_a_missing_run_leaves_the_one_before_in_force():
    bust = WEEKS[1]
    assert 108 not in [r["init"] for r in bust["runs"]]
    assert [wind.delivered(bust, h)["init"] for h in (114, 120, 126)] == [102, 102, 114]


def test_observed_before_merges_only_the_hours_before():
    storm = WEEKS[0]
    assert storm["windows"][1:] == [{"start": 74, "end": 88, "min_length": 300, "unvalidated": 1.0},
                                    {"start": 80, "end": 84, "min_length": 0, "unvalidated": 1.0}]
    assert wind.observed_before(storm, 37) == []
    assert wind.observed_before(storm, 82) == [{"start": 37, "end": 41, "min_length": 300},
                                               {"start": 74, "end": 82, "min_length": 300},
                                               {"start": 80, "end": 82, "min_length": 0}]
    assert wind.observed_before(storm, 264) == [{k: w[k] for k in ("start", "end", "min_length")}
                                                for w in storm["windows"]]
