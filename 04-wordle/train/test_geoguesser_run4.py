"""GeoGuesser group classifier, using the numbers in LEARNINGS.md."""

from __future__ import annotations

import sys
from pathlib import Path

_TRAIN = Path(__file__).resolve().parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from geoguesser_run4 import inspect_group, run4_env, should_stop


def test_subtract_floor_group_is_a_cliff():
    # scoring.py: 3,324 km and 18,723 km both scored 0.0.
    rewards = [0.0] * 8
    distances = [3324, 18723, 4100, 9000, 12000, 5500, 8000, 16000]
    report = inspect_group(rewards, fingerprints=distances, turns=[7] * 8)
    assert report.kind == "cliff"
    assert report.skip
    assert "densify" in report.recommendation


def test_run1_converged_policy_is_collapse():
    # LEARNINGS.md: 1.1 turns, 66 tokens, 0.5% non-submission.
    rewards = [0.64] * 8
    fingerprints = [("guess", 51.5074, -0.1278)] * 8
    report = inspect_group(rewards, fingerprints=fingerprints, turns=[1] * 8)
    assert report.kind == "collapse"
    assert "one-glance" in report.recommendation


def test_live_group_trains():
    rewards = [0.1, 0.2, 0.4, 0.5, 0.55, 0.6, 0.7, 0.9]
    fingerprints = list(range(8))
    report = inspect_group(rewards, fingerprints=fingerprints, turns=[4, 5, 6, 3, 7, 2, 8, 5])
    assert report.kind == "live"
    assert not report.skip
    assert report.recommendation == "train"


def test_collapse_with_many_turns_is_still_not_train():
    rewards = [0.64] * 8
    fingerprints = [("look", "guess")] * 8
    report = inspect_group(rewards, fingerprints=fingerprints, turns=[6] * 8)
    assert report.kind == "collapse"
    assert report.skip
    assert report.recommendation != "train"
    assert "resampling will not help" in report.recommendation


def test_run4_config_does_not_silently_change_run1():
    env = run4_env()
    assert env["SCALE_REWARDS"] == "none"
    assert env["MAX_STEPS"] == "250"
    assert env["COST_SCALE"] == "0.2"
    assert env["ACCUM"] == "2"
    assert env["ALIVE_SKIP_CLIFF"] == "1"
    assert env["ALIVE_STOP_COLLAPSE"] == "0.8"
    assert should_stop(0.9, threshold=float(env["ALIVE_STOP_COLLAPSE"]))
    assert not should_stop(0.1, threshold=float(env["ALIVE_STOP_COLLAPSE"]))
