"""GeoGuesser Monte Carlo: never-submit vs floor, same-task resample."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TRAIN = Path(__file__).resolve().parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from analyse_geoguesser import (
    P_NO_SUBMIT,
    game_subtract,
    mixture_multiply,
    rewards_from,
    sample_task_distances,
)


def test_never_submit_is_not_the_subtract_floor():
    rng = np.random.default_rng(0)
    distances, _ = sample_task_distances(rng, 8_000, 8)
    never = float(np.mean(np.isnan(distances)))
    sub = float(np.mean(rewards_from(distances, game_subtract).ravel() == 0.0))
    mix = float(np.mean(rewards_from(distances, mixture_multiply).ravel() == 0.0))
    assert abs(never - P_NO_SUBMIT) < 0.02
    assert abs(mix - P_NO_SUBMIT) < 0.02
    # Far submitted guesses also floor under subtract, so zeros exceed never-submit.
    assert sub > never + 0.04
    assert abs(sub - 77 / 200) < 0.05


def test_resample_keeps_the_task_median():
    rng = np.random.default_rng(1)
    first, median = sample_task_distances(rng, 64, 8)
    second, again = sample_task_distances(rng, 64, 8, task_median=median)
    assert np.allclose(median, again)
    # New jitter / submit draws, not a copy of the first group.
    assert not np.array_equal(np.isnan(first), np.isnan(second)) or not np.allclose(
        np.nan_to_num(first, nan=-1.0), np.nan_to_num(second, nan=-1.0)
    )
