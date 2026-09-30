"""Advantage estimator tests — numpy only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TRAIN = Path(__file__).resolve().parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from alive import (
    ReplayQueue,
    advantages,
    choose_task,
    classify_group,
    drop_zero_advantage_group,
    group_rank_rewards,
    group_std,
    is_dead,
    should_skip,
)


def test_identical_rewards_are_dead():
    rewards = [0.0] * 8
    assert is_dead(rewards)
    assert classify_group(rewards, fingerprints=["a"] * 8) == "collapse"
    assert classify_group(rewards, fingerprints=list(range(8))) == "cliff"


def test_geoguesser_floor_is_a_cliff_not_a_collapse():
    # 3,324 km and 18,723 km both scored 0.0 under subtract-and-floor.
    rewards = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fingerprints = [3324, 18723, 4100, 9000, 12000, 5500, 8000, 16000]
    assert classify_group(rewards, fingerprints) == "cliff"
    assert should_skip("cliff")
    assert not should_skip("live")


def test_collapse_is_identical_trajectories():
    rewards = [0.64] * 8
    fingerprints = [("guess", 51.5, -0.1)] * 8
    assert classify_group(rewards, fingerprints) == "collapse"


def test_live_group_has_nonzero_std():
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert classify_group(rewards) == "live"
    assert group_std(rewards) > 0.2


def test_grpo_divides_by_std_the_way_run1_did():
    # Run 1's median group std was 0.016. `scale_rewards=group` divides by
    # that, so the same 0.016 gap that Dr.GRPO treats as 0.016 becomes ~1.
    tight = [0.492, 0.500, 0.508, 0.516]  # std ≈ 0.009
    grpo = advantages(tight, kind="grpo")
    dr = advantages(tight, kind="dr_grpo")
    ratio = float(np.max(np.abs(grpo)) / (np.max(np.abs(dr)) + 1e-12))
    assert ratio > 50
    rank = advantages(tight, kind="rank")
    assert float(np.max(np.abs(rank))) <= 0.5 + 1e-9
    assert rank.argmax() == grpo.argmax()


def test_rank_advantages_are_bounded_and_zero_sum():
    rewards = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.99]
    adv = advantages(rewards, kind="rank")
    assert float(np.max(np.abs(adv))) <= 0.5 + 1e-9
    assert abs(float(adv.sum())) < 1e-9
    # The numeric gaps are exponential; ranks should not care.
    assert adv[0] < adv[-1]
    assert adv.argmin() == 0
    assert adv.argmax() == 7


def test_rank_and_grpo_are_both_zero_on_a_tie():
    tied = [0.0] * 8
    assert np.allclose(advantages(tied, kind="rank"), 0.0)
    assert np.allclose(advantages(tied, kind="grpo"), 0.0)
    assert np.allclose(advantages(tied, kind="dr_grpo"), 0.0)
    assert np.allclose(advantages(tied, kind="loo"), 0.0)


def test_loo_is_zero_sum():
    rewards = [0.2, 0.4, 0.4, 0.9]
    adv = advantages(rewards, kind="loo")
    assert abs(float(adv.sum())) < 1e-9
    assert adv[-1] > adv[0]


def test_vanilla_grpo_keeps_tied_groups_in_the_batch():
    adv = advantages([0.0] * 8, kind="grpo")
    assert drop_zero_advantage_group(False, adv) is False
    assert drop_zero_advantage_group(True, adv) is True
    live = advantages([0.1, 0.2, 0.3, 0.4], kind="grpo")
    assert drop_zero_advantage_group(True, live) is False


def test_choose_task_replays_cliffs_instead_of_rewriting_a_consumed_row():
    rng = np.random.default_rng(0)
    q = ReplayQueue()
    q.observe(7, "cliff")
    draws = [choose_task(0, 10, q, rng, mix=1.0) for _ in range(400)]
    assert draws.count(7) > 2 * draws.count(0)
    uniform = [choose_task(3, 10, q, rng, mix=0.0) for _ in range(20)]
    assert uniform == [3] * 20


def test_group_rank_rewards_chunk_by_group_size():
    raws = [0.1, 0.9, 0.2, 0.8]
    ranked = group_rank_rewards(raws, group_size=2)
    assert len(ranked) == 4
    assert ranked[0] < ranked[1]
    assert ranked[2] < ranked[3]
    assert abs(ranked[0] + ranked[1]) < 1e-9


def test_replay_oversamples_cliff_tasks():
    rng = np.random.default_rng(0)
    q = ReplayQueue()
    q.observe(1, "cliff")
    q.observe(2, "live")
    q.observe(3, "collapse")
    draws = [q.sample([1, 2, 3], rng) for _ in range(3000)]
    counts = {i: draws.count(i) for i in (1, 2, 3)}
    assert counts[1] > 2 * counts[2]
    assert counts[1] > 2 * counts[3]
