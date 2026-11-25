import math

import pytest

from airops_core.core.cost import K_CANCEL, breakdown, ferry_cost, miles
from airops_core.core.reward import reward


def test_the_sourced_constants():
    assert K_CANCEL == 523 == round(456.9 + 0.3 * 235 / 1.07)
    assert round(98.41 / 1.07) == 92


def test_ferries_cost_92_per_block_minute_over_30_plus_miles_over_8(make_task):
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 700}], [{"tail": "N1", "pos": "JFK"}])
    m = miles(task.airports["JFK"], task.airports["BOS"])
    assert 180 < m < 190
    assert ferry_cost(task, "JFK", "BOS") == round(92 * (30 + m / 8)) == ferry_cost(task, "BOS", "JFK")
    assert ferry_cost(task, "JFK", "JFK") == 0
    a, b, c = (ferry_cost(task, *p) for p in (("BOS", "MCO"), ("BOS", "DCA"), ("DCA", "MCO")))
    assert a <= b + c


def test_the_breakdown_charges_delay_cancellations_and_ferries(make_task):
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 700, "pax": 100},
                      {"orig": "BOS", "dest": "JFK", "dep": 800, "pax": 80}], [{"tail": "N1", "pos": "JFK"}])
    cost = breakdown(task, {0: 760 + 25, 1: None}, [("BOS", "JFK")])
    assert (cost.pax_delay, cost.cancel, cost.ferry) == (2500, 523 * 80, ferry_cost(task, "BOS", "JFK"))
    assert cost.total == 2500 + 523 * 80 + ferry_cost(task, "BOS", "JFK")
    assert cost.to_dict()["total"] == cost.total
    assert breakdown(task, {0: 700, 1: 800}, []).total == 0


def test_reward_is_one_at_the_anchor_and_falls_to_a_floor_of_point_two():
    anchor, floor = 1_000_000, 800_000
    assert reward(anchor, anchor, floor) == reward(anchor - 1, anchor, floor) == 1.0
    span = anchor - floor + 0.02 * anchor
    assert reward(anchor + 0.5 * span, anchor, floor) == pytest.approx(0.2 + 0.8 * math.exp(-1))
    costs = [anchor + k * 50_000 for k in range(40)]
    rewards = [reward(c, anchor, floor) for c in costs]
    assert all(a > b for a, b in zip(rewards, rewards[1:], strict=False))
    assert all(0.2 < r <= 1.0 for r in rewards)
    assert reward(10**12, anchor, floor) == pytest.approx(0.2)


def test_reward_survives_an_anchor_at_the_floor():
    assert reward(0, 0, 0) == 1.0
    assert 0.2 <= reward(100, 0, 0) < 1.0
