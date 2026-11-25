from dataclasses import replace

import pytest
from day_helpers import day, flight, tail

from airops_core.core.cost import miles
from airops_core.programs import cost, sim
from airops_core.programs.cost import Rates


def test_the_faa_rates_reach_whole_passenger_minutes_per_aircraft_minute_in_2025_dollars():
    assert round(cost.CPI_2025 / cost.CPI_YE_JUNE_2023, 4) == 1.0743
    assert round(5650 / 60 * cost.CPI_2025 / cost.CPI_YE_JUNE_2023 / 1.07, 2) == 94.54
    assert cost.RATES["A320"] == cost.RATES["A321"] == cost.RATES["A321P"] == Rates(34, 95, 61)
    assert cost.RATES["A220"] == cost.RATES["E190"] == Rates(32, 88, 56)


@pytest.mark.parametrize(("family", "gate", "block", "saved"), [("A321", 34, 95, 61), ("E190", 32, 88, 56)])
def test_gate_delay_block_overrun_and_a_cancellations_saving_are_priced_per_family(family, gate, block, saved):
    f = replace(flight(0, "JFK", "BUF", 600, 75, pax=100, sched_block=60), family=family)
    assert cost.flown_price(f, 630, 705) == 100 * (705 - 660) + gate * 30 + block * 15
    assert cost.flown_price(f, 590, 640) == cost.flown_price(f, 600, 660) == 0
    assert cost.flat_price(f) == 100 * 523 - saved * 60
    assert cost.cancel_price(f, 480) == cost.cancel_price(f, 300) == cost.flat_price(f)
    assert cost.cancel_price(f, 600) == 100 * (523 + 120) - saved * 60
    assert cost.cancel_price(f, 600 + 900) == 100 * 1_543 - saved * 60


def test_a_cancellation_costs_523_a_passenger_plus_a_minute_for_each_minute_past_the_notice_line():
    f = flight(0, "JFK", "BUF", 600, 60, pax=100)
    assert cost.cancel_price(f, 480) == cost.cancel_price(f, 300) == 52_300 - 61 * 60
    assert cost.cancel_price(f, 600) - cost.cancel_price(f, 480) == 100 * 120


def test_the_day_adds_delay_cancellations_their_notice_ferries_and_its_aircrafts_minutes():
    a, b, c = flight(0, "JFK", "BUF", 600, 70, pax=100, sched_block=60), flight(1, "BUF", "JFK", 700, 60, pax=50), \
        flight(2, "JFK", "MCO", 800, 150, pax=10)
    task = day([a, b, c], [tail("N1", "JFK")])
    got = cost.priced(task, {0: 630, 1: None, 2: 830}, {0: 700, 1: None, 2: None}, {1: 700, 2: 830},
                      [("A320", "BUF", "JFK"), ("E190", "JFK", "BUF")])
    assert (got.delay, got.cancel, got.notice) == (100 * 40, 523 * 60, 50 * 120 + 10 * 150)
    assert (got.gate, got.overrun, got.saved) == (34 * 30 + 34 * 30, 95 * 10, 61 * 60)
    trip = 30 + miles(task.airports["BUF"], task.airports["JFK"]) / 8
    assert got.ferry == round(95 * trip) + round(88 * trip)
    assert got.total == got.flat + got.notice
    assert got.flat == got.delay + got.cancel + got.ferry + got.gate + got.overrun - got.saved
    assert cost.ferry_cost(task, "E190", "JFK", "JFK") == 0


def test_an_auto_cancel_is_decided_at_its_deadline():
    f = flight(0, "BUF", "JFK", 600, 60, pax=100, deadline=900)
    task = day([f], [tail("N1", "JFK", end="JFK")])
    run = sim.execute(task, {})
    assert run.flights[0].status == "auto_cancelled" and run.flights[0].decided == 900
    assert run.cost.notice == 100 * (900 - 600 + 120) and run.cost.flat == 52_300 - 61 * 60 + run.cost.ferry
