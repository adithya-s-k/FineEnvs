"""The scripted customer desk by hand: it holds an outbound a few minutes for its connecting passengers in the window
ops commits next, it keeps ops from cancelling a flight whose passengers have no later seat, and without passengers
it does nothing."""

from dataclasses import replace

from day_helpers import day, flight, tail
from test_occ import crew, storm

from airops_core.programs import occ, sim
from airops_core.programs.pax import Group, Passengers


def case(task, units, pax) -> occ.Case:
    return occ.Case(task, tuple(units), lambda s, m: m, task.airports, (), sim.truth(task), pax)


def test_an_outbound_waits_a_few_minutes_for_its_connecting_passengers_and_that_lowers_the_cost():
    flights = [flight(0, "BUF", "JFK", 300, 60, lb=20), flight(1, "JFK", "PWM", 420, 60)]
    task = day(flights, [tail("N1", "BUF", legs=(0,)), tail("N2", "JFK", legs=(1,))])
    pax = Passengers((Group(0, (0,), 80), Group(1, (0, 1), 20), Group(2, (1,), 80)), fare=200.0)
    c = case(task, crew(1, (0,), "BUF", 240) + crew(2, (1,), "JFK", 360), pax)
    talk, customer = occ.run(c, "talk"), occ.run(c, "talk-customer")
    assert customer.holds == {1: 425} and [w["hold"] for w in customer.watches if w["hold"]] == [[[1, 425, 20]]]
    assert customer.day.run.flights[1].dep == 425 and talk.day.run.flights[1].dep == 420
    assert talk.day.run.pax.misconnected() == 20 and customer.day.run.pax.misconnected() == 0
    assert customer.day.total < talk.day.total and occ.roles(customer, occ.MODES["talk-customer"], task) == []


def test_ops_leaves_open_a_flight_whose_passengers_have_no_later_seat():
    task = day([flight(0, "BUF", "JFK", 300, 60, lb=200, deadline=400)], [tail("N1", "BUF", legs=(0,))])
    c = case(task, crew(1, (0,), "BUF", 240), Passengers((Group(0, (0,), 100),), fare=200.0))
    talk, customer = occ.run(c, "talk"), occ.run(c, "talk-customer")
    assert talk.causes == {0: occ.RULE} and 0 not in customer.plan and customer.protected == {0}
    assert customer.watches[3]["protect"] == [0] and customer.watches[3]["t"] == 180


def test_without_passengers_the_customer_desk_does_nothing():
    c = storm()
    talk, customer = occ.run(c, "talk"), occ.run(c, "talk-customer")
    assert occ.same(talk.day, customer.day) and talk.plan == customer.plan and talk.replies == customer.replies
    assert replace(c, pax=None) == c and not customer.holds and not customer.protected
