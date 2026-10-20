import random
from dataclasses import replace

import pytest
from day_helpers import GDP, crewed, day, flight, network_day, random_day, shortened, stop, tail
from program_helpers import adv

from airops_core.programs import mip, references, sim
from airops_core.programs.cost import cancel_price
from airops_core.programs.model import Move, Outcome, Task


@pytest.fixture(scope="module", params=["task", "crewed"])
def fixture_day(request) -> Task:
    """The fixture day, and the same day with the crew model."""
    return shortened(request.getfixturevalue(request.param))


def stopped_then_released():
    """A ground stop issued to 10:00 and cancelled at 03:20; the desk sees the cancellation only from 03:20 on."""
    flights = [flight(0, "BUF", "JFK", 100, 60, deadline=500), flight(1, "JFK", "MCO", 500, 150)]
    return day(flights, [tail("N1", "BUF", end="JFK"), tail("N2", "JFK", end="MCO")], stops=(stop(0, 200),),
               advisories=(adv("s1", "gs", -30, 0, 600), adv("c1", "gs_cnx", 200, 200, None)))


def delay_revealed_late():
    """A flight whose 5-h delay, known two hours before its departure, puts it past its deadline."""
    return day([flight(0, "DCA", "JFK", 400, 60, lb=300, deadline=640)], [tail("N1", "DCA")])


def committed_then_stopped():
    """A flight the desk commits at 01:00 that a ground stop signed at 01:30 then holds past its deadline."""
    flights = [flight(0, "DCA", "JFK", 100, 60, deadline=220), flight(1, "JFK", "BUF", 400, 60)]
    return day(flights, [tail("N1", "DCA"), tail("N2", "JFK", end="BUF")], stops=(stop(90, 300),),
               advisories=(adv("s1", "gs", 90, 90, 300),))


def released_early():
    """A ground stop issued to 05:00 and cancelled at 02:30, after which the flight it held may leave."""
    flights = [flight(0, "BUF", "JFK", 100, 60, deadline=900), flight(1, "JFK", "MCO", 500, 150)]
    return day(flights, [tail("N1", "BUF", end="JFK"), tail("N2", "JFK", end="MCO")], stops=(stop(0, 150),),
               advisories=(adv("s1", "gs", -30, 0, 300), adv("c1", "gs_cnx", 150, 150, None)))


def gdp_announced_long():
    """A GDP signed at 00:00 for 02:30-06:40 whose forecast gives JetBlue no slot, and a flight into it that, held
    past the period, misses its deadline; on the truth it gets a slot."""
    return day([flight(0, "BUF", "JFK", 100, 60, deadline=300)], [tail("N1", "BUF", end="JFK")], segments=(GDP,))


def test_watches_run_hourly_until_every_delay_is_known():
    task = stopped_then_released()
    assert references.watch_times(task) == [0, 60, 120, 180, 240, 300, 360, 420]
    assert references.watch_times(delay_revealed_late()) == [0, 60, 120, 180, 240, 300]


def test_a_confirmed_cancellation_is_final_even_when_the_day_turns_out_better():
    task = stopped_then_released()
    followed = references.follow(task)
    assert followed.plan[0] == {"cancel": True, "decided": 0} and followed.run.flights[0].status == "cancelled"
    assert [w["cancelled"] for w in followed.watches] == [[0]] + [[]] * 7
    assert references.rule(task).plan[0] == {"cancel": True, "decided": 0}
    assert references.naive(task).flights[0].dep == 200
    sol, run = references.anchor(task)
    assert run.flights[0].dep == 200 and sol.objective < followed.cost


def test_at_a_watch_the_day_so_far_is_history_and_every_other_flight_waits_for_the_watch():
    task = stopped_then_released()
    plan = {0: {"not_before": 150}, 1: {"not_before": 520}}
    assert references.so_far(task, plan, 300) == {0: {"left": 200}, 1: {"not_before": 520}}
    assert references.so_far(task, plan, 180) == {0: {"not_before": 180}, 1: {"not_before": 520}}
    cancelled = {0: {"cancel": True, "decided": 0}}
    assert references.so_far(task, cancelled, 300) == {0: {"cancel": True, "decided": 0}, 1: {"not_before": 300}}


def test_neither_the_rule_nor_the_follower_cancels_a_flight_once_it_is_committed():
    task = committed_then_stopped()
    rule, followed = references.rule(task), references.follow(task)
    assert [w["cancelled"] for w in rule.watches] == [[]] * len(rule.watches)
    assert not followed.plan.get(0, {}).get("cancel") and followed.run.flights[0].status == "auto_cancelled"
    assert rule.run.flights[0].decided == followed.run.flights[0].decided == 220


def test_the_follower_holds_only_what_it_decides_to_and_a_flight_it_did_not_hold_leaves_when_the_stop_ends():
    task = released_early()
    followed = references.follow(task)
    assert 0 not in followed.plan and followed.run.flights[0].dep == 150
    assert [w["held"] for w in followed.watches] == [0] * len(followed.watches)
    world = sim.known(task, 0)
    assert references.deliberate(task, {}, {0: 300, 1: 500}, world, 0) == {}
    assert references.deliberate(task, {}, {0: 330, 1: 500}, world, 0) == {0: 330}
    assert references.deliberate(task, {0: {"cancel": True, "decided": 0}}, {1: 520}, world, 0) == {1: 520}


def test_a_base_rate_follower_expects_each_gdp_to_end_early_but_never_before_the_watch():
    task = gdp_announced_long()
    issued = sim.known(task, 0).segments["JFK"]
    assert [s.until for s in issued] == [401]
    assert [s.until for s in sim.known(task, 0, early=250).segments["JFK"]] == [151]
    assert [s.until for s in sim.known(task, 200, early=250).segments["JFK"]] == [200]
    assert sim.known(task, 0, early=250).segments["JFK"][0].slots == issued[0].slots
    assert references.follow(task).plan == {0: {"cancel": True, "decided": 0}}
    followed = references.follow(task, early=250)
    assert followed.plan == {0: {"not_before": 151}} and followed.run.flights[0].status == "flown"


def test_each_cancellation_is_priced_at_the_watch_that_confirms_it():
    task = delay_revealed_late()
    followed, rule = references.follow(task), references.rule(task)
    assert followed.plan[0] == rule.plan[0] == {"cancel": True, "decided": 300}
    assert followed.run.cost.notice == rule.run.cost.notice == 100 * (300 - 280)
    assert references.naive(task).cost.notice == 100 * (640 - 400 + 120)
    sol, run = references.anchor(task)
    assert sol.objective == run.cost.total == 52_300 - 61 * 60 and run.cost.notice == 0


def crew_timing_out():
    """A 5-h delay known two hours before the 06:40 departure puts a flight past its crew's deadline at 07:30, their
    rest ending past its own."""
    return day([crewed(flight(0, "DCA", "JFK", 400, 60, lb=300, deadline=900), 450)], [tail("N1", "DCA")])


def stopped_past_the_crew():
    """A ground stop signed at 01:00 holds a 01:40 departure past its crew's 01:50 deadline, their rest ending past its
    own: from that watch its auto-cancel is within the freeze. A later flight keeps the watches going."""
    flights = [crewed(flight(0, "DCA", "JFK", 100, 60, deadline=600), 110), flight(1, "JFK", "BOS", 400, 60)]
    return day(flights, [tail("N1", "DCA"), tail("N2", "JFK")], stops=(stop(60, 300),),
               advisories=(adv("s1", "gs", 60, 60, 300),))


def test_a_crew_deadline_is_known_from_the_start_and_an_auto_cancel_it_forces_is_priced_there():
    task = crew_timing_out()
    followed, rule = references.follow(task), references.rule(task)
    assert followed.plan[0] == rule.plan[0] == {"cancel": True, "decided": 300}
    assert followed.run.cost.notice == rule.run.cost.notice == 100 * (300 - 280)
    assert references.naive(task).flights[0].decided == 450 and references.naive(task).cost.notice == 100 * 170
    assert references.follow(replace(task, flights=(replace(task.flights[0], crew=None),))).run.flights[0].dep == 700
    task = stopped_past_the_crew()
    followed, rule = references.follow(task), references.rule(task)
    assert 0 not in followed.plan and 0 not in rule.plan and [w["frozen"] for w in followed.watches[:2]] == [0, 1]
    assert followed.run.flights[0].decided == rule.run.flights[0].decided == 110
    assert followed.run.cost.notice == rule.run.cost.notice == 100 * (110 - 100 + 120)


def test_the_recorded_day_prices_jetblues_outcome_like_for_like():
    flown = flight(0, "BUF", "JFK", 100, 60, actual=Outcome(False, False, "", 100, 190, "JFK", "N1"))
    cancelled = flight(1, "JFK", "DCA", 300, 60, pax=50, actual=Outcome(True, False, "B", None, None, None, "N1"))
    diverted = flight(2, "JFK", "MCO", 180, 150, pax=10, actual=Outcome(False, True, "", 200, None, "PWM", "N2"))
    tails = [tail("N1", "BUF", end="JFK"), replace(tail("N2", "JFK", end="MCO"), actual_end="PWM")]
    got, ferries = references.recorded(day([flown, cancelled, diverted], tails))
    assert (got.delay, got.cancel, got.notice, ferries) == (100 * 30, 523 * 60, 10 * (200 - 60), 1)
    assert (got.gate, got.overrun, got.saved) == (34 * (200 - 180), 95 * (190 - 100 - 60), 61 * 60)
    later, _ = references.recorded(day([flown, cancelled, diverted], tails, start=250))
    assert later.notice == 50 * (250 - 180) + 10 * (200 - 60)


def test_the_airline_replay_holds_jetblues_departures_and_claims_its_arrival_order(fixture_day):
    task = fixture_day
    plan = references.airline_plan(task)
    for f in task.flights:
        if f.actual.dep is None:
            assert plan[f.id] == {"cancel": True, "decided": max(f.sched_dep - 120, task.start)}
        else:
            assert plan[f.id] == {"not_before": f.actual.dep, "claim": f.actual.dep + f.sched_block}
    replay = references.airline(task)
    assert replay.flew == sum(f.actual.dep is not None and f.actual.dep <= f.deadline for f in task.flights)
    assert 0 <= replay.auto <= replay.flew


def test_the_airline_replay_counts_no_leg_whose_aircraft_came_on_a_leg_flown_past_its_deadline():
    out = flight(0, "JFK", "MCO", 1300, 150, actual=Outcome(False, False, "", 1650, 1800, "MCO", "N1"))
    back = flight(1, "MCO", "JFK", 1500, 150, actual=Outcome(False, False, "", 1900, 2050, "JFK", "N1"))
    other = flight(2, "BUF", "JFK", 1000, 60, actual=Outcome(False, False, "", 1010, 1070, "JFK", "N2"))
    task = sim.with_cutoff(day([out, back, other], [tail("N1", "JFK"), tail("N2", "BUF", end="JFK")]), 1620)
    replay = references.airline(task)
    assert [replay.run.flights[i].status for i in range(3)] == ["auto_cancelled", "auto_cancelled", "flown"]
    assert references.counted(task) == [2] and (replay.flew, replay.auto) == (1, 0)


def test_no_reference_costs_less_than_the_anchor(fixture_day):
    none_below_the_anchor(fixture_day)


@pytest.mark.parametrize("seed", range(4))
def test_no_reference_costs_less_than_the_anchor_on_days_with_moves(seed):
    none_below_the_anchor(random_day(random.Random(seed)))


@pytest.mark.parametrize("seed", range(6))
def test_no_reference_costs_less_than_the_anchor_on_whole_network_days(seed):
    none_below_the_anchor(network_day(random.Random(seed)))


def stopped_bound_leg():
    """A ground stop signed at 01:00 holds DCA-JFK past its 01:50 deadline, so from that watch its auto-cancel is
    pinned; the Move after it would have taken its aircraft to MCO for MCO-JFK. A later flight keeps the watches
    going."""
    flights = [flight(0, "DCA", "JFK", 100, 60, deadline=110), flight(1, "MCO", "JFK", 250, 150, deadline=600),
               flight(2, "JFK", "BOS", 400, 60)]
    move = Move(0, "JFK", "MCO", 160, 220)
    return day(flights, [tail("N1", "DCA", end="JFK", repositions=(move,), legs=(0, 1)), tail("N2", "JFK", end="BOS")],
               stops=(stop(60, 300),), advisories=(adv("s1", "gs", 60, 60, 300),))


def lapse_over_cancel():
    """JFK-PWM can't leave by its deadline; cancelling it would send the aircraft that lands from BUF through its Move
    to MCO, so the desk lets it auto-cancel and the aircraft flies JFK-DCA's 200 passengers."""
    flights = [flight(0, "BUF", "JFK", 20, 60), flight(1, "JFK", "PWM", 100, 60, pax=10, lb=400, deadline=300),
               flight(2, "JFK", "DCA", 200, 60, pax=200), flight(3, "MCO", "JFK", 400, 150, pax=10)]
    move = Move(1, "PWM", "MCO", 160, 370)
    return day(flights, [tail("N1", "BUF", end="DCA", repositions=(move,), legs=(0, 1, 3))])


def test_every_follower_watch_solve_replays_with_its_pins_and_a_pinned_auto_cancel_moves_no_aircraft(monkeypatch):
    task, solves = stopped_bound_leg(), []
    solve = mip.solve

    def kept(task, world, price, fixed, earliest, **kw):
        sol = solve(task, world, price, fixed, earliest, **kw)
        solves.append((world, price, fixed, kw["lapsed"], sol))
        return sol

    monkeypatch.setattr(mip, "solve", kept)
    followed = references.follow(task)
    assert any(0 in lapsed for *_, lapsed, _ in solves)
    for world, price, fixed, lapsed, sol in solves:
        plan = {}
        for f in task.flights:
            d = sol.deps[f.id]
            if f.id in fixed and d is not None:
                plan[f.id] = {"left": d}
            elif f.id in sol.lapsed:
                plan[f.id] = {"not_before": f.deadline + 1}
            elif d is None:
                decided = next(t for t in range(task.start, f.deadline + 1) if cancel_price(f, t) == price[f.id])
                plan[f.id] = {"cancel": True, "decided": decided}
            else:
                plan[f.id] = {"not_before": d}
        run = sim.execute(task, plan, world)
        assert run.cost.total == sol.objective
        assert all((r.dep if r.status == "flown" else None) == sol.deps[i] for i, r in run.flights.items())
        assert {i for i in fixed if fixed[i] is None} & set(sol.lapsed) == set(lapsed) & set(task.moved)
    assert followed.run.flights[0].status == "auto_cancelled" and followed.run.flights[1].status != "flown"


def test_a_flight_the_follower_lets_auto_cancel_is_held_past_its_deadline_not_cancelled():
    task = lapse_over_cancel()
    followed = references.follow(task)
    assert followed.plan[1] == {"not_before": task.flight(1).deadline + 1}
    assert [r.status for r in followed.run.flights.values()] == ["flown", "auto_cancelled", "flown", "cancelled"]
    assert followed.run.flights[2].tail == "N1"


def none_below_the_anchor(task: Task) -> None:
    sol, run = references.anchor(task)
    assert sol.proven and run.cost.total == sol.objective
    played = {"naive": references.naive(task), "airline": references.airline(task).run,
              "rule": references.rule(task).run, "follower": references.follow(task).run,
              "blind": references.follow(task, programs=False).run}
    assert references.below_anchor({k: r.cost.total for k, r in played.items()}, sol.objective) == {}
    assert all(references.seats_ok(task, r) for r in [run, *played.values()])
    flat = references.flat_anchor(task)
    assert flat.objective <= played["naive"].cost.flat and flat.objective <= sol.objective


def test_the_calm_day_flies_a_feasible_schedule_without_cancellations_or_ferries():
    flights = [flight(0, "JFK", "BUF", 100, 60, lb=90), flight(1, "BUF", "JFK", 200, 60, lb=50)]
    calm = references.calm(day(flights, [tail("N1", "JFK")]))
    assert (calm.cancelled, calm.ferries) == (0, 0) and calm.run.cost.total == 0
    assert calm.solution.objective == 0


def test_past_midnight_counts_hub_movements_after_2400(fixture_day):
    task = fixture_day
    run = sim.execute(task, {})
    late = sum((r.dep or 0) >= 1440 and task.flight(i).orig in ("JFK", "BOS") for i, r in run.flights.items())
    late += sum((r.arr or 0) >= 1440 and task.flight(i).dest in ("JFK", "BOS") for i, r in run.flights.items())
    assert references.past_midnight(task, run) == late
