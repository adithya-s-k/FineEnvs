from dataclasses import replace

from day_helpers import (
    GDP,
    broken_day,
    busy_day,
    crewed,
    day,
    flight,
    moved_day,
    quick_day,
    stop,
    tail,
    three_into_a_program,
    turned_through_day,
)

from airops_core.programs import sim


def slots(run):
    return {i: (r.dep, r.slot) for i, r in run.flights.items()}


def test_controlled_flights_leave_at_the_edct_of_the_earliest_free_slot_and_a_cancellation_moves_the_queue_up():
    task = three_into_a_program()
    assert slots(sim.execute(task, {})) == {0: (120, 180), 1: (180, 240), 2: (240, 300)}
    run = sim.execute(task, {0: {"cancel": True, "decided": 0}})
    assert slots(run) == {0: (None, None), 1: (120, 180), 2: (180, 240)}
    assert run.flights[1].arr == 180 and run.flights[1].program == "g"


def test_claims_bind_first_and_a_claim_the_flight_cannot_make_is_lost_and_its_slot_wasted():
    task = three_into_a_program()
    assert slots(sim.execute(task, {2: {"claim": 180}})) == {0: (180, 240), 1: (240, 300), 2: (120, 180)}
    late = three_into_a_program(lb=100)
    run = sim.execute(late, {2: {"claim": 180}})
    assert slots(run) == {0: (180, 240), 1: (240, 300), 2: (341, None)}
    assert run.wasted == (("g", 180),) and run.flights[2].arr == 401


def test_a_flight_that_has_left_leaves_then_on_its_slot_if_it_is_one_and_on_nothing_if_not():
    task = three_into_a_program()
    run = sim.execute(task, {2: {"left": 110}, 0: {"left": 180}})
    assert slots(run) == {0: (180, 240), 1: (120, 180), 2: (110, None)}
    held = day([flight(0, "BUF", "JFK", 120, 60)], [tail("N1", "BUF")], stops=(stop(100, 200),))
    assert sim.execute(held, {0: {"left": 150}}).flights[0].dep == 150


def test_a_ground_stop_holds_departures_to_its_hub_only():
    flights = [flight(0, "BUF", "JFK", 120, 60), flight(1, "DCA", "BOS", 120, 60), flight(2, "MCO", "JFK", 90, 150)]
    task = day(flights, [tail("N1", "BUF"), tail("N2", "DCA"), tail("N3", "MCO")], stops=(stop(100, 200),))
    run = sim.execute(task, {})
    assert [run.flights[i].dep for i in range(3)] == [200, 120, 90]


def test_the_nominal_cap_holds_arrivals_beyond_the_hours_schedule():
    flights = [flight(0, "BUF", "JFK", 100, 60, lb=30), flight(1, "DCA", "JFK", 130, 60),
               flight(2, "MCO", "JFK", 140, 60)]
    task = day(flights, [tail("N1", "BUF"), tail("N2", "DCA"), tail("N3", "MCO")], aar=1)
    assert [sim.execute(task, {}).flights[i].arr for i in range(3)] == [190, 190, 240]


def test_a_hub_departure_auto_cancels_at_the_cutoff_and_counts_as_decided_there():
    flights = [flight(0, "BOS", "JFK", 1300, 70, lb=300), flight(1, "JFK", "MCO", 1500, 150, pax=10),
               flight(2, "MCO", "BUF", 1400, 150)]
    task = sim.with_cutoff(day(flights, [tail("N1", "BOS", end="MCO"), tail("N2", "MCO", end="BUF")]), 1620)
    assert [f.deadline for f in task.flights] == [1620, 1620, 2300]
    run = sim.execute(task, {})
    assert run.flights[0].dep == 1600 and run.flights[1].status == "auto_cancelled"
    assert run.flights[1].decided == 1620 and run.cost.notice == 10 * (1620 - 1500 + 120)
    later = sim.execute(sim.with_cutoff(task, 1740), {})
    assert later.flights[1].dep == 1700 and later.cost.notice == 0


def test_every_confirmed_cancellation_stands_whatever_the_day_does():
    task = three_into_a_program()
    run = sim.execute(task, {1: {"cancel": True, "decided": 130, "not_before": 100, "claim": 180}})
    assert run.flights[1].status == "cancelled" and run.flights[1].decided == 130
    assert run.cost.notice == 100 * (130 - 100 + 120)


def test_a_move_takes_the_aircraft_that_flew_its_leg_once_turned_and_leaves_the_others_where_they_are():
    run = sim.execute(moved_day(), {})
    assert [(r.tail, r.dep) for r in run.flights.values()] == [("N1", 20), ("N1", 120), ("N1", 330), ("N3", 150)]
    assert run.moves == {1: 300} and run.ends["N2"] == "BUF" and not run.ferries
    late = sim.execute(moved_day(lb=200), {})
    assert (late.flights[1].dep, late.moves, late.flights[2].dep) == (320, {1: 410}, 410)
    assert late.flights[2].tail == late.flights[1].tail and late.ends["N2"] == "BUF"


def test_a_moves_leg_cancelled_leaves_its_aircraft_to_reposition_from_its_origin_and_takes_no_other():
    task = moved_day()
    run = sim.execute(task, {1: {"cancel": True}})
    assert [(r.status, r.tail, r.dep) for r in run.flights.values()] == [
        ("flown", "N1", 20), ("cancelled", None, None), ("flown", "N1", 330), ("flown", "N3", 150)]
    assert run.moves == {1: 300} and run.ends["N2"] == "BUF"
    run = sim.execute(task, {0: {"cancel": True}, 1: {"cancel": True}})
    assert (run.flights[2].tail, run.flights[3].tail, run.moves, run.ends["N2"]) == ("N1", "N3", {1: 300}, "BUF")


def test_cancelled_legs_into_a_break_join_its_repositioning():
    task = broken_day()
    run = sim.execute(task, {})
    assert [r.dep for r in run.flights.values()] == [20, 200, 400, 640] and run.moves == {0: 170, 1: 370}
    run = sim.execute(task, {0: {"cancel": True}})
    assert (run.flights[1].tail, run.flights[1].dep, run.moves) == ("N1", 200, {0: 170, 1: 370})
    run = sim.execute(task, {1: {"cancel": True}})
    assert [(r.status, r.tail, r.dep) for r in run.flights.values()] == [
        ("flown", "N1", 20), ("cancelled", None, None), ("flown", "N1", 400), ("flown", "N1", 640)]
    assert run.moves == {0: 170, 1: 370}
    run = sim.execute(task, {3: {"cancel": True}})
    assert run.ends == {"N1": "MCO"}


def test_a_leg_cancelled_after_a_break_leaves_its_aircraft_where_the_move_put_it():
    task = broken_day()
    run = sim.execute(task, {2: {"cancel": True}})
    assert [r.status for r in run.flights.values()] == ["flown", "flown", "cancelled", "auto_cancelled"]
    assert run.ends == {"N1": "DCA"} and run.moves == {0: 170, 1: 370}
    run = sim.execute(task, {1: {"cancel": True}, 2: {"cancel": True}})
    assert (run.flights[3].status, run.ends, run.moves) == ("auto_cancelled", {"N1": "DCA"}, {0: 170, 1: 370})
    run = sim.execute(task, {0: {"cancel": True}, 1: {"cancel": True}})
    assert (run.flights[2].dep, run.flights[3].dep, run.ends) == (400, 640, {"N1": "JFK"})


def test_a_bound_leg_that_auto_cancels_moves_no_aircraft():
    run = sim.execute(moved_day(lb=300), {})
    assert [r.status for r in run.flights.values()] == ["flown", "auto_cancelled", "auto_cancelled", "flown"]
    assert run.moves == {1: None} and sorted(run.ends.values()) == ["BUF", "DCA", "JFK"]


def test_a_busy_block_keeps_the_aircraft_that_flew_its_leg_and_a_cancelled_leg_has_none():
    task = busy_day()
    run = sim.execute(task, {})
    assert [(r.tail, r.dep) for r in run.flights.values()] == [("N2", 160), ("N1", 170), ("N2", 430)]
    assert not run.moves
    run = sim.execute(task, {0: {"cancel": True}})
    assert [(r.status, r.tail, r.dep) for r in run.flights.values()] == [
        ("cancelled", None, None), ("flown", "N1", 170), ("auto_cancelled", None, None)]


def test_the_day_as_known_hides_delays_and_true_blocks_until_look():
    flights = [flight(0, "BUF", "JFK", 100, 80, lb=20, sched_block=60), flight(1, "DCA", "JFK", 400, 90, lb=40,
                                                                              sched_block=60)]
    task = day(flights, [tail("N1", "BUF"), tail("N2", "DCA")], segments=(GDP,))
    world = sim.known(task, 0)
    assert (world.lb, world.block) == ({0: 20, 1: 0}, {0: 80, 1: 60})
    assert sim.known(task, 0, programs=False).segments == {"JFK": (), "BOS": ()}
    assert sim.calm(task).lb == {0: 0, 1: 0} and sim.truth(task).segments["JFK"] == (GDP,)


def test_a_crew_past_its_deadline_rests_ten_hours_or_its_flight_auto_cancels_there_and_a_base_has_reserves():
    rests = crewed(flight(0, "BUF", "JFK", 100, 60, lb=60, deadline=900), 130)
    times_out = crewed(flight(1, "DCA", "JFK", 100, 60, lb=60, pax=50, deadline=600), 130)
    reserves = crewed(flight(2, "JFK", "BOS", 100, 60, lb=60), None)
    task = day([rests, times_out, reserves], [tail("N1", "BUF"), tail("N2", "DCA"), tail("N3", "JFK")])
    run = sim.execute(task, {})
    assert [(r.status, r.dep, r.decided) for r in run.flights.values()] == [
        ("flown", 730, None), ("auto_cancelled", None, 130), ("flown", 160, None)]
    assert run.cost.notice == 50 * (130 - 100 + 120) and (task.auto_at(times_out), task.auto_at(rests)) == (130, 900)
    held = [sim.execute(task, {0: {"not_before": d}}).flights[0].dep for d in (0, 729, 731, 900)]
    assert held == [730, 730, 731, 900]
    on_time = sim.execute(replace(task, flights=(replace(rests, lb=0), replace(times_out, lb=30), reserves)), {})
    assert (on_time.flights[0].dep, on_time.flights[1].dep) == (100, 130)
    assert sim.execute(task, {0: {"left": 300}, 1: {"left": 300}}).flights[1].dep == 300
    assert sim.execute(replace(task, start=200), {}).flights[1].decided == 200


def test_a_turn_jetblue_flew_faster_than_the_rule_holds_through_the_moves_its_aircraft_passes():
    run = sim.execute(turned_through_day(), {1: {"cancel": True}})
    assert (run.flights[2].tail, run.flights[2].dep, run.moves) == ("N1", 100, {1: 110})
    assert sim.execute(turned_through_day(None), {1: {"cancel": True}}).flights[2].dep == 110
    assert sim.execute(turned_through_day(40), {1: {"cancel": True}}).flights[2].dep == 110


def test_a_turn_jetblue_flew_faster_than_the_rule_is_allowed_for_that_connection_only():
    run = sim.execute(quick_day(), {})
    assert {i: (r.dep, r.tail) for i, r in run.flights.items()} == {0: (100, "N1"), 1: (170, "N1"), 2: (170, "N2")}
    assert sorted(r.dep for r in sim.execute(quick_day(None), {}).flights.values()) == [100, 170, 190]
    alone = replace(quick_day(), tails=quick_day().tails[:1])
    assert sim.execute(alone, {1: {"cancel": True}}).flights[2].dep == 190
    assert sim.execute(alone, {}).flights[1].dep == 170
    assert sim.execute(quick_day(40), {}).flights[1].dep == sim.execute(quick_day(None), {}).flights[1].dep
