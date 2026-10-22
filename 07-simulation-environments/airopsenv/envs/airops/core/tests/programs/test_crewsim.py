"""The crew sim by hand: trips, time-outs, the extension, reserves (standby and short-call, its cap), the rest move
with its hotel, deadhead seats, swaps, (b)(14), named crews, the day so far, costs and the event log."""

import pytest
from day_helpers import day, flight, tail

from airops_core.core.sim import AUTO_CANCELLED, CANCELLED, FLOWN
from airops_core.programs import crews, crewsim
from airops_core.programs.crews import Unit


def unit(uid: str, kind: str = "pilot", station: str = "JFK", report: int | None = 540, hour: int | None = None,
         trip: tuple = (), home: str = "JFK", end: str | None = None, **kw) -> Unit:
    hour = report // 60 % 24 if hour is None and report is not None else hour
    ready = kw.pop("ready", report + crews.BRIEF if report is not None else 0)
    return Unit(uid, kind, "A32X", crews.size(kind, "A32X"), home, end or station, station, ready, report, hour,
                at=station, trip=tuple(trip), **kw)


def pair(trip: tuple, station: str = "JFK", report: int = 540, **kw) -> list[Unit]:
    return [unit("P1", "pilot", station, report, trip=trip, **kw), unit("C1", "cabin", station, report, trip=trip,
                                                                          **kw)]


def test_trip_crews_fly_their_trip_and_the_log_has_each_duty():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "BOS", "JFK", 720, 60)], [tail("N1", "JFK")])
    d = crewsim.execute(task, {}, pair((0, 1)))
    assert [d.run.flights[i].dep for i in (0, 1)] == [600, 720]
    assert d.crews == {0: {"pilot": "P1", "cabin": "C1"}, 1: {"pilot": "P1", "cabin": "C1"}}
    assert d.uncrewed == () and d.crew_cost == 0
    assert d.units["P1"]["duties"] == [{"report": 540, "at": "JFK", "begins": "on duty", "legs": [(0, 600, 660),
                                        (1, 720, 780)], "deadheads": [], "release": 795}]
    limit = crews.pilot_limit(540, 9, 1) - 60
    assert d.status[0] == [(0, "times out", limit), (600, "flown", None)]
    assert crewsim.statuses(d)[0][0] == {"at": "00:00", "status": "times out", "until": crewsim.clock(limit)}


def test_a_flight_waits_for_a_crew_and_auto_cancels_when_its_crew_times_out():
    task = day([flight(0, "JFK", "BOS", 700, 60, deadline=760)], [tail("N1", "JFK")])
    d = crewsim.execute(task, {}, pair((0,), report=0, hour=0, ready=120, landed=100))
    assert crews.pilot_limit(0, 0, 1) == 660
    assert d.run.flights[0].status == AUTO_CANCELLED
    assert d.status[0][0] == (0, "no crew", None)


def test_the_extension_lets_a_late_leg_leave():
    crew = pair((0,), report=300, hour=5, ready=1020, landed=1000)
    task = day([flight(0, "JFK", "BOS", 1080, 60, deadline=1200)], [tail("N1", "JFK")])
    d = crewsim.execute(task, {}, crew)
    assert d.run.flights[0].dep == 1080 and 1080 + 60 > 300 + crews.fdp(5, 1)
    late = day([flight(0, "JFK", "BOS", 1081, 60, deadline=1200)], [tail("N1", "JFK")])
    assert crewsim.execute(late, {}, crew).run.flights[0].status == AUTO_CANCELLED


def test_table_a_counts_the_actual_blocks_flown():
    flights = [flight(0, "JFK", "BOS", 400, 400, sched_block=300), flight(1, "BOS", "JFK", 900, 150)]
    task = day(flights, [tail("N1", "JFK")])
    d = crewsim.execute(task, {}, pair((0, 1), report=340, hour=5))
    assert d.run.flights[1].status == AUTO_CANCELLED
    assert d.units["P1"]["duties"][0]["legs"] == [(0, 400, 800)]


def test_standby_reserve_takes_the_flight_of_a_timed_out_crew_until_its_shift_ends():
    standby = [unit("RP", "pilot", report=300, reserve="standby", until=660),
               unit("RC", "cabin", report=300, reserve="standby", until=660)]
    crew = [unit("P1", report=0, hour=0, ready=320, landed=300, trip=(0,)), unit("C1", "cabin", trip=(0,))]
    task = day([flight(0, "JFK", "BOS", 600, 90)], [tail("N1", "JFK")])
    d = crewsim.execute(task, {}, crew + standby)
    assert 600 + 90 > crews.pilot_limit(0, 0, 1) and d.crews[0] == {"pilot": "RP", "cabin": "C1"}
    late = day([flight(0, "JFK", "BOS", 670, 90, deadline=800)], [tail("N1", "JFK")])
    d = crewsim.execute(late, {}, crew + standby)
    assert d.run.flights[0].status == AUTO_CANCELLED


def test_short_call_reserve_reports_on_the_hour_its_lead_after_the_call():
    reserve = unit("RP", "pilot", report=None, hour=None, reserve="short", rap=0, until=720, lead=150)
    task = day([flight(0, "JFK", "BOS", 600, 60, deadline=1000)], [tail("N1", "JFK")])
    cabin = [unit("C1", "cabin", trip=(0,))]
    d = crewsim.execute(task, {}, [reserve] + cabin)
    assert d.run.flights[0].dep == 780 + crews.BRIEF
    assert d.units["RP"]["call"] == 600
    assert d.units["RP"]["duties"][0]["report"] == 780 and d.units["RP"]["duties"][0]["begins"] == "call"
    named = {0: {"crew": {"pilot": "RP"}}, "reports": {"RP": 540}}
    d = crewsim.execute(task, named, [reserve] + cabin)
    assert d.run.flights[0].dep == 600 and d.units["RP"]["duties"][0]["report"] == 540


def test_short_call_cap_ends_the_duty_by_its_period_start_plus_sixteen_hours():
    reserve = unit("RP", "pilot", report=None, hour=None, reserve="short", rap=0, until=720, lead=150)
    cabin = [unit("C1", "cabin", report=840, trip=(0, 1))]
    plan = {0: {"crew": {"pilot": "RP"}}, 1: {"crew": {"pilot": "RP"}}, "reports": {"RP": 840}}
    assert crews.pilot_limit(840, 14, 2, 0) == 960 + 120 < 840 + crews.fdp(14, 2) + 120
    for dep, status in ((1020, FLOWN), (1021, AUTO_CANCELLED)):
        task = day([flight(0, "JFK", "BOS", 900, 60), flight(1, "BOS", "JFK", dep, 60, deadline=1100)],
                   [tail("N1", "JFK")])
        d = crewsim.execute(task, plan, [reserve] + cabin)
        assert d.run.flights[0].dep == 900 and d.run.flights[1].status == status


def test_rest_move_flies_next_morning_after_ten_hours_with_a_hotel():
    units = [unit(k, kind, "BOS", 400, 6, ready=1020, landed=1000, trip=(0,) if kind == "pilot" else ())
             for k, kind in (("P1", "pilot"), ("C1", "cabin"))]
    task = day([flight(0, "BOS", "JFK", 1650, 60)], [tail("N1", "BOS")])
    d = crewsim.execute(task, {}, units)
    assert d.run.flights[0].dep == 1665
    for k in ("P1", "C1"):
        duties = d.units[k]["duties"]
        assert duties[0]["release"] == 1015 and duties[-1] == {"report": 1620, "at": "BOS", "begins": "rest",
                                                                "legs": [(0, 1665, 1725)], "deadheads": [],
                                                                "release": 1740}
        assert d.units[k]["hotels"] == [{"station": "BOS", "from": 1015, "to": 1620}]
    assert d.cost["hotel"] == pytest.approx(6 * crews.hotel("BOS", task.date))


def test_deadhead_rides_only_within_free_seats_and_costs_off_its_trip():
    flights = [flight(0, "JFK", "BOS", 600, 60, pax=147), flight(1, "JFK", "DCA", 700, 60)]
    riders = [unit("P2", "pilot", end="BOS"), unit("C2", "cabin", end="BOS")]
    task = day(flights, [tail("N1", "JFK"), tail("N2", "JFK")])
    d = crewsim.execute(task, {0: {"deadhead": ["P2", "C2"]}}, pair((0, 1)) + riders)
    assert d.riders == {0: ("P2",)}
    assert d.cost["deadhead"] == pytest.approx(crews.deadhead("pilot", "A32X", 60, task.date))
    planned = [unit("P2", "pilot", end="BOS", deadheads=(0,))]
    d = crewsim.execute(task, {0: {"deadhead": ["P2"]}}, pair((0, 1)) + planned)
    assert d.riders == {0: ("P2",)} and d.cost["deadhead"] == 0


def test_swap_a_crew_with_no_trip_left_takes_a_stranded_crews_flight():
    task = day([flight(0, "JFK", "DCA", 600, 60)], [tail("N1", "JFK")])
    units = pair((0,), station="BOS") + [unit("P2"), unit("C2", "cabin")]
    d = crewsim.execute(task, {}, units)
    assert d.crews[0] == {"pilot": "P2", "cabin": "C2"}
    assert d.status[0][0] == (0, "no crew", None)
    assert d.cost["end"] == pytest.approx(2 * crews.reposition(units[2], "DCA", task.airports, task.date) / 2
                                          + crews.reposition(units[3], "DCA", task.airports, task.date))


def test_b14_attendants_fly_their_own_late_leg_but_take_no_other_past_fourteen_hours():
    task = day([flight(0, "JFK", "BOS", 900, 60, deadline=950)], [tail("N1", "JFK")])
    pilot = [unit("P1", report=600, trip=(0,))]
    own = [unit("C1", "cabin", report=0, hour=0, ready=320, landed=300, trip=(0,))]
    assert crewsim.execute(task, {}, pilot + own).run.flights[0].dep == 900
    other = [unit("C1", "cabin", report=0, hour=0, ready=320, landed=300, trip=())]
    assert crewsim.execute(task, {}, pilot + other).run.flights[0].status == AUTO_CANCELLED


def test_a_named_crew_is_waited_for():
    flights = [flight(0, "BOS", "JFK", 500, 60), flight(1, "JFK", "DCA", 570, 60)]
    task = day(flights, [tail("N1", "BOS"), tail("N2", "JFK")])
    units = pair((1,), report=500) + [unit("P2", station="BOS", report=400, trip=(0,)),
                                      unit("C2", "cabin", "BOS", 400, trip=(0,))]
    d = crewsim.execute(task, {1: {"crew": {"pilot": "P2"}}}, units)
    assert d.run.flights[1].dep == 560 + crews.CONNECT and d.crews[1] == {"pilot": "P2", "cabin": "C1"}


def test_the_day_so_far_leaves_without_a_crew_and_says_so():
    task = day([flight(0, "JFK", "BOS", 600, 60)], [tail("N1", "JFK")])
    d = crewsim.execute(task, {0: {"left": 600}}, [])
    assert d.run.flights[0].status == FLOWN and d.uncrewed == (0,)
    assert d.status[0][-1] == (600, "flown without a legal crew", None)


def test_cancelled_flights_leave_their_crews_and_log_it():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "JFK", "DCA", 700, 60)], [tail("N1", "JFK")])
    d = crewsim.execute(task, {0: {"cancel": True}}, pair((0, 1)))
    assert d.crews[1] == {"pilot": "P1", "cabin": "C1"}
    assert d.status[0][-1] == (0, "cancelled", None)


def test_a_flight_waiting_for_a_crew_is_woken_by_its_arrival():
    flights = [flight(0, "JFK", "DCA", 500, 60), flight(1, "BOS", "JFK", 520, 60), flight(2, "JFK", "MCO", 700, 120)]
    task = day(flights, [tail("N1", "JFK"), tail("N2", "BOS"), tail("N3", "JFK")])
    units = [unit(k, kind, "BOS", 400, trip=(1, 0)) for k, kind in (("P1", "pilot"), ("C1", "cabin"))] + \
        [unit(k, kind, "JFK", 600, trip=(2,)) for k, kind in (("P2", "pilot"), ("C2", "cabin"))]
    d = crewsim.execute(task, {}, units)
    assert [d.run.flights[i].dep for i in (0, 1, 2)] == [600, 520, 700]
    assert d.crews[0] == {"pilot": "P1", "cabin": "C1"} and d.crews[2] == {"pilot": "P2", "cabin": "C2"}


def test_the_audit_follows_a_diversion_to_where_the_flight_ended():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "PWM", "JFK", 760, 70)],
               [tail("N1", "JFK"), tail("N2", "PWM")])
    crew = pair((0, 1))
    plan = {i: {"crew": {"pilot": "P1", "cabin": "C1"}} for i in (0, 1)}
    deps, blocks = {0: 600, 1: 760}, {0: 90, 1: 70}
    assert crewsim.audit(task, crew, plan, deps, blocks, ends={0: "PWM"}) == []
    assert crewsim.audit(task, crew, plan, deps, blocks) == ["B6 101 PWM-JFK at 760: pilot P1",
                                                             "B6 101 PWM-JFK at 760: cabin C1"]


def test_the_trip_rule_waits_for_the_trips_crew_and_never_takes_a_crew_off_its_trip():
    flights = [flight(0, "JFK", "BOS", 600, 60, deadline=700), flight(1, "JFK", "DCA", 800, 60, deadline=1000)]
    task = day(flights, [tail("N1", "JFK"), tail("N2", "JFK")])
    busy = [unit("P2", report=500, trip=(1,)), unit("C2", "cabin", report=500, trip=(1,))]
    d = crewsim.execute(task, {}, pair((0,), station="BUF") + busy)
    assert d.run.flights[0].status == AUTO_CANCELLED and d.crews[1] == {"pilot": "P2", "cabin": "C2"}
    assert d.status[0][-1] == (task.auto_at(task.flight(0)), "cancelled", None)
    d = crewsim.execute(task, {}, pair((0,), report=580) + busy)
    assert d.crews[0] == {"pilot": "P1", "cabin": "C1"} and d.run.flights[0].dep == 580 + crews.BRIEF
    assert d.crews[1] == {"pilot": "P2", "cabin": "C2"} and d.run.flights[1].dep == 800
    done = [unit("P6", report=400, ready=520, landed=500), unit("C6", "cabin", report=400, ready=520, landed=500)]
    d = crewsim.execute(task, {}, pair((0,), station="BUF") + busy + done)
    assert d.crews[0] == {"pilot": "P6", "cabin": "C6"} and d.crews[1] == {"pilot": "P2", "cabin": "C2"}


def test_a_reserve_is_called_only_to_report_by_its_periods_end_plus_its_lead():
    task = day([flight(0, "JFK", "DCA", 700, 60, deadline=1200)], [tail("N1", "JFK")])
    late = [unit(k, kind, report=None, hour=None, reserve="short", rap=0, until=720, lead=150)
            for k, kind in (("RP", "pilot"), ("RC", "cabin"))]
    assert crews.on_hour(700 + 150) - 150 > 720
    assert crewsim.execute(task, {}, late).run.flights[0].status == AUTO_CANCELLED


def test_a_called_reserve_flies_on_from_where_its_flight_took_it():
    flights = [flight(0, "JFK", "DCA", 600, 60, deadline=1200), flight(1, "DCA", "JFK", 950, 60, deadline=1400)]
    task = day(flights, [tail("N1", "JFK")])
    reserves = [unit(k, kind, report=None, hour=None, reserve="short", rap=0, until=720, lead=150)
                for k, kind in (("RP", "pilot"), ("RC", "cabin"))]
    d = crewsim.execute(task, {}, reserves)
    assert d.run.flights[0].dep == 825 and d.crews[1] == {"pilot": "RP", "cabin": "RC"}


def test_a_flight_holding_a_reserve_leaves_with_its_crew_on_arrival_and_frees_the_reserve():
    flights = [flight(0, "BUF", "JFK", 700, 80), flight(1, "JFK", "BOS", 600, 60, deadline=1000),
               flight(2, "JFK", "DCA", 810, 60, deadline=1300)]
    task = day(flights, [tail("N0", "BUF"), tail("N1", "JFK"), tail("N2", "JFK")])
    inbound = pair((0, 1), station="BUF", report=600)
    reserves = [unit(k, kind, report=None, hour=None, reserve="short", rap=0, until=900, lead=150)
                for k, kind in (("RP", "pilot"), ("RC", "cabin"))]
    d = crewsim.execute(task, {}, inbound + reserves)
    assert crews.on_hour(600 + 150) + crews.BRIEF > 780 + crews.CONNECT
    assert d.run.flights[1].dep == 780 + crews.CONNECT and d.crews[1] == {"pilot": "P1", "cabin": "C1"}
    assert d.crews[2] == {"pilot": "RP", "cabin": "RC"} and d.units["RP"]["call"] == 810
    assert d.run.flights[2].dep == crews.on_hour(810 + 150) + crews.BRIEF


def test_a_second_period_named_keeps_an_uncalled_reserves_short_call_cap():
    reserve = unit("RP", "pilot", report=None, hour=None, reserve="short", rap=120, until=900, lead=150)
    task = day([flight(0, "JFK", "DCA", 1005, 240, deadline=1200)], [tail("N1", "JFK")])
    cabin = [unit("C1", "cabin", report=960, trip=(0,))]
    assert 1005 + 240 > crews.pilot_limit(960, 16, 1, 120)
    for reports in ({"RP": 960}, {"RP": 960, "RP#2": 1200}):
        plan = {0: {"crew": {"pilot": "RP"}}, "reports": reports}
        assert crewsim.execute(task, plan, [reserve] + cabin).run.flights[0].status == AUTO_CANCELLED
        assert crewsim.audit(task, [reserve] + cabin, plan, {0: 1005}, {0: 240}) == ["B6 100 JFK-DCA at 1005: pilot RP"]


def test_a_ride_a_full_rest_after_the_release_pays_a_night_away_from_base():
    flights = [flight(0, "DCA", "JFK", 1100, 60, pax=100), flight(1, "DCA", "JFK", 1300, 60, pax=100)]
    task = day(flights, [tail("N1", "DCA"), tail("N2", "DCA")])
    crew = pair((0,), station="DCA", report=1000) + [unit("P2", station="DCA", report=1200, trip=(1,)),
                                                     unit("C2", "cabin", station="DCA", report=1200, trip=(1,))]
    rider = [unit("P9", station="DCA", report=400, ready=605, landed=585, flown=2, end="JFK")]
    for i, hotels in ((0, []), (1, [{"station": "DCA", "from": 600, "to": 1200}])):
        d = crewsim.execute(task, {i: {"deadhead": ["P9"]}}, crew + rider)
        assert d.riders[i] == ("P9",) and d.units["P9"]["hotels"] == hotels
        assert d.units["P9"]["cost"]["hotel"] == pytest.approx(2 * crews.hotel("DCA", task.date) if hotels else 0)


def test_the_end_of_the_day_sends_a_bases_crews_to_its_trips_tonight_at_the_least_cost():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "JFK", "DCA", 600, 60)],
               [tail("N1", "JFK"), tail("N2", "JFK")])
    units = [unit("PA", trip=(0,), end="DCA"), unit("PB", trip=(1,), end="BOS"),
             unit("CA", "cabin", trip=(0,), end="BOS"), unit("CB", "cabin", trip=(1,), end="DCA")]
    d = crewsim.execute(task, {}, units)
    assert d.cost["end"] == 0.0
    ends = crews.Ends(units, task.airports, task.date)
    alone = crews.reposition(units[0], "BOS", task.airports, task.date)
    shares = ends.cost({"PA": "BOS", "PB": "BOS", "CA": "BOS", "CB": "DCA"})
    assert sorted([shares["PA"], shares["PB"]]) == [0.0, pytest.approx(crews.reposition(units[0], "BOS", task.airports,
                                                                                         task.date, "DCA"))]
    assert alone > 0 and shares["CA"] == shares["CB"] == 0.0
    assert ends.least(units[0], "BOS") == 0.0 and ends.least(units[0], "BUF") == pytest.approx(
        min(crews.reposition(units[0], "BUF", task.airports, task.date, to) for to in ("BOS", "DCA")))


def standbys(until: int = 660, report: int = 300) -> list[Unit]:
    return [unit(k, kind, report=report, reserve="standby", until=until) for k, kind in (("SP", "pilot"),
                                                                                        ("SC", "cabin"))]


def test_the_desk_keeps_the_planned_crew_and_waits_for_it_unless_a_cheaper_crew_is_there():
    task = day([flight(0, "JFK", "BOS", 600, 60)], [tail("N1", "JFK")])
    d = crewsim.execute(task, {}, pair((0,)) + standbys(), desk=True)
    assert d.run.flights[0].dep == 600 and d.crews[0] == {"pilot": "P1", "cabin": "C1"}
    d = crewsim.execute(task, {}, pair((0,), report=580), desk=True)
    assert d.run.flights[0].dep == 580 + crews.BRIEF and d.crews[0] == {"pilot": "P1", "cabin": "C1"}
    d = crewsim.execute(task, {}, pair((0,), report=580) + standbys(), desk=True)
    assert d.run.flights[0].dep == 600 and d.crews[0] == {"pilot": "SP", "cabin": "SC"}


def test_the_desk_gives_a_stranded_crews_trip_to_a_standby_else_calls_a_reserve_to_report_in_time():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "BOS", "JFK", 720, 60)], [tail("N1", "JFK")])
    stranded = pair((0, 1), station="BUF")
    d = crewsim.execute(task, {}, stranded + standbys(), desk=True)
    assert [d.run.flights[i].dep for i in (0, 1)] == [600, 720]
    assert d.crews[0] == d.crews[1] == {"pilot": "SP", "cabin": "SC"}
    reserves = [unit(k, kind, report=None, hour=None, reserve="short", rap=0, until=720, lead=150)
                for k, kind in (("RP", "pilot"), ("RC", "cabin"))]
    d = crewsim.execute(task, {}, stranded + reserves, desk=True)
    assert [d.run.flights[i].dep for i in (0, 1)] == [600, 720] and d.crews[1] == {"pilot": "RP", "cabin": "RC"}
    assert d.units["RP"]["call"] == 540 - 150 and d.units["RP"]["duties"][0]["report"] == 540
    assert crewsim.execute(task, {}, stranded + reserves).run.flights[0].dep == 780 + crews.BRIEF


def test_the_desk_swaps_trips_with_a_crew_there_whose_trip_the_late_crew_can_fly():
    flights = [flight(0, "JFK", "BOS", 600, 60), flight(1, "JFK", "DCA", 800, 60),
               flight(2, "BUF", "JFK", 500, 100, sched_block=60)]
    task = day(flights, [tail("N1", "JFK"), tail("N2", "JFK"), tail("N3", "BUF")])
    late = [unit(k, kind, "BUF", 440, trip=(2, 0)) for k, kind in (("P1", "pilot"), ("C1", "cabin"))]
    there = [unit(k, kind, report=540, trip=(1,)) for k, kind in (("P2", "pilot"), ("C2", "cabin"))]
    d = crewsim.execute(task, {}, late + there, desk=True)
    assert d.run.flights[0].dep == 600 and d.crews[0] == {"pilot": "P2", "cabin": "C2"}
    assert d.run.flights[1].dep == 800 and d.crews[1] == {"pilot": "P1", "cabin": "C1"}
    trip = crewsim.execute(task, {}, late + there)
    assert trip.run.flights[0].dep == 600 + crews.CONNECT and trip.crews[0] == {"pilot": "P1", "cabin": "C1"}
    task = day([flights[0], flights[2]], [tail("N1", "JFK"), tail("N3", "BUF")])
    spare = [unit(k, kind, report=540) for k, kind in (("P3", "pilot"), ("C3", "cabin"))]
    d = crewsim.execute(task, {}, late + spare, desk=True)
    assert d.run.flights[0].dep == 600 and d.crews[0] == {"pilot": "P3", "cabin": "C3"}


def test_the_desk_deadheads_the_first_crew_that_can_reach_a_stranded_crews_flight():
    flights = [flight(0, "JFK", "BOS", 700, 60), flight(1, "BOS", "JFK", 560, 60), flight(3, "JFK", "DCA", 760, 60)]
    task = day(flights, [tail("N1", "JFK"), tail("N2", "BOS")])
    units = pair((0,), station="BUF") + [unit(k, kind, "BOS", 480, trip=(1, 3)) for k, kind in
                                         (("P3", "pilot"), ("C3", "cabin"))] + \
        [unit(k, kind, "BOS", 480) for k, kind in (("P2", "pilot"), ("C2", "cabin"))]
    d = crewsim.execute(task, {}, units, desk=True)
    assert d.run.flights[0].dep == 700 and d.crews[0] == {"pilot": "P2", "cabin": "C2"}
    assert d.riders[1] == ("C2", "P2") and d.cost["deadhead"] > 0
    assert crewsim.execute(task, {}, units).run.flights[0].status == AUTO_CANCELLED


def test_the_desk_extends_a_pilots_duty_only_when_nothing_else_makes_the_flight_legal():
    crew = pair((0,), report=300, hour=5, ready=1020, landed=1000)
    task = day([flight(0, "JFK", "BOS", 1080, 60, deadline=1200)], [tail("N1", "JFK")])
    assert 1080 + 60 > 300 + crews.fdp(5, 1) and 1080 + 60 <= crews.pilot_limit(300, 5, 1)
    d = crewsim.execute(task, {}, crew, desk=True)
    assert d.run.flights[0].dep == 1080 and d.crews[0] == {"pilot": "P1", "cabin": "C1"}
    d = crewsim.execute(task, {}, crew + standbys(until=1260, report=900), desk=True)
    assert d.run.flights[0].dep == 1080 and d.crews[0] == {"pilot": "SP", "cabin": "C1"}


def test_the_desk_cancels_a_flight_no_crew_can_reach_as_soon_as_that_is_certain_at_that_notice():
    task = day([flight(0, "JFK", "BOS", 600, 60)], [tail("N1", "JFK")])
    d = crewsim.execute(task, {}, pair((0,), station="BUF"), desk=True)
    trip = crewsim.execute(task, {}, pair((0,), station="BUF"))
    assert (d.run.flights[0].status, d.run.flights[0].decided) == (CANCELLED, 0)
    assert trip.run.flights[0].status == AUTO_CANCELLED and d.run.cost.total < trip.run.cost.total
    flights = [flight(0, "JFK", "BOS", 400, 400, sched_block=300), flight(1, "BOS", "JFK", 900, 150)]
    task = day(flights, [tail("N1", "JFK")])
    d = crewsim.execute(task, {}, pair((0, 1), report=340, hour=5), desk=True)
    assert (d.run.flights[1].status, d.run.flights[1].decided) == (CANCELLED, 400)


def test_the_desk_relieves_a_late_crew_only_when_waiting_costs_more_than_the_end_of_the_day_the_move_shifts():
    task = day([flight(0, "JFK", "BOS", 600, 60)], [tail("N1", "JFK")])
    for report, dep, crew in ((580, 580 + crews.BRIEF, ("P1", "C1")), (615, 600, ("SP", "SC"))):
        late = [unit(k, kind, report=report, home="BOS", end="BOS", trip=(0,)) for k, kind in (("P1", "pilot"),
                                                                                              ("C1", "cabin"))]
        d = crewsim.execute(task, {}, late + standbys(), desk=True)
        assert d.run.flights[0].dep == dep and d.crews[0] == dict(zip(crews.KINDS, crew, strict=True))
    pilot = unit("SP", report=300, reserve="standby", until=660)
    moved = 2 * crews.reposition(pilot, "BOS", task.airports, task.date, "JFK")
    assert 100 * 25 < moved < 100 * 60


def test_the_desk_prices_a_rest_away_from_base_into_waiting_for_a_crew():
    units = [unit("P1", "pilot", "BOS", 400, 6, ready=1020, landed=1000, trip=(0,)),
             unit("C1", "cabin", "BOS", 1500, trip=(0,))]
    task = day([flight(0, "BOS", "JFK", 1650, 60)], [tail("N1", "BOS")])
    pool = crewsim.Pool(task, units, lambda s, m: m, {}, True, False, task.airports, desk=True)
    own = [o for o in pool.options("pilot", "A32X", task.flight(0), 0, 1650) if o[3] == "P1"]
    assert own == [(100 * 15 + crews.hotel("BOS", task.date) * 2, 0, 1665, "P1", ("own",))]


def test_the_extension_is_open_to_any_crew_it_alone_makes_legal():
    task = day([flight(0, "JFK", "BOS", 1080, 60, deadline=1200)], [tail("N1", "JFK")])
    units = [unit("P1", "pilot", "BUF", trip=(0,)), unit("C1", "cabin", report=900, trip=(0,)),
             unit("SP", "pilot", report=300, hour=5, reserve="standby", until=1260)]
    assert 1080 + 60 > 300 + crews.fdp(5, 1)
    d = crewsim.execute(task, {}, units, desk=True)
    assert d.run.flights[0].dep == 1080 and d.crews[0] == {"pilot": "SP", "cabin": "C1"}


def test_the_desk_does_not_cancel_a_flight_a_crew_at_its_origin_comes_back_to_fly():
    flights = [flight(0, "JFK", "BOS", 600, 60, deadline=900), flight(1, "JFK", "PWM", 450, 60),
               flight(2, "PWM", "JFK", 570, 60)]
    task = day(flights, [tail("N1", "JFK"), tail("N2", "JFK")])
    units = pair((0,), station="BUF") + [unit(k, kind, report=390, trip=(1, 2)) for k, kind in (("P2", "pilot"),
                                                                                                 ("C2", "cabin"))]
    d = crewsim.execute(task, {}, units, desk=True)
    assert d.run.flights[0].dep == 630 + crews.CONNECT and d.crews[0] == {"pilot": "P2", "cabin": "C2"}


def test_a_reserve_the_morning_plan_called_logs_its_call():
    task = day([flight(0, "JFK", "BOS", 705, 60)], [tail("N1", "JFK")], start=600)
    called = [unit(k, kind, report=660, reserve="short", rap=120, until=840, lead=150, trip=(0,))
              for k, kind in (("RP", "pilot"), ("RC", "cabin"))]
    d = crewsim.execute(task, {}, called)
    assert d.run.flights[0].dep == 705 and d.units["RP"]["call"] == 660 - 150


def test_without_drop_the_desk_asks_ops_to_cancel_and_the_flight_stays_on_the_board():
    task = day([flight(0, "JFK", "BOS", 600, 60)], [tail("N1", "JFK")])
    d = crewsim.execute(task, {}, pair((0,), station="BUF"), desk=True, drop=False)
    assert d.run.flights[0].status == AUTO_CANCELLED and d.dropped == {}
    assert d.messages == ({"sender": "crew", "receiver": "ops", "type": crewsim.CANCEL,
                           "params": {"flight": 0, "kind": "pilot", "minute": None, "reason": "none"}, "text": "",
                           "sent": 0, "reply": None, "replied": None, "actions": []},)
    d = crewsim.execute(task, {}, pair((0,), station="BUF"), desk=True)
    assert d.dropped == {0: 0} and [(m["type"], m["reply"], m["replied"], m["actions"]) for m in d.messages] == [
        (crewsim.CANCEL, "accept", 0, [{"cancel": 0, "at": 0}])]


def test_the_desk_says_hold_until_its_crew_can_leave_and_says_it_again_only_when_that_changes():
    task = day([flight(0, "JFK", "BOS", 600, 60)], [tail("N1", "JFK")])
    d = crewsim.execute(task, {}, pair((0,), report=580), desk=True, drop=False)
    assert d.run.flights[0].dep == 580 + crews.BRIEF
    assert [(m["sent"], m["type"], m["params"]["minute"], m["params"]["reason"]) for m in d.messages] == [
        (0, crewsim.HOLD, 580 + crews.BRIEF, "later")]
    assert crewsim.execute(task, {}, pair((0,), report=500), desk=True, drop=False).messages == ()


def test_with_no_crew_until_a_trip_brings_one_the_desk_holds_until_the_first_is_there():
    flights = [flight(0, "JFK", "BOS", 600, 60, deadline=900), flight(1, "JFK", "PWM", 450, 60),
               flight(2, "PWM", "JFK", 570, 60), flight(3, "PWM", "JFK", 640, 60)]
    task = day(flights, [tail("N1", "JFK"), tail("N2", "JFK"), tail("N3", "PWM")])
    units = pair((0,), station="BUF") + [unit(k, kind, report=390, trip=(1, 2)) for k, kind in (("P2", "pilot"),
                                                                                                 ("C2", "cabin"))] + \
        [unit(k, kind, "PWM", 580, trip=(3,)) for k, kind in (("P3", "pilot"), ("C3", "cabin"))]
    pool = crewsim.Pool(task, units, lambda s, m: m, {}, True, False, task.airports, desk=True, drop=False)
    assert pool.coming("pilot", "A32X", task.flight(0)) == 630
    d = crewsim.execute(task, {}, units, desk=True, drop=False)
    assert (d.messages[0]["sent"], d.messages[0]["type"], d.messages[0]["params"]["minute"],
            d.messages[0]["params"]["reason"]) == (0, crewsim.HOLD, 630 + crews.CONNECT, "coming")
    assert d.run.flights[0].dep == 630 + crews.CONNECT and d.crews[0] == {"pilot": "P2", "cabin": "C2"}


def test_the_flag_follows_the_crew_the_desk_takes_and_counts_the_deadhead_it_booked():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "BOS", "JFK", 720, 60)], [tail("N1", "JFK")])
    d = crewsim.execute(task, {}, pair((0, 1), station="BUF") + standbys(), desk=True)
    cabin = 300 + crews.CABIN_DUTY - crews.DEBRIEF - 60
    assert d.status[0][:2] == [(0, crewsim.NO_CREW, None), (0, crewsim.TIMES_OUT, cabin)]
    flights = [flight(0, "JFK", "BOS", 700, 60), flight(1, "BOS", "JFK", 560, 60), flight(3, "JFK", "DCA", 760, 60)]
    task = day(flights, [tail("N1", "JFK"), tail("N2", "BOS")])
    units = pair((0,), station="BUF") + [unit(k, kind, "BOS", 480, trip=(1, 3)) for k, kind in
                                         (("P3", "pilot"), ("C3", "cabin"))] + \
        [unit(k, kind, "BOS", 480) for k, kind in (("P2", "pilot"), ("C2", "cabin"))]
    d = crewsim.execute(task, {}, units, desk=True)
    assert d.riders[1] == ("C2", "P2") and d.status[0][0] == (0, crewsim.NO_CREW, None)
    assert d.status[0][1] == (0, crewsim.TIMES_OUT, 480 + crews.CABIN_DUTY - crews.DEBRIEF - 60)
    pool = crewsim.Pool(task, units, lambda s, m: m, {}, True, False, task.airports, desk=True)
    p2, f = pool.state["P2"], task.flight(0)
    assert p2.deadheads == [1] and pool.outlook(p2, f) == (crewsim.NO_CREW, None)
    assert pool.flag(p2, f, 0) == (crewsim.TIMES_OUT, crews.pilot_limit(480, 8, 1) - 60)


def test_without_drop_the_desk_crews_alike_where_it_never_had_to_cancel():
    task = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "BOS", "JFK", 720, 60)], [tail("N1", "JFK")])
    swap = day([flight(0, "JFK", "BOS", 600, 60), flight(1, "JFK", "DCA", 800, 60),
                flight(2, "BUF", "JFK", 500, 100, sched_block=60)],
               [tail("N1", "JFK"), tail("N2", "JFK"), tail("N3", "BUF")])
    late = [unit(k, kind, "BUF", 440, trip=(2, 0)) for k, kind in (("P1", "pilot"), ("C1", "cabin"))]
    there = [unit(k, kind, report=540, trip=(1,)) for k, kind in (("P2", "pilot"), ("C2", "cabin"))]
    for t, units in ((task, pair((0, 1), station="BUF") + standbys()), (task, pair((0, 1), report=580) + standbys()),
                     (swap, late + there)):
        a, b = (crewsim.execute(t, {}, units, desk=True, drop=drop) for drop in (True, False))
        assert a.run.rows() == b.run.rows() and a.crews == b.crews and a.units == b.units
        assert a.status == b.status and a.messages == b.messages
