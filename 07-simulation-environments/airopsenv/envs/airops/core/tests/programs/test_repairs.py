from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from airops_core.build.fleet import Aircraft
from airops_core.programs import repairs
from airops_core.programs.model import Connection, Crew, Flight, Outcome

ECFR_TABLE_B = """
0000-0359 | 9 | 9 | 9 | 9 | 9 | 9 | 9
0400-0459 | 10 | 10 | 10 | 10 | 9 | 9 | 9
0500-0559 | 12 | 12 | 12 | 12 | 11.5 | 11 | 10.5
0600-0659 | 13 | 13 | 12 | 12 | 11.5 | 11 | 10.5
0700-1159 | 14 | 14 | 13 | 13 | 12.5 | 12 | 11.5
1200-1259 | 13 | 13 | 13 | 13 | 12.5 | 12 | 11.5
1300-1659 | 12 | 12 | 12 | 12 | 11.5 | 11 | 10.5
1700-2159 | 12 | 12 | 11 | 11 | 10 | 9 | 9
2200-2259 | 11 | 11 | 10 | 10 | 9 | 9 | 9
2300-2359 | 10 | 10 | 10 | 9 | 9 | 9 | 9
"""
ZONES = {"BUF": "America/New_York", "JFK": "America/New_York", "RDU": "America/New_York", "PWM": "America/New_York",
         "BOS": "America/New_York", "LAX": "America/Los_Angeles", "SJU": "America/Puerto_Rico",
         "ORD": "America/Chicago"}


@pytest.mark.parametrize(("model", "seats", "family"), [
    ("A320-232", 162, "A320"), ("A320-232", 150, "A320"), ("A321-231", 190, "A321"), ("A321-271NX", 200, "A321"),
    ("A321-231", 159, "A321P"), ("A321-271NX", 138, "A321P"), ("BD-500-1A11", 140, "A220"), ("E190-100", 100, "E190"),
])
def test_the_a32x_family_splits_by_seats_and_type(model, seats, family):
    old = "A32X" if "A32" in model else "A220" if "BD-500" in model else "E190"
    assert repairs.family(Aircraft("N1", old, model, seats, "B-43 2024")) == family


def test_schedule_turns_are_the_fifth_percentile_floored():
    rows = []
    for k in range(40):
        rows += [{"Tail_Number": f"N{k}", "FlightDate": "2024-07-01", "Origin": "JFK", "Dest": "BOS",
                  "sched_dep": 600, "sched_arr": 680},
                 {"Tail_Number": f"N{k}", "FlightDate": "2024-07-01", "Origin": "BOS", "Dest": "JFK",
                  "sched_dep": 680 + 20 + k, "sched_arr": 800}]
    fleet = {f"N{k}": Aircraft(f"N{k}", "E190", "E190-100", 100, "B-43 2024") for k in range(40)}
    turns = repairs.turns(pd.DataFrame(rows), fleet)
    assert turns["E190"] == 25 and set(turns) == set(repairs.FAMILIES)
    fleet["N39"] = Aircraft("N39", "A32X", "A320-232", 162, "B-43 2024")
    assert repairs.turns(pd.DataFrame(rows), fleet)["A320"] == 59


@pytest.mark.parametrize(("dep", "late", "nas", "removed", "lb"), [
    (60, 20, 30, False, 40), (60, 20, 30, True, 10), (5, 0, 60, True, 0), (-8, np.nan, np.nan, True, -8),
    (-30, np.nan, np.nan, False, -15), (45, np.nan, 50, True, 0),
])
def test_lb_drops_nas_delay_but_never_departs_before_the_flight_did(dep, late, nas, removed, lb):
    row = SimpleNamespace(DepDelay=dep, LateAircraftDelay=late, NASDelay=nas)
    assert repairs.lb_of(row, removed) == lb
    frame = pd.DataFrame([{"DepDelay": dep, "LateAircraftDelay": late, "NASDelay": nas, "Cancelled": 0}])
    assert repairs.new_lb(frame, pd.Series([removed])).iloc[0] == lb


def test_a_cancelled_flight_takes_the_upper_quartile_by_destination_then_origin_then_day():
    pool = pd.DataFrame([{"Origin": o, "Dest": d, "sd": sd, "lb": lb} for o, d, sd, lb in [
        ("BOS", "JFK", 600, 10), ("DCA", "JFK", 630, 20), ("ORD", "JFK", 660, 30), ("ORD", "JFK", 640, 40),
        ("MCO", "JFK", 900, 400), ("PIT", "ATL", 610, 5), ("PIT", "LAX", 620, 15), ("PIT", "MIA", 630, 25),
    ]])
    assert repairs.imputed_lb(pool, "BOS", "JFK", 620) == 33
    assert repairs.imputed_lb(pool, "PIT", "SEA", 620) == 20
    assert repairs.imputed_lb(pool, "SAV", "SEA", 300) == 33


def test_the_truth_block_is_clipped_and_a_diverted_flight_takes_its_routes_median():
    day = pd.DataFrame([{"Origin": "MKE", "Dest": "JFK", "sd": sd, "ad": sd + 5, "aa": sd + 5 + b, "Cancelled": 0,
                         "Diverted": 0, "held_taxi": 0} for sd, b in ((600, 130), (630, 150), (650, 140), (900, 100))])
    flown = SimpleNamespace(Origin="MKE", Dest="JFK", sd=600, ad=610, aa=1000, CRSElapsedTime=130, Cancelled=0,
                            Diverted=0, DivReachedDest=np.nan, held_taxi=0)
    assert repairs.truth_block(flown, day) == 250
    diverted = SimpleNamespace(**{**vars(flown), "Diverted": 1, "aa": 2058})
    assert repairs.truth_block(diverted, day) == 140
    reached = SimpleNamespace(**{**vars(flown), "Diverted": 1, "DivReachedDest": 1, "aa": 900})
    assert repairs.truth_block(reached, day) == 140
    assert repairs.truth_block(SimpleNamespace(**{**vars(flown), "Cancelled": 1, "Origin": "SAV"}), day) == 130


def test_a_flight_that_pushed_back_during_a_stop_loses_the_taxi_out_the_stop_held_from_its_block():
    assert repairs.held(1187, 1261, 95.0, 15.0) == 74
    assert repairs.held(1187, 1261, 40.0, 15.0) == 25
    assert repairs.held(1187, 1187, 95.0, 15.0) == repairs.held(1187, 1261, 10.0, 15.0) == 0
    rdu = SimpleNamespace(Origin="RDU", Dest="JFK", sd=1065, ad=1187, aa=1393, CRSElapsedTime=105, Cancelled=0,
                          Diverted=0, DivReachedDest=np.nan, held_taxi=74)
    assert 1261 + repairs.truth_block(rdu, pd.DataFrame()) == rdu.aa
    route = pd.DataFrame([{"Origin": "RDU", "Dest": "JFK", "sd": 1070, "ad": 1190, "aa": 1390, "Cancelled": 0,
                           "Diverted": 0, "held_taxi": 70}])
    assert repairs.truth_block(SimpleNamespace(**{**vars(rdu), "Diverted": 1}), route) == 130


def test_deadlines_run_dmax_past_the_schedule_capped_at_the_hub_cutoff():
    assert repairs.deadline(600, "MCO", 540) == 1500
    assert repairs.deadline(900, "JFK", 540) == repairs.CUTOFF
    assert repairs.deadline(100, "MCO", 1080, dmax=900) == 1140
    assert repairs.deadline(100, "MCO", 1080, dmax=360) == 1140


def test_out_and_backs_pair_from_a_hub():
    legs = [(0, "JFK", "BOS"), (1, "BOS", "JFK"), (2, "JFK", "MCO"), (3, "MCO", "FLL"), (4, "FLL", "MCO"),
            (5, "MCO", "JFK"), (6, "JFK", "MCO")]
    assert repairs.pairs(legs) == {0: 1, 1: 0}
    assert repairs.pairs([(7, "MCO", "JFK"), (8, "JFK", "MCO"), (9, "MCO", "JFK")]) == {8: 9, 9: 8}


def test_every_chain_connects_or_repositions_and_pax_fit_the_smallest_tail(task):
    seats = defaultdict(list)
    for t in task.tails:
        seats[t.family].append(t.seats)
        legs = [task.flight(i) for i in t.legs]
        assert not legs or legs[0].orig == t.pos
        breaks = [(a.id, a.dest, b.orig) for a, b in zip(legs, legs[1:], strict=False) if a.dest != b.orig]
        assert breaks == [(m.leg, m.orig, m.dest) for m in t.repositions]
        assert all(b.leg in t.legs[:-1] and b.ready > task.flight(b.leg).sched_arr for b in t.busy)
    assert all(f.pax == min(f.pax_raw, *seats[f.family]) for f in task.flights)
    assert {f.family for f in task.flights} <= set(task.turn)


def test_table_b_is_the_ecfrs_unaugmented_flight_duty_period_by_local_start_hour_and_segments():
    for line in ECFR_TABLE_B.strip().splitlines():
        band, *hours = line.split(" | ")
        for hour in range(int(band[:2]), int(band[5:7]) + 1):
            assert [repairs.fdp(hour, n) for n in range(1, 10)] == [round(60 * float(h))
                                                                    for h in hours + hours[-1:] * 2]
    assert (repairs.fdp(6, 3), repairs.fdp(12, 5), repairs.fdp(21, 7), repairs.fdp(23, 4)) == (720, 750, 540, 540)
    assert repairs.ECFR == "https://www.ecfr.gov/current/title-14/chapter-I/subchapter-G/part-117"
    assert all(c in repairs.__doc__ for c in ("14 CFR 117 Table B", "Unaugmented Operations", "117.19(a)(1)",
                                               "117.25(e)"))
    assert (repairs.BASES, repairs.REPORT, repairs.EXTENSION, repairs.REST) == (("JFK", "BOS", "FLL", "MCO"), 45, 120,
                                                                               600)


def test_a_pairing_runs_from_a_base_departure_to_the_next_base_arrival_and_only_outstations_have_a_crew_deadline():
    legs = pd.DataFrame([(11, "N1", "RDU", "PWM", 660, 110), (10, "N1", "JFK", "RDU", 500, 100),
                         (12, "N1", "PWM", "BOS", 830, 60), (13, "N1", "BOS", "JFK", 1000, 70),
                         (14, "N1", "BUF", "JFK", 360, 80), (20, "N2", "LAX", "JFK", 30, 330),
                         (21, "N2", "JFK", "SJU", 300, 225), (22, "N2", "SJU", "JFK", 600, 240),
                         (23, "N2", "ORD", "BOS", 800, 150)],
                        columns=["label", "Tail_Number", "Origin", "Dest", "sd", "CRSElapsedTime"]).set_index("label")
    base = int(datetime(2024, 7, 1, tzinfo=ZoneInfo("America/New_York")).timestamp() // 60)

    def crew(pairing, report, station, segments, hours, deadline):
        return Crew(pairing, report, station, segments, 60 * hours, deadline,
                    None if deadline is None else deadline + 600)

    assert repairs.crews(legs, base, repairs.BASES, ZONES) == {
        14: crew("N1/0", 315, "BUF", 1, 12, 315 + 720 + 120 - 80),
        10: crew("N1/1", 455, "JFK", 3, 13, None), 11: crew("N1/1", 455, "JFK", 3, 13, 455 + 780 + 120 - 110),
        12: crew("N1/1", 455, "JFK", 3, 13, 455 + 780 + 120 - 60), 13: crew("N1/2", 955, "BOS", 1, 12, None),
        20: crew("N2/0", -15, "LAX", 1, 12, -15 + 720 + 120 - 330),
        21: crew("N2/1", 255, "JFK", 2, 10, None), 22: crew("N2/1", 255, "JFK", 2, 10, 255 + 600 + 120 - 240),
        23: crew("N2/2", 755, "ORD", 1, 14, 755 + 840 + 120 - 150)}
    wider = repairs.crews(legs, base, (*repairs.BASES, "LAX", "SJU"), ZONES)
    assert [wider[k].deadline for k in (20, 21, 22)] == [None, None, None]
    assert (wider[21].segments, wider[22].pairing, wider[23]) == (1, "N2/2", crew("N2/3", 755, "ORD", 1, 14, 1565))


def test_a_crewed_build_gives_every_flight_its_pairings_crew_and_changes_nothing_else(task, crewed):
    assert crewed.bases == repairs.BASES and not task.bases
    assert replace(crewed, bases=(), flights=tuple(replace(f, crew=None) for f in crewed.flights)) == task
    outstation = [f for f in crewed.flights if f.orig not in repairs.BASES]
    assert outstation and all(f.crew.deadline is None for f in crewed.flights if f.orig in repairs.BASES)
    for f in outstation:
        c = f.crew
        assert (c.deadline, c.rest) == (c.report + c.fdp + 120 - f.sched_block, c.report + c.fdp + 720 - f.sched_block)
    assert all(f.crew.report <= f.sched_dep - 45 for f in crewed.flights)


def leg(i: int, orig: str, dest: str, sd: int, dep: int | None, arr: int | None, tail: str = "N1",
        end: str | None = None) -> Flight:
    reached = end or (dest if dep is not None else None)
    actual = Outcome(dep is None, False, "B" if dep is None else "", dep, arr, reached, tail)
    return Flight(i, f"B6 {i}", orig, dest, sd, sd + 60, 60, 60, 100, 100, "A320", 0, 0, None, None, False, None,
                  sd + 900, actual)


def test_the_turns_jetblue_flew_are_its_consecutive_flown_legs_from_the_station_it_reached():
    legs = [leg(0, "BUF", "JFK", 100, 105, 170), leg(1, "JFK", "DCA", 200, 190, 250),
            leg(2, "DCA", "JFK", 300, None, None), leg(3, "DCA", "BOS", 320, 330, 400),
            leg(4, "BOS", "MCO", 430, 450, 600, end="JAX"), leg(5, "MCO", "JFK", 700, 720, 900)]
    assert repairs.connections(legs) == (Connection(0, 1, 20), Connection(1, 3, 80), Connection(3, 4, 50))
    assert repairs.connections([leg(0, "BUF", "JFK", 100, 105, 170), leg(1, "BOS", "JFK", 200, 230, 300)]) == ()


def test_tomorrows_need_is_the_last_scheduled_destination_today_cancelled_legs_included():
    cur = pd.DataFrame([("N1", "BUF", "JFK", 100, 0, 105, "JFK"), ("N1", "JFK", "DCA", 200, 0, 190, "DCA"),
                        ("N1", "DCA", "MCO", 300, 1, np.nan, np.nan), ("N2", "JFK", "BOS", 150, 0, 150, "BOS")],
                       columns=["Tail_Number", "Origin", "Dest", "sd", "Cancelled", "ad", "end"])
    nxt = pd.DataFrame([("N1", "JFK", 360), ("N2", "PWM", 400)], columns=["Tail_Number", "Origin", "sd"])
    empty = pd.DataFrame(columns=["Tail_Number", "Cancelled", "ad", "aa", "sa", "end", "CRSElapsedTime"])
    fleet = {t: Aircraft(t, "A32X", "A320-232", 162, "B-43 2024") for t in ("N1", "N2")}
    day = SimpleNamespace(fleet=fleet, prev=empty, cur=cur, nxt=nxt)
    sc = SimpleNamespace(tails=["N1", "N2"], frozen=empty)
    flights = [leg(0, "BUF", "JFK", 100, 105, 170), leg(1, "JFK", "DCA", 200, 190, 250),
               leg(2, "DCA", "MCO", 300, None, None), leg(3, "JFK", "BOS", 150, 150, 210, tail="N2")]
    tails, _ = repairs.tails(day, sc, flights, {"N1": "A320", "N2": "A320"}, {"A320": 40})
    assert [(t.end_req, t.actual_end) for t in tails] == [("MCO", "DCA"), ("BOS", "BOS")]
    assert tails[0].connections == (Connection(0, 1, 20),) and tails[1].connections == ()
