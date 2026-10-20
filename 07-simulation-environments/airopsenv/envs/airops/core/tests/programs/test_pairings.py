"""The trip builder by hand and on random schedules: every leg covered by a duty legal as scheduled (Table B for its
report's hour and segments, Table A, 14 h for a cabin duty with its deadheads), connections of CONNECT at least,
outstation starts fed by a layover or a crew deadheaded there, the column generation run until nothing prices out, and
the date's duties only; the planned rotations from scheduled times and the turn rule alone."""

import random

import pytest

from airops_core.programs import crews, pairings
from airops_core.programs.pairings import Leg


def leg(key: int, orig: str, dest: str, sd: int, block: int, family: str = "A320", group: str = "A32X",
        after: int | None = None) -> Leg:
    return Leg(key, f"B6 {key}", group, orig, dest, sd, sd + block, block, family, (sd - crews.REPORT) // 60 % 24,
               after)


def legal(d: pairings.Duty, legs: dict[int, Leg]) -> bool:
    flown = [legs[k] for k in d.legs]
    if any(b.sd < a.sa + crews.CONNECT or a.dest != b.orig for a, b in zip(flown, flown[1:], strict=False)):
        return False
    if d.kind == "pilot":
        return flown[-1].sa - d.report <= crews.fdp(d.hour, len(flown)) and \
            sum(g.block for g in flown) <= crews.flight_time(d.hour)
    return d.release - d.report <= crews.CABIN_DUTY


def test_a_base_turn_is_one_duty():
    legs = [leg(1, "JFK", "BOS", 600, 60), leg(2, "BOS", "JFK", 700, 60), leg(3, "JFK", "DCA", 800, 80),
            leg(4, "DCA", "JFK", 920, 80)]
    for kind in crews.KINDS:
        b = pairings.build(legs, kind, "A32X", ("JFK",), time_limit=30)
        assert b.uncovered == [] and len(b.duties) == 1
        d = b.duties[0]
        assert d.legs == (1, 2, 3, 4) and (d.start, d.end, d.report, d.release) == ("JFK", "JFK", 540, 1015)


def test_crews_connect_in_connect_minutes_even_on_their_aircraft():
    legs = [leg(1, "JFK", "BOS", 600, 60, after=2), leg(2, "BOS", "JFK", 670, 60)]
    assert pairings.build(legs, "pilot", "A32X", ("JFK",), time_limit=30).uncovered == [2]
    legs = [leg(1, "JFK", "BOS", 600, 60, after=2), leg(2, "BOS", "JFK", 680, 60)]
    assert [d.legs for d in pairings.build(legs, "pilot", "A32X", ("JFK",), time_limit=30).duties] == [(1, 2)]


def test_table_b_splits_a_long_day():
    legs = [leg(k, "JFK" if k % 2 else "BOS", "BOS" if k % 2 else "JFK", 300 + 100 * k, 60) for k in range(1, 12)]
    b = pairings.build(legs, "pilot", "A32X", ("JFK", "BOS"), time_limit=30)
    by = {g.key: g for g in legs}
    assert b.uncovered == [] and len(b.duties) >= 2 and all(legal(d, by) for d in b.duties)


def test_an_outstation_start_follows_a_layover_from_the_day_before():
    legs = [leg(1, "JFK", "PWM", -700, 70), leg(2, "PWM", "JFK", 360, 70)]
    b = pairings.build(legs, "pilot", "A32X", ("JFK",), time_limit=30)
    assert [d.legs for d in b.duties] == [(2,)] and b.duties[0].start == "PWM"


def test_a_cabin_duty_ends_away_rather_than_deadhead_past_fourteen_hours():
    legs = [leg(1, "JFK", "MCO", 300, 180, group="A220"), leg(2, "MCO", "BUF", 520, 150, group="A220"),
            leg(3, "BUF", "MCO", 700, 150, group="A220"), leg(4, "MCO", "BUF", 880, 150, group="A220"),
            leg(5, "BUF", "JFK", 1060, 90)]
    b = pairings.build(legs, "cabin", "A220", ("JFK",), time_limit=30)
    assert all(d.release - d.report <= crews.CABIN_DUTY for d in b.duties)
    assert all(5 not in d.deadheads for d in b.duties if 4 in d.legs and d.report < 300)


@pytest.mark.parametrize("seed", range(6))
def test_random_schedules_are_covered_by_legal_duties(seed):
    rng = random.Random(seed)
    stations = ("JFK", "BOS", "DCA", "MCO", "BUF", "PWM")
    legs = []
    for _ in range(6):
        at, t = rng.choice(stations[:2]), rng.randrange(-900, 600, 5)
        for _ in range(rng.randint(2, 6)):
            dest = rng.choice([s for s in stations if s != at])
            block = rng.randrange(50, 200, 5)
            legs.append(leg(len(legs) + 1, at, dest, t, block))
            at, t = dest, t + block + rng.randrange(40, 120, 5)
    for kind in crews.KINDS:
        b = pairings.build(legs, kind, "A32X", ("JFK", "BOS"), time_limit=30)
        by = {g.key: g for g in legs}
        assert b.converged and b.rounds < 50
        assert all(legal(d, by) and max(by[k].sd for k in d.legs) >= 0 for d in b.duties)
        flown = [k for d in b.duties for k in d.legs]
        today = {g.key for g in legs if g.sd >= 0}
        assert len(flown) == len(set(flown)) and today <= set(flown) | set(b.uncovered)
        assert not set(flown) & set(b.uncovered)


def test_a_crew_deadheads_to_an_outstation_to_lay_over_for_a_dawn_leg_no_inbound_reaches_rested():
    legs = [leg(1, "BOS", "JFK", 660 - 1440, 80, "E190", "E190"), leg(2, "JFK", "SYR", 790 - 1440, 70),
            leg(3, "JFK", "SYR", 1240 - 1440, 80, "E190", "E190"), leg(4, "SYR", "JFK", 345, 75, "E190", "E190")]
    for kind in crews.KINDS:
        b = pairings.build(legs, kind, "E190", ("JFK", "BOS"), time_limit=30)
        assert b.uncovered == [] and b.converged
        assert [(d.legs, d.start) for d in b.duties] == [((4,), "SYR")]
    alone = [g for g in legs if g.key != 2]
    assert pairings.build(alone, "pilot", "E190", ("JFK", "BOS"), time_limit=30).uncovered == [4]


def test_a_deadhead_priced_off_its_rounded_cost_does_not_stall_the_column_generation():
    legs = [leg(1, "JFK", "BUF", 400, 77), leg(2, "BUF", "DCA", 520, 83), leg(3, "DCA", "JFK", 700, 71),
            leg(4, "JFK", "BUF", 1000, 77), leg(5, "BUF", "JFK", 1150, 77)]
    for kind in crews.KINDS:
        b = pairings.build(legs, kind, "A32X", ("JFK",), time_limit=30)
        assert b.converged and b.rounds < 10 and b.uncovered == [] and b.gap_pct < 1e-6


def test_planned_rotations_turn_each_arrival_into_the_family_s_departure_the_turn_allows_evenly():
    legs = [leg(1, "JFK", "BOS", 540, 60), leg(2, "DCA", "BOS", 540, 70), leg(3, "BOS", "JFK", 700, 60),
            leg(4, "BOS", "DCA", 800, 60), leg(6, "JFK", "BOS", 545, 60, "A321"), leg(7, "BOS", "JFK", 720, 60, "A321"),
            leg(8, "BUF", "BOS", 100, 60)]
    after = pairings.rotations(legs, {"A320": 45, "A321": 60})
    assert after == {1: 3, 2: 4, 6: 7} and 100 ** 2 + 190 ** 2 < 200 ** 2 + 90 ** 2
    assert pairings.rotations(legs, {"A320": 45, "A321": 120}) == {1: 3, 2: 4}
    turned = [leg(g.key, g.orig, g.dest, g.sd, g.block, g.family, g.group, after.get(g.key)) for g in legs]
    assert pairings.overnights(turned, "A32X", ("JFK",), 700) == {"DCA": 1, "BUF": 1}
