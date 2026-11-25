import json
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
from day_helpers import TURN, day, flight, tail
from program_helpers import DATE, WINDOW

from airops_core.build import pack
from airops_core.build.fleet import Aircraft
from airops_core.programs import candidates, mip, network, references, sim, unseen
from airops_core.programs.model import Busy, Move, Outcome, Tail, Task

START = 360
RECORD = ("Tail_Number", "Origin", "Dest", "sd", "sa", "ad", "aa", "CRSElapsedTime", "Cancelled", "end")


def record(*rows) -> pd.DataFrame:
    """BTS rows (tail, origin, destination, scheduled departure and arrival, actual departure and arrival, cancelled,
    the station the leg ended at) with local minutes."""
    return pd.DataFrame([dict(zip(RECORD, (t, o, d, sd, sa, ad, aa, sa - sd, c, end), strict=True))
                         for t, o, d, sd, sa, ad, aa, c, end in rows], columns=list(RECORD))


def flown(i: int, tail: str, orig: str, dest: str, sched: int, dep: int, block: int = 60, end: str | None = None):
    """A task flight `tail` flew, leaving at `dep` and ending at `end` (its destination unless it diverted)."""
    return flight(i, orig, dest, sched, block, actual=Outcome(False, end is not None and end != dest, "", dep,
                                                               dep + block, end or dest, tail))


def cancelled(i: int, tail: str, orig: str, dest: str, sched: int):
    """A task flight JetBlue cancelled, BTS leaving it on `tail`."""
    return flight(i, orig, dest, sched, 60, actual=Outcome(True, False, "A", None, None, None, tail))


def built(flights, today=(), before=(), tails=None):
    """network_tails over the flights, with `today` the record's legs of the date and `before` the day before's."""
    cur = record(*today)
    names = sorted(tails or {f.actual.tail for f in flights} | set(cur.Tail_Number))
    fleet = {t: Aircraft(t, "A32X", "A320-232", 150, "B-43 2024") for t in names}
    day = pack.Day(DATE, "JFK", "America/New_York", 0, record(*before), cur, record(), record(), fleet, {}, {}, TURN)
    sc = pack.Scope((START + 60, START + 360), START, names, 0, True, cur, cur[(cur.Cancelled == 0) & (cur.ad < START)],
                    {})
    tails, counts = network.network_tails(day, sc, flights, dict.fromkeys(names, "A320"), {"A320": TURN})
    return {t.tail: t for t in tails}, counts


def test_a_tail_flies_the_legs_it_flew_in_that_order_and_a_swap_or_a_stale_tail_makes_no_move():
    flights = [flown(0, "N2", "JFK", "BOS", 400, 405), cancelled(1, "N1", "JFK", "DCA", 450),
               flown(2, "N2", "BOS", "JFK", 500, 520), flown(3, "N1", "JFK", "MCO", 550, 560),
               flown(4, "N3", "JFK", "BUF", 600, 720), flown(5, "N3", "DCA", "JFK", 610, 615)]
    tails, counts = built(flights, today=[("N1", "BUF", "JFK", 300, 360, 300, 350, 0, "JFK"),
                                          ("N2", "MCO", "JFK", 200, 350, 200, 340, 0, "JFK"),
                                          ("N3", "BOS", "DCA", 200, 260, 205, 265, 0, "DCA")])
    assert {t: x.legs for t, x in tails.items()} == {"N1": (3,), "N2": (0, 2), "N3": (5, 4)}
    assert not any(x.repositions for x in tails.values()) and "moves" not in counts
    assert all(f.id not in x.legs for x in tails.values() for f in flights if f.actual.dep is None)


def test_a_station_gap_on_the_flown_path_is_a_move_and_a_long_daytime_ground_gap_no_cancelled_leg_fills_is_busy():
    flights = [flown(0, "N1", "BUF", "JFK", 400, 400), flown(1, "N1", "BOS", "DCA", 600, 600),
               flown(2, "N1", "DCA", "JFK", 1000, 1000), flown(3, "N2", "JFK", "PWM", 420, 420, end="BOS"),
               flown(4, "N2", "BOS", "JFK", 700, 700), flown(5, "N3", "JFK", "BOS", 400, 400),
               cancelled(6, "N3", "BOS", "DCA", 600), flown(7, "N3", "BOS", "JFK", 1000, 1000),
               flown(8, "N4", "JFK", "BOS", 500, 800), flown(9, "N4", "BOS", "JFK", 900, 920)]
    tails, counts = built(flights)
    away = Busy(2, 1000 + 240 + 1)
    assert tails["N1"].repositions == (Move(0, "JFK", "BOS", 460, 600 - TURN),)
    assert tails["N1"].busy == (Busy(1, 1000 - TURN), away)
    assert tails["N2"].repositions == (Move(3, "PWM", "BOS", 480, 700 - TURN),)
    assert tails["N3"].legs == (5, 7) and not tails["N3"].repositions
    assert tails["N4"].legs == (8, 9)
    assert [b for t in ("N2", "N3", "N4") for b in tails[t].busy] == [replace(away, leg=i) for i in (4, 7, 9)]
    assert (counts["moves"], counts["moves after a diversion"], counts["busy"], counts["gateway busy"]) == (1, 1, 1, 4)


def test_each_tail_starts_where_the_record_has_it_at_the_start():
    flights = [flown(0, "N1", "JFK", "BOS", 400, 400), flown(1, "N3", "MCO", "JFK", 480, 495),
               flown(2, "N4", "PWM", "JFK", 500, 505), flown(3, "N6", "BOS", "JFK", 700, 700),
               flown(4, "N2", "BOS", "JFK", 450, 450), cancelled(5, "N5", "BUF", "JFK", 600),
               flown(6, "N8", "BOS", "JFK", 700, 700), cancelled(7, "N8", "BOS", "PWM", 500),
               flown(8, "N9", "SJU", "JFK", 420, 700), flown(9, "N10", "MCO", "JFK", 500, 520),
               flown(10, "N11", "BOS", "JFK", 650, 660)]
    today = [("N1", "BUF", "JFK", 250, 320, 260, 330, 0, "JFK"), ("N3", "JFK", "DCA", 250, 320, 255, 340, 0, "DCA"),
             ("N5", "BUF", "JFK", 600, 660, None, None, 1, None), ("N6", "JFK", "BOS", 300, 400, 300, 400, 0, "BOS"),
             ("N7", "JFK", "PWM", 200, 300, 210, 310, 0, "PWM"), ("N8", "JFK", "BOS", 300, 400, 300, 400, 0, "BOS"),
             ("N10", "JFK", "BOS", 200, 260, 200, 260, 0, "BOS"), ("N11", "JFK", "BOS", 300, 400, 350, 450, 0, "BOS")]
    before = [("N2", "JFK", "BOS", -200, -100, -190, -90, 0, "BOS"),
              ("N2", "BOS", "JFK", -500, -400, -490, -390, 0, "JFK"),
              ("N2", "MCO", "PWM", -80, 100, 900, 1080, 0, "PWM"),
              ("N9", "JFK", "BOS", -300, -200, -290, -190, 0, "BOS"),
              ("N9", "BOS", "SJU", -60, 180, 400, 640, 0, "SJU"),
              ("N10", "BOS", "MCO", -30, 120, 300, 450, 0, "MCO")]
    tails, counts = built(flights, today, before)
    starts = {t: (x.pos, x.ready) for t, x in tails.items()}
    assert starts == {"N1": ("JFK", 330 + TURN), "N2": ("BOS", 0), "N3": ("MCO", 480 - TURN), "N4": ("PWM", 500 - TURN),
                      "N6": ("BOS", 700 - TURN), "N7": ("PWM", 310 + TURN), "N8": ("BOS", 400 + TURN),
                      "N9": ("SJU", 640 + TURN), "N10": ("MCO", 450 + TURN), "N11": ("BOS", 450 + TURN)}
    assert counts == {"busy": 0, "gateway busy": 9, "out of service": 1, "start: day before": 3,
                      "start: first flown leg": 1, "start: today": 3, "start: today, busy": 1,
                      "start: today, moved to the first flown leg": 1, "start: today, no flown leg": 1}
    late = [flown(0, "N3", "MCO", "JFK", 360, 365)]
    assert built(late, today[1:2])[0]["N3"].ready == 340 + TURN


def test_a_tail_whose_last_flown_leg_lands_at_a_gateway_by_day_is_busy_past_every_deadline_and_flies_no_more():
    flights = [flown(0, "N1", "DCA", "JFK", 600, 600), flown(1, "N2", "DCA", "JFK", 600, 600),
               cancelled(2, "N2", "JFK", "BUF", 800), flown(3, "N3", "DCA", "JFK", 1020, 1020),
               flown(4, "N4", "JFK", "DCA", 600, 600), cancelled(5, "N3", "JFK", "DCA", 900),
               flown(6, "N6", "DCA", "SJU", 960, 960)]
    tails, counts = built(flights, today=[("N5", "DCA", "BOS", 200, 260, 200, 260, 0, "BOS")])
    end = 1020 + 240 + 1
    assert tails["N1"].busy == (Busy(0, end),) and (tails["N5"].pos, tails["N5"].ready) == ("BOS", end)
    assert not any(tails[t].busy for t in ("N2", "N3", "N4", "N6")) and counts["gateway busy"] == 2
    task = day(flights[:6], [x for t, x in sorted(tails.items()) if t != "N6"])
    run = sim.execute(task, {})
    assert run.legs["N1"] == (0,) and run.flights[5].dep == 1080 + TURN
    sol = mip.solve(task)
    assert sol.deps[5] in (None, 1080 + TURN) and sim.execute(task, sol.plan(0)).cost.total == sol.objective
    assert next(t for t in network.scheduled(task).tails if t.tail == "N5") == tails["N5"]


def test_a_tail_that_flew_nothing_that_day_nor_the_day_before_is_out_of_service_and_others_fly_its_legs():
    flights = [flown(0, "N1", "BUF", "DCA", 380, 380), cancelled(1, "N9", "DCA", "BUF", 500),
               flown(2, "N1", "DCA", "PWM", 700, 700), cancelled(3, "N2", "PWM", "BOS", 900)]
    tails, counts = built(flights, before=[("N2", "BUF", "PWM", -300, -240, -300, -240, 0, "PWM")])
    assert set(tails) == {"N1", "N2"} and tails["N2"].pos == "PWM" and counts["out of service"] == 1
    run = sim.execute(day(flights, list(tails.values())), {})
    assert (run.flights[1].status, run.flights[1].tail) == ("flown", "N1") and run.flights[3].tail == "N2"


def test_the_calm_day_starts_where_the_schedule_needs_its_aircraft_and_flies_it_all_on_time():
    flights = [flight(0, "JFK", "BOS", 100, 60), flight(1, "BOS", "JFK", 200, 60), flight(2, "BOS", "DCA", 150, 60),
               flight(3, "DCA", "JFK", 300, 60), flight(4, "JFK", "MCO", 400, 150)]
    tails = [tail("N1", "JFK"), tail("N2", "JFK", 50), tail("N3", "JFK", 500)]
    task = replace(day(flights, tails), needs=network.needs(tails, flights)[0])
    assert network.assumed(task) == {("A320", "JFK"): 1, ("A320", "BOS"): 1}
    calm = network.scheduled(task)
    assert {t.tail: (t.pos, t.ready) for t in calm.tails} == {"N1": ("JFK", 0), "N2": ("JFK", 0), "N3": ("BOS", 0)}
    assert calm.needs == network.needs(calm.tails, flights)[0] == {"A320": {"JFK": 2, "MCO": 1}}
    run = sim.execute(calm, {}, sim.calm(calm))
    assert all(r.dep == task.flight(i).sched_dep for i, r in run.flights.items())
    c = references.calm(task)
    assert (c.cancelled, c.ferries) == (0, 0) and c.task == calm
    assert references.calm(task, record=True).cancelled >= 1
    elsewhere = network.scheduled(replace(task, tails=tuple(replace(t, pos="DCA", ready=300) for t in tails)))
    assert Counter(t.pos for t in elsewhere.tails) == Counter(t.pos for t in calm.tails)
    assert elsewhere.needs == calm.needs
    assert network.shortfall(task) == {"aircraft": 1, "departures": 1, "fleet": 0, "short": {"A320 BOS": 1}}
    one = replace(task, tails=tuple(tails[:1]))
    assert [(t.pos, t.ready) for t in network.scheduled(one).tails] == [("BOS", 0)]
    assert network.shortfall(one)["fleet"] == 1


def test_tomorrows_needs_fly_the_whole_schedule_and_its_moves_from_the_start_and_a_short_station_counts_zero():
    flights = [flown(0, "N1", "JFK", "BOS", 400, 400), cancelled(1, "N2", "MCO", "JFK", 450),
               flown(2, "N3", "BOS", "DCA", 500, 500), flown(3, "N3", "PWM", "JFK", 700, 700),
               replace(flown(4, "N4", "JFK", "BUF", 400, 400), family="E190")]
    tails = [Tail(n, fam, "", 150, pos, 0, legs, (), moves, pos, pos) for n, fam, pos, legs, moves in (
        ("N1", "A320", "JFK", (0,), ()), ("N2", "A320", "JFK", (), ()),
        ("N3", "A320", "BOS", (2, 3), (Move(2, "DCA", "PWM", 560, 670),)), ("N4", "E190", "JFK", (4,), ()))]
    need, short = network.needs(tails, flights)
    assert need == {"A320": {"BOS": 1, "JFK": 2}, "E190": {"BUF": 1}} and short == {"A320 MCO": 1}
    assert sum(need["A320"].values()) == 3


def test_a_storm_date_holds_every_jetblue_leg_from_the_start_and_every_typed_tail_in_service(slice_inputs, fits,
                                                                                            tmp_path_factory):
    derived = tmp_path_factory.mktemp("derived")
    ones = {"JFK": 1.0, "BOS": 1.0}
    e_fit, e_median = unseen.table(fits, ones)
    b = network.build(DATE, ("JFK",), slice_inputs, derived, e_fit, e_median, ones, windows={"JFK": WINDOW},
                      split="control")
    task = b.task
    assert (task.task_id, task.start, task.hub, task.hubs, task.version) == ("b6-20230626-0400", 240, "JFK",
                                                                             ("JFK",), 3)
    day = candidates.context(slice_inputs, derived, DATE, "JFK").day
    cur, prev = day.cur, day.prev
    decided = cur[((cur.Cancelled == 0) & (cur.ad >= 240)) | ((cur.Cancelled == 1) & (cur.sd >= 240))]
    idle = set(cur.Tail_Number) - set(cur[cur.Cancelled == 0].Tail_Number) - set(prev[prev.Cancelled == 0].Tail_Number)
    assert len(task.flights) == len(decided) == len(cur) == 39
    assert len(idle) == task.clauses["network"]["out of service"] == 1
    assert {t.tail for t in task.tails} == set(cur.Tail_Number) - idle
    for t in task.tails:
        mine = sorted((f for f in task.flights if f.actual.tail == t.tail and f.actual.dep is not None),
                      key=lambda f: (f.actual.dep, f.id))
        assert t.legs == tuple(f.id for f in mine)
    assert not any(t.repositions for t in task.tails)
    families = Counter(t.family for t in task.tails)
    assert {fam: sum(by.values()) for fam, by in task.needs.items()} == families
    assert task.clauses["short"] == {"E190 ACK": 1, "E190 RDU": 1, "E190 ROC": 1}
    assert task.needed == Counter({(fam, s): n for fam, by in task.needs.items() for s, n in by.items()})
    again = Task.from_dict(json.loads(task.to_json()))
    assert again == task and again.to_json() == task.to_json()
    replay = references.airline(task)
    assert replay.auto == 0 and replay.flew == sum(f.actual.dep is not None for f in task.flights)


def test_a_storm_date_starts_at_four_its_windows_from_its_first_program_hour_and_needs_five_flights_held(
        slice_inputs, fits, tmp_path_factory, monkeypatch):
    derived = tmp_path_factory.mktemp("derived")
    ones = {"JFK": 1.0, "BOS": 1.0}
    e_fit, e_median = unseen.table(fits, ones)
    b = network.build(DATE, ("JFK",), slice_inputs, derived, e_fit, e_median, ones)
    assert b.task.start == network.START == 240 and b.task.task_id == "b6-20230626-0400"
    assert b.row["fallback"] and b.task.window == (60 * b.row["first_hours"]["JFK"], 900)
    assert b.task.clauses["controlled_or_held"] == 10 >= network.MIN_CONTROLLED == 5
    monkeypatch.setattr(network, "MIN_CONTROLLED", 11)
    b = network.build(DATE, ("JFK",), slice_inputs, derived, e_fit, e_median, ones)
    assert b.task is None and b.row["reason"] == "fewer than 11 JetBlue flights controlled or held"
    control = network.build(DATE, ("JFK",), slice_inputs, derived, e_fit, e_median, ones, windows={"JFK": WINDOW},
                            split="control")
    assert control.task.start == network.START and control.task.clauses["controlled_or_held"] == 10


def test_the_aircraft_start_at_four_where_the_record_has_them(slice_inputs, fits, tmp_path_factory):
    derived = tmp_path_factory.mktemp("derived")
    ones = {"JFK": 1.0, "BOS": 1.0}
    e_fit, e_median = unseen.table(fits, ones)
    task = network.build(DATE, ("JFK",), slice_inputs, derived, e_fit, e_median, ones, windows={"JFK": WINDOW},
                         split="control").task
    day = candidates.context(slice_inputs, derived, DATE, "JFK").day
    prev = day.prev[day.prev.Cancelled == 0]
    airborne = prev[(prev.ad < task.start) & (prev.aa > task.start)]
    assert sorted(airborne.Tail_Number) == ["N729JB", "N763JB", "N976JT"]
    at = {t.tail: (t.pos, t.ready) for t in task.tails}
    for r in airborne.itertuples():
        t = next(t for t in task.tails if t.tail == r.Tail_Number)
        assert at[t.tail] == (r.Dest, r.aa + day.turn[t.family]) and t.ready > task.start
    assert at["N972JT"] == ("JFK", 115 + day.turn["A321"]) and at["N249JB"] == ("RIC", 0)
    assert not len(day.cur[(day.cur.Cancelled == 0) & (day.cur.ad < task.start)])


def dates(first: str, n: int, gap: int) -> list[str]:
    start = date.fromisoformat(first)
    return [(start + timedelta(days=gap * k)).isoformat() for k in range(n)]


def test_the_eval_draw_takes_twelve_dates_per_storm_hub_one_per_event_and_draws_a_date_for_one_hub_at_most():
    both = dates("2023-06-01", 15, 6)
    hubs = {**{d: ("JFK", "BOS") for d in both}, **{d: ("JFK",) for d in dates("2024-01-01", 6, 4)},
            **{d: ("BOS",) for d in dates("2024-12-01", 3, 4) + ["2024-12-02"]}}
    event = candidates.events(hubs)
    drawn = network.draw(hubs, event, seed=7)
    by = Counter((h, candidates.season(d)) for d, h in drawn.items())
    assert by == {("JFK", "summer"): 8, ("JFK", "winter"): 4, ("BOS", "summer"): 7, ("BOS", "winter"): 3}
    assert all(n == 1 for n in Counter((h, event[d]) for d, h in drawn.items()).values())
    assert drawn == network.draw(hubs, event, seed=7) and all(h in hubs[d] for d, h in drawn.items())
    split = network.split(hubs, event, seed=7)
    taken = {event[d] for d in drawn}
    assert {d for d, v in split.items() if v == "eval"} == set(drawn)
    assert all(event[d] not in taken for d, v in split.items() if v == "train")
    assert network.settle(split, [d for d in split if d != min(drawn)], event) == {
        d: v for d, v in split.items() if d != min(drawn)}
