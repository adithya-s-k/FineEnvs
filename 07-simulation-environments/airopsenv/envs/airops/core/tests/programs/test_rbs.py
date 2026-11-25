import math
from dataclasses import replace

from program_helpers import adv

from airops_core.programs import advisories, rbs, unseen
from airops_core.programs.advisories import Run, Scope, Stop
from airops_core.programs.model import Arrival

SCOPE = Scope(False, frozenset({"ZTL", "ZDC"}), None, frozenset(), frozenset(), True)
ARTCC = {"ATL": "ZTL", "DCA": "ZDC", "SJU": "ZSU", "BOS": "ZBW"}
GDP = adv("g", "gdp", 400, 600, 719, (5, 4), scope=SCOPE)
RUN = Run(GDP, 400, 900)


def covers(scope: Scope, hub: str, origin: str) -> bool:
    return scope.covers(origin, ARTCC.get(origin), 200)


def arrival(carrier, number, orig, sd, sa, dep=None, cancelled=False, flight=None) -> Arrival:
    return Arrival(carrier, number, orig, sd, sa, dep, cancelled, 0, flight, None)


DAY = (
    arrival("AA", 1, "ATL", 390, 600, dep=395),
    arrival("B6", 2, "ATL", 500, 605, dep=560, flight=0),
    arrival("AA", 3, "DCA", 520, 605, dep=570),
    arrival("DL", 4, "SJU", 300, 640, dep=310),
    arrival("DL", 5, "ATL", 560, 602, cancelled=True),
    arrival("B6", 6, "DCA", 600, 700, cancelled=True, flight=1),
    arrival("B6", 7, "ATL", 370, 610, dep=380, flight=2),
)


def test_slots_follow_the_hourly_rates_inside_the_period():
    assert rbs.slots(600, 719, (5, 4)) == [600, 612, 624, 636, 648, 660, 675, 690, 705]
    assert rbs.slots(630, 719, (4, 4)) == [630, 645, 660, 675, 690, 705]


def test_the_unseen_stream_is_even_and_may_be_fractional():
    assert rbs.unseen(600, 719, 2) == [600, 630, 660, 690]
    assert rbs.unseen(600, 719, 1.5) == [600, 640, 680]
    assert rbs.unseen(600, 719, 0) == []


def test_a_ration_by_schedule_day_with_exemptions_compression_and_the_departure_lock():
    seg, got = rbs.segment(RUN, DAY, covers, 2.0, ())
    assert got == {0: 600, 6: 624, 3: 648, 2: 675, 1: 705, 5: 720}
    assert (seg.slots, seg.holders) == ((705,), (0,))
    forecast, early = rbs.segment(RUN, DAY, covers, 2.0, (), known=420)
    assert early[4] == 675 and early[1] == 720 and forecast.slots == ()


def test_a_pushback_inside_a_ground_stop_is_not_airborne_when_the_program_is_signed():
    held, got = rbs.segment(RUN, DAY, covers, 2.0, (Stop("s", "JFK", 370, 450, SCOPE),))
    assert (held.slots, held.holders) == ((675, 705), (0, 2)) and got[0] == 612 and got[3] == 648


def test_controlled_reads_the_schedule_and_the_departure_never_lb():
    runs = (Run(GDP, 550, 700), Run(replace(GDP, id="r", issued=700, start=700, end=899), 700, 900))
    assert rbs.governing(runs, covers, "ATL", 650, 560).adv.id == "g"
    assert rbs.governing(runs, covers, "ATL", 750, 560) is None
    assert rbs.governing(runs, covers, "ATL", 750, 720).adv.id == "r"
    assert rbs.governing(runs, covers, "SJU", 650, 560) is None
    assert rbs.governing(runs, covers, "ATL", 650, 540) is None


def test_gs_held_reads_the_scheduled_departure_inside_the_window():
    stops = (Stop("s", "JFK", 600, 660, SCOPE),)
    assert rbs.gs_held(stops, covers, "ATL", 630) == "s"
    assert rbs.gs_held(stops, covers, "ATL", 660) is None
    assert rbs.gs_held(stops, covers, "BOS", 630) is None


def test_every_task_flight_is_controlled_by_the_test_on_its_recorded_departure(task):
    for f in task.flights:
        runs = advisories.as_ran(task.programs[f.dest].advisories)[0] if f.dest in task.programs else ()
        departs = f.actual.dep if f.actual.dep is not None else f.sched_dep
        run = rbs.governing(runs, task.covers, f.orig, f.sched_arr, departs)
        assert f.controlled == (run.adv.id if run else None)
    assert any(f.controlled for f in task.flights)


def test_the_nominal_cap_leaves_what_others_and_the_unseen_stream_do_not_take(task):
    caps = rbs.nominal(task, "JFK")
    p = task.programs["JFK"]
    h = 19
    mine = sum(1 for f in task.flights if f.dest == "JFK" and f.sched_arr // 60 == h)
    others = sum(1 for a in p.demand if a.flight is None and a.sched_arr // 60 == h)
    assert caps[h] == max(mine, math.floor(p.aar - others - p.e_bar)) and min(caps) == task.start // 60


def test_more_unseen_traffic_never_gives_jetblue_more_slots(task, fits):
    def at(k):
        scale = {"JFK": k, "BOS": k}
        return rbs.rescale(task, scale, *unseen.table(fits, scale))

    pools = [[len(s.slots) for s in at(k).programs["JFK"].segments] for k in (0.0, 1.0, 2.0)]
    assert all(a >= b for a, b in zip(pools[0], pools[1], strict=True))
    assert all(a >= b for a, b in zip(pools[1], pools[2], strict=True))
    assert at(1.0) == task


def test_the_forecast_sees_only_what_was_signed(task):
    gdp = next(a for a in task.programs["JFK"].advisories if a.kind == "gdp")
    assert rbs.forecast(task, "JFK", gdp.issued - 1)[0][0].id != gdp.id
    seg = rbs.forecast(task, "JFK", gdp.issued)[0][-1]
    assert (seg.id, seg.until) == (gdp.id, gdp.end + 1)
