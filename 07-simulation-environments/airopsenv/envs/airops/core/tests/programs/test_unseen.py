import math
from dataclasses import replace

from program_helpers import EVERY, adv

from airops_core.programs import candidates, unseen
from airops_core.programs.advisories import Run, Stop
from airops_core.programs.model import Arrival

PROGRAM = adv("g", "gdp", 400, 600, 779, (10, 10, 10))
DEMAND = tuple(Arrival("AA", k, "ATL", 450 + 7 * k, 600 + 7 * k, None, False, 0, None, None) for k in range(26))


def covers(scope, hub, origin):
    return True


def test_the_in_sample_rate_reproduces_the_stated_average_delay():
    run = Run(PROGRAM, 400, 780)
    delays = [unseen.average_delay(run, DEMAND, covers, e, ()) for e in range(12)]
    assert delays == sorted(delays) and delays[0] < delays[-1]
    fit = unseen.in_sample(replace(PROGRAM, avg_delay=round(delays[4])), DEMAND, covers, ())
    assert (fit.e, fit.matched) == (4.0, True) and list(fit.curve[:12]) == delays
    miss = unseen.in_sample(replace(PROGRAM, avg_delay=10_000), DEMAND, covers, ())
    assert not miss.matched


def test_the_scale_sits_on_the_stated_delay_and_an_unmatched_advisory_takes_its_airports_median():
    curve = tuple(float(5 * e) for e in range(61))
    fits = [unseen.Fit(a, hub, e, 5 * e, stated, ok, curve) for a, hub, e, stated, ok in (
        ("a", "JFK", 20.0, 100, True), ("b", "JFK", 26.0, 130, True), ("c", "JFK", 60.0, 400, False),
        ("d", "BOS", 8.0, 40, True))]
    e, median = unseen.table(fits)
    assert median == {"BOS": 8.0, "JFK": 23.0} and e == {"a": 20.0, "b": 26.0, "c": 23.0, "d": 8.0}
    e, median = unseen.table(fits, {"JFK": 0.5, "BOS": 0.5})
    assert e == {"a": 10.0, "b": 13.0, "c": 40.0, "d": 4.0} and median == {"BOS": 4.0, "JFK": 13.0}


def test_fidelity_grows_with_the_scale_and_the_fit_picks_from_the_grid(task, fits):
    def at(s):
        return unseen.table(fits, {"JFK": s, "BOS": s})

    ratios = [unseen.fidelity(task, at(s)) for s in (0.0, 0.5, 1.0, 2.0)]
    assert ratios == sorted(ratios)
    assert unseen.fidelity(task, at(1.0)) == unseen.fidelity(task)
    modelled, observed, flights = unseen.program_delay(task)
    assert flights > 0 and observed > 0
    scale = unseen.fit_scale([task], fits)
    loss = {s: math.log(max(unseen.fidelity(task, at(s)), unseen.FLOOR)) ** 2 for s in unseen.GRID}
    assert scale == {"JFK": min(unseen.GRID, key=lambda s: (round(loss[s], 9), s))}


def test_a_date_storming_at_both_hubs_counts_in_each_hubs_fit(task, fits):
    both = replace(task, hubs=("JFK", "BOS"))
    assert unseen.program_delay(both, hub="JFK") == unseen.program_delay(task)
    assert unseen.program_delay(both, hub="BOS")[2] == 0
    scale = unseen.fit_scale([both], fits)
    assert set(scale) == {"JFK", "BOS"} and scale["JFK"] == unseen.fit_scale([task], fits)["JFK"]


def test_a_diverted_arrival_is_left_out_and_a_ground_stop_holds_the_modelled_departure(task):
    p = task.programs["JFK"]
    modelled, observed, flights = unseen.program_delay(task)
    i = next(i for i, a in enumerate(p.demand) if a.controlled and a.carrier != task.carrier and not a.cancelled
             and 0 <= a.sched_arr < 1440)
    demand = list(p.demand)
    demand[i] = replace(demand[i], diverted=True)
    gone = replace(task, programs={**task.programs, "JFK": replace(p, demand=tuple(demand))})
    assert unseen.program_delay(gone)[2] == flights - 1
    a = p.demand[i]
    stop = Stop("s", "JFK", a.sched_dep - 1, a.sched_arr + 600, EVERY)
    held = replace(task, programs={**task.programs, "JFK": replace(p, stops=(*p.stops, stop))})
    assert unseen.program_delay(held)[0] > modelled


def test_the_airports_own_counts_set_what_bts_misses():
    jfk = unseen.missed_share("JFK", 317.0, [], {})
    assert jfk == {"counted_per_day": 640.1, "bts_per_day": 317.0, "missed_share": 0.505}
    bos = unseen.missed_share("BOS", 400.0, ["2024-07"], {"2024-07": 31})
    assert bos["counted_per_day"] == round(19_130 / 31, 1)
    assert sorted(unseen.MASSPORT_ARRIVALS) == list(candidates.MONTHS)
