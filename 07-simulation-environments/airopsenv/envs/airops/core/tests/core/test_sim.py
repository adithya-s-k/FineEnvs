import itertools
import random

from airops_core.core.cost import ferry_cost
from airops_core.core.sim import AUTO_CANCELLED, CANCELLED, FLOWN, _assign, as_plan, execute
from airops_core.refs import policies


def test_an_empty_plan_flies_everything_as_early_as_it_can(make_task):
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 700, "lb": 20},
                      {"orig": "BOS", "dest": "JFK", "dep": 800}], [{"tail": "N1", "pos": "JFK"}])
    run = execute(task, {})
    a, b = run.flights[0], run.flights[1]
    assert (a.status, a.tail, a.dep, a.arr, a.arr_delay) == (FLOWN, "N1", 720, 780, 20)
    assert (b.dep, b.arr, b.arr_delay) == (810, 870, 10)
    assert run.legs == {"N1": (0, 1)} and run.ends == {"N1": "JFK"} and not run.ferries
    assert run.cost.total == 100 * 20 + 100 * 10


def test_nothing_leaves_before_the_first_watch(make_task):
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 500}], [{"tail": "N1", "pos": "JFK"}], start=600)
    assert execute(task, {}).flights[0].dep == 600


def test_holds_and_cancellations_are_obeyed(make_task):
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 700}, {"orig": "BOS", "dest": "JFK", "dep": 800}],
                     [{"tail": "N1", "pos": "JFK"}])
    run = execute(task, {0: {"cancel": False, "not_before": 745}, "1": {"cancel": True, "not_before": None}})
    assert run.flights[0].dep == 745 and run.flights[1].status == CANCELLED
    assert run.cost.cancel == 523 * 100 and run.cost.pax_delay == 4500
    assert run.cost.ferry == ferry_cost(task, "BOS", "JFK") and run.ferries[0].orig == "BOS"


def test_a_flight_waits_for_the_first_ready_aircraft_and_ties_go_by_tail(make_task):
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 700}, {"orig": "JFK", "dest": "DCA", "dep": 700}],
                     [{"tail": "N2", "pos": "JFK", "ready": 650}, {"tail": "N1", "pos": "JFK", "ready": 650},
                      {"tail": "N3", "pos": "JFK", "ready": 900}])
    run = execute(task, {})
    assert (run.flights[0].tail, run.flights[1].tail) == ("N1", "N2")
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 700}], [{"tail": "N1", "pos": "JFK", "ready": 760}])
    assert execute(task, {}).flights[0].dep == 760


def test_an_aircraft_flown_in_later_wakes_a_waiting_flight(make_task):
    task = make_task([{"orig": "BOS", "dest": "JFK", "dep": 700},
                      {"orig": "DCA", "dest": "BOS", "dep": 705, "block": 60}],
                     [{"tail": "N1", "pos": "BOS", "ready": 1000}, {"tail": "N2", "pos": "DCA", "end_req": "BOS"}],
                     turn=30)
    run = execute(task, {})
    assert run.flights[1].dep == 705
    assert (run.flights[0].tail, run.flights[0].dep) == ("N2", 705 + 60 + 30)


def test_a_flight_with_no_aircraft_ever_is_auto_cancelled(make_task):
    task = make_task([{"orig": "BOS", "dest": "JFK", "dep": 700}], [{"tail": "N1", "pos": "JFK"}])
    run = execute(task, {})
    assert run.flights[0].status == AUTO_CANCELLED and run.cost.cancel == 523 * 100
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 700, "dmax": 120}],
                     [{"tail": "N1", "pos": "JFK", "ready": 821}])
    assert execute(task, {}).flights[0].status == AUTO_CANCELLED
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 700, "dmax": 120}],
                     [{"tail": "N1", "pos": "JFK", "ready": 820}])
    assert execute(task, {}).flights[0].dep == 820
    assert execute(task, {0: {"cancel": False, "not_before": 821}}).flights[0].status == AUTO_CANCELLED


def test_full_hours_push_a_flight_to_the_next_hour_with_room_at_both_ends(make_task):
    flights = [{"orig": "JFK", "dest": "BOS", "dep": 700, "block": 50} for _ in range(3)]
    tails = [{"tail": f"N{i}", "pos": "JFK"} for i in range(3)]
    run = execute(make_task(flights, tails, caps={"JFK:dep": {"11": 2, "12": 0}}), {})
    assert [run.flights[i].dep for i in range(3)] == [700, 700, 780]
    run = execute(make_task(flights, tails, caps={"BOS:arr": {"12": 1, "13": 1}}), {})
    assert [run.flights[i].dep for i in range(3)] == [700, 730, 790]
    assert [run.flights[i].arr for i in range(3)] == [750, 780, 840]
    run = execute(make_task(flights, tails, caps={"JFK:dep": {"11": 1}, "BOS:arr": {"12": 1}}), {})
    assert [run.flights[i].dep for i in range(3)] == [700, 730, 730]


def test_hours_not_listed_are_unconstrained(make_task):
    flights = [{"orig": "JFK", "dest": "BOS", "dep": 700} for _ in range(4)]
    tails = [{"tail": f"N{i}", "pos": "JFK"} for i in range(4)]
    run = execute(make_task(flights, tails, caps={"JFK:dep": {"12": 0}}), {})
    assert {r.dep for r in run.flights.values()} == {700}


def test_end_of_day_ferries_match_each_family_to_its_required_stations(make_task):
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 700}],
                     [{"tail": "N1", "pos": "JFK", "end_req": "BOS"}, {"tail": "N2", "pos": "BOS", "end_req": "JFK"}])
    run = execute(task, {})
    assert run.ends == {"N1": "BOS", "N2": "BOS"}
    assert [(f.orig, f.dest) for f in run.ferries] == [("BOS", "JFK")]
    task = make_task([], [{"tail": "N1", "pos": "JFK", "end_req": "BOS"},
                          {"tail": "N2", "pos": "BOS", "end_req": "JFK"},
                          {"tail": "E1", "pos": "DCA", "end_req": "MCO", "family": "E190"}])
    run = execute(task, {})
    assert [(f.tail, f.family, f.orig, f.dest) for f in run.ferries] == [("E1", "E190", "DCA", "MCO")]


def test_the_ferry_matching_is_optimal():
    rng = random.Random(7)
    for n in range(1, 7):
        for _ in range(30):
            m = [[rng.randint(0, 50) for _ in range(n)] for _ in range(n)]
            best = min(sum(m[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
            col = _assign(m)
            assert sorted(col) == list(range(n)) and sum(m[i][col[i]] for i in range(n)) == best


def test_stored_reference_plans_reproduce_their_costs_exactly(pack):
    """The gate: the hindsight optimum, run as a plan, costs exactly the MIP's objective, and so on for each
    reference; the solver-free audit: no flight leaves before its earliest time, after its deadline or off its
    hold, every aircraft's legs chain with the family's turn, and no capped hour overflows."""
    for task in pack:
        ref = task.reference
        hindsight = execute(task, as_plan(ref["hindsight_plan"]))
        assert hindsight.cost.total == ref["hindsight"] == ref["anchor"], task.task_id
        assert hindsight.counts[AUTO_CANCELLED] == 0
        for i, v in ref["hindsight_plan"].items():
            assert hindsight.flights[int(i)].dep == v["dep"]
        assert execute(task, as_plan(ref["follower_plan"])).cost.total == ref["follower"]
        assert execute(task, {}).cost.total == ref["naive"]
        assert execute(task, policies.airline(task)).cost.total == ref["airline"]
        assert ref["floor"] <= ref["hindsight"] <= ref["naive"]
        for run in (hindsight, execute(task, {}), execute(task, policies.airline(task))):
            assert audit(task, run) == []


def test_execution_is_deterministic(pack):
    task = pack[0]
    plan = as_plan(task.reference["follower_plan"])
    assert execute(task, plan) == execute(task, dict(reversed(list(plan.items()))))


def audit(task, run) -> list[str]:
    problems = []
    pos = {t.tail: (t.pos, t.ready) for t in task.tails}
    used = {}
    for i in sorted((i for i, r in run.flights.items() if r.status == FLOWN), key=lambda i: run.flights[i].dep):
        f, r = task.flight(i), run.flights[i]
        if not task.earliest(f, f.lb) <= r.dep <= task.deadline(f, f.lb):
            problems.append(f"{f.flight} outside its earliest time and deadline")
        station, ready = pos[r.tail]
        if station != f.orig or ready > r.dep:
            problems.append(f"{f.flight} on {r.tail}, which is at {station} from {ready}")
        pos[r.tail] = (f.dest, r.arr + task.turn_for(f.family))
        for key, h in ((f"{f.orig}:dep", r.dep // 60), (f"{f.dest}:arr", r.arr // 60)):
            used[(key, h)] = used.get((key, h), 0) + 1
    problems += [f"{k} {h} over its cap" for (k, h), n in used.items() if n > task.caps.get(k, {}).get(h, n)]
    for tail, legs in run.legs.items():
        if run.ends[tail] != (task.flight(legs[-1]).dest if legs else next(t.pos for t in task.tails
                                                                             if t.tail == tail)):
            problems.append(f"{tail} ends off its last leg")
    return problems
