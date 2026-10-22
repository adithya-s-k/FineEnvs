from airops_core.core.known import (
    CAP_LOOK,
    FREEZE,
    LOOK,
    STEP,
    frozen,
    known_caps,
    known_lb,
    news_until,
    view,
    watch_times,
)
from airops_core.core.sim import execute


def test_a_delay_is_known_two_hours_before_the_scheduled_departure(pack):
    task = pack[1]
    for t in (task.start, task.start + 60, task.start + 300):
        lb = known_lb(task, t)
        for f in task.flights:
            assert lb[f.id] == (f.lb if f.sched_dep <= t + LOOK else 0)
    f = max(task.flights, key=lambda f: f.lb)
    assert known_lb(task, f.sched_dep - LOOK)[f.id] == f.lb and known_lb(task, f.sched_dep - LOOK - 1)[f.id] == 0


def test_a_cap_is_known_four_hours_before_its_hour_and_nominal_until_then(pack):
    task = pack[1]
    t = task.start
    caps = known_caps(task, t)
    assert set(caps) == set(task.caps) | set(task.nominal)
    for key, hours in caps.items():
        for h, v in hours.items():
            assert v == (task.caps[key][h] if 60 * h <= t + CAP_LOOK else task.nominal[key][h])
    assert known_caps(task, 24 * 60) == {k: dict(sorted(v.items())) for k, v in task.caps.items()}
    assert view(task, t) == (known_lb(task, t), caps)


def test_news_arrives_as_issued(make_task):
    news = [{"id": f"n{i}", "at": at, "kind": "metar", "speaker": "KJFK", "text": "x", "data": {}}
            for i, at in enumerate((590, 600, 661))]
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 700}], [{"tail": "N1", "pos": "JFK"}], news=news)
    assert [n.id for n in news_until(task, 600)] == ["n0", "n1"]
    assert [n.id for n in news_until(task, 660)] == ["n0", "n1"]
    assert len(news_until(task, 661)) == 3


def test_watches_run_hourly_from_the_start_until_the_day_is_known(pack):
    for task in pack:
        ts = watch_times(task)
        assert ts[0] == task.start and all(b - a == STEP for a, b in zip(ts, ts[1:], strict=False))
        assert ts[-1] + FREEZE <= max(f.sched_dep + f.dmax for f in task.flights)
        last = ts[-1]
        assert known_lb(task, last) == task.lb
        assert known_caps(task, last) == {k: dict(sorted(v.items())) for k, v in task.caps.items()}
        assert known_lb(task, last - STEP) != task.lb or known_caps(task, last - STEP) != known_caps(task, last)


def test_the_freeze_takes_what_leaves_within_the_hour_but_nothing_at_the_first_watch(make_task):
    task = make_task([{"orig": "JFK", "dest": "BOS", "dep": 650}, {"orig": "JFK", "dest": "DCA", "dep": 720},
                      {"orig": "JFK", "dest": "MCO", "dep": 700}, {"orig": "JFK", "dest": "BOS", "dep": 640}],
                     [{"tail": f"N{i}", "pos": "JFK"} for i in range(4)], start=600)
    plan = {2: {"cancel": False, "not_before": 730}, 3: {"cancel": True, "not_before": None}}
    run = execute(task, plan, *view(task, 660))
    assert frozen(task, run, task.start) == set()
    assert frozen(task, run, 640) == frozen(task, run, 660) == {0, 3}
    assert frozen(task, run, 661) == {0, 1, 3}
    assert frozen(task, run, 671) == {0, 1, 2, 3}
