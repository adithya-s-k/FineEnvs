import pytest

from port_week_core.schedule import BULLETIN_HOURS, FREEZE_HOURS, MAX_TURNS, label, schedule, virtual_time

WATCHES = {
    "24B-w06x1-busy": 4, "24B-w07x1-busy": 7, "24B-w16x1-busy": 9, "24B-w35x1-busy": 8, "36A-w05x1-busy": 5,
    "36A-w06x1-busy": 6, "36A-w15x1-busy": 7, "36A-w17x1-busy": 5, "36A-w37x1-busy": 7,
    "24B-w06x1-standard": 4, "24B-w37x1-standard": 4, "36A-w06x1-standard": 3, "36A-w10x1-standard": 2,
    "36A-w15x1-standard": 4, "36A-w16x1-standard": 5, "36A-w17x1-standard": 5, "36A-w35x1-standard": 5,
    "36A-w37x1-standard": 5,
}


def test_the_eval_weeks_have_their_watch_counts(one_week):
    assert {t.task_id[5:-2]: len(schedule(t)) for t in one_week if t.split == "eval"} == WATCHES


def test_the_example_week(example):
    assert [(w.index, w.hour, [(n.event_id, n.name) for n in w.notices]) for w in schedule(example)] == [
        (0, 0, [("closure-2", "Terminal ops")]),
        (1, 30, [("extra-1", "MAERSK NUBA")]),
        (2, 42, [("emergency-7", "Harbour master")]),
        (3, 54, [("late-3", "CARLOTA B")]),
        (4, 84, [("late-4", "MAERSK NARMADA"), ("crane_outage-6", "Terminal ops")]),
        (5, 90, [("gale-0", "Harbour master")]),
        (6, 120, [("bunching-5", "Ship agents")]),
    ]


def test_every_disruption_is_revealed_once_in_task_order_within_its_bulletin(one_week):
    for task in one_week:
        watches = schedule(task)
        assert [w.index for w in watches] == list(range(len(watches))) and watches[0].hour == 0
        assert [w.hour for w in watches] == sorted({w.hour for w in watches})
        assert sorted(n.index for w in watches for n in w.notices) == list(range(len(task.disruptions)))
        for w in watches:
            assert w.hour % BULLETIN_HOURS == 0 and [n.index for n in w.notices] == sorted(n.index for n in w.notices)
            for n in w.notices:
                assert w.hour <= n.hour < w.hour + BULLETIN_HOURS
                assert (n.text, n.event, n.event_id) == (task.notices[n.index], task.disruptions[n.index],
                                                         f"{n.kind}-{n.index}")


def test_reveals_after_hour_0_lead_by_at_least_the_freeze(one_week):
    assert (len(one_week), sum(t.split == "eval" for t in one_week)) == (424, 18)
    short = [(t.task_id, n.event_id) for t in one_week for w in schedule(t) for n in w.notices
             if w.hour > 0 and n.event_hour - w.hour < FREEZE_HOURS]
    assert short == []


def test_a_ships_own_notice_comes_before_its_emergency_or_priority(one_week):
    for task in one_week:
        order = [n for w in schedule(task) for n in w.notices]
        for k, n in enumerate(order):
            if n.kind in ("emergency", "priority"):
                own = [m for m in order if m.kind in ("late", "extra") and m.event["ship"] == n.event["ship"]
                       or m.kind == "bunching" and n.event["ship"] in m.event["ships"]]
                assert all(order.index(m) < k for m in own)


def test_closures_are_known_at_hour_0_and_the_rest_by_their_leads(example):
    hours = {n.event_id: (n.hour, n.event_hour) for w in schedule(example) for n in w.notices}
    assert hours == {"gale-0": (91, 115), "extra-1": (34, 58), "closure-2": (0, 0), "late-3": (54, 78),
                     "late-4": (88, 112), "bunching-5": (120, 144), "crane_outage-6": (84, 108),
                     "emergency-7": (42, 54)}


def test_multi_week_tasks_are_refused(pack):
    task = next(t for t in pack.tasks if "x2-" in t.task_id)
    with pytest.raises(ValueError, match=f"^{task.task_id} is not a one-week task: live weeks are one week$"):
        schedule(task)


def test_turns_labels_and_virtual_time(example):
    assert MAX_TURNS == 47
    assert (label(0), label(30), label(120)) == ("Mon 00:00", "Tue 06:00", "Sat 00:00")
    assert (virtual_time(example, 0), virtual_time(example, 30)) == ("2024-02-12T00:00:00Z", "2024-02-13T06:00:00Z")
