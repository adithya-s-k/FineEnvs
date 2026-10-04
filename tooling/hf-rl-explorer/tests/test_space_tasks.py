import time

import pytest

from app import space_tasks as tasks, spaces_live as live


def test_multiple_environments_count_once_and_keep_unknowns(monkeypatch):
    docs = {"/list_environments": ["a", "a", "b", "../bad"],
            "/a/splits": [{"name": "train", "num_tasks": 12}, {"name": "train", "num_tasks": 12}],
            "/b/splits": [{"name": "test", "num_tasks": True}, {"name": "valid", "num_tasks": 0}]}
    monkeypatch.setattr(live, "_get_json", lambda rec, path: docs[path])
    d = tasks.discover({}, {"/{env_name}/splits"})
    assert [e["env"] for e in d["environments"]] == ["a", "b"]
    s = tasks.summary(d)
    assert s["tasks"] == 12 and s["counted_splits"] == 2 and s["unknown_splits"] == 1
    assert not s["complete"]


@pytest.mark.parametrize("value", [True, -1, "100", 2.5, float("inf"), 10**30])
def test_invalid_counts_never_inflate_totals(value):
    assert tasks.count(value) is None


def test_census_filters_visibility_expiry_and_inactive_spaces():
    now = time.time()
    s = {"tasks": 19, "counted_splits": 1, "checked_at": now, "complete": True}
    records = {"a": {"stage": "RUNNING", "task_catalog": s},
               "private": {"stage": "RUNNING", "task_catalog": s},
               "asleep": {"stage": "SLEEPING", "task_catalog": s},
               "stale": {"stage": "RUNNING", "task_catalog": {**s, "checked_at": now - tasks.FRESH}}}
    assert tasks.census(records, ["a", "asleep", "stale"], now)["tasks"] == 19
    assert tasks.census(records, ["a"], now - 1)["tasks"] == 0


def test_task_environment_must_be_advertised(monkeypatch):
    monkeypatch.setattr(live, "probe", lambda spec: {"running": True, "task_api": {"env": "a", "environments": [{"env": "a"}, {"env": "b"}]}})
    monkeypatch.setattr(live, "record", lambda spec: {})
    assert live._task_api("o/s", "b")[1] == "b"
    with pytest.raises(live.SpaceError):
        live._task_api("o/s", "../secrets")
