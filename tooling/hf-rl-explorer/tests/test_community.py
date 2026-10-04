import json

import pytest
from fastapi.testclient import TestClient

from app import auth, community, main, settings, store

client = TestClient(main.app)


def record(id, runner="harbor", **fields):
    return {"id": id, "dataset": f"org/{runner}", "path": "task/1", "title": "A task", "user": "hidden-user",
            "runner": runner, "status": "done", "reward": 1, "visibility": "public", "model": "org/model",
            "created_at": 10, **fields}


@pytest.fixture
def records(monkeypatch):
    rows = [record("harbor", reward=-3, dataset="a/same"), record("mimo", "mimo", domain="code", reward=0.25),
            record("nemo", "nemo-gym", reward=1, dataset="b/same"), record("future", "custom-runtime", reward=20),
            record("private", visibility="private"), record("restricted", restricted=True),
            record("failed", status="failed", reward=None), record("hidden")]
    monkeypatch.setattr(store, "list_runs", lambda **kw: rows)
    monkeypatch.setattr(store, "has_artifact", lambda *a: False)
    monkeypatch.setattr(settings, "get", lambda key, default=None: ["hidden"] if key == "hidden_runs" else default)
    return rows


def test_all_saved_runners_share_the_feed_with_scoped_counts(records):
    data = client.get("/api/community").json()
    assert {r["id"] for r in data["runs"]} == {"harbor", "mimo", "nemo", "future"}
    assert data["total"] == data["stats"]["runs"] == 4
    assert data["stats"]["datasets"] == data["stats"]["runners"] == 4
    assert not data["stats"]["comparable"]
    assert "mean" not in data["stats"]  # no meaningless global reward average
    assert {r["group"] for r in data["runs"]} >= {"a/same", "b/same"}
    assert "hidden-user" not in json.dumps(data)
    for runner in ("harbor", "mimo", "nemo-gym", "custom-runtime"):
        subset = client.get("/api/community", params={"runner": runner}).json()
        assert subset["total"] == subset["stats"]["runs"] == subset["stats"]["datasets"] == 1
        assert {r["runner"] for r in subset["runs"]} == {runner}
        assert len(subset["facets"]["runner"]) == 4  # own facet retains alternatives
        assert len(subset["facets"]["dataset"]) == 1


def test_search_includes_environment_runner_and_model_and_empty_counts(records):
    for query, expected in (("b/same", "nemo"), ("nemo-gym", "nemo"), ("org/mimo model", "mimo")):
        data = client.get("/api/community", params={"q": query}).json()
        assert [r["id"] for r in data["runs"]] == [expected]
    empty = client.get("/api/community?dataset=nobody/missing").json()
    assert empty["total"] == empty["stats"]["datasets"] == empty["stats"]["models"] == empty["stats"]["runners"] == 0
    assert empty["stats"]["results"] == []


def test_reward_summaries_separate_metrics_domains_judges_and_revisions():
    rows = [record("a", reward=-4), record("b", reward=-2)]
    for field, value in (("reward_key", "latency"), ("domain", "code"), ("judge", "judge/new"), ("sha", "new-revision")):
        rows.append(record(field, reward=100, **{field: value}))
    data = community.stats(rows)
    assert len(data["results"]) == 5
    original = next(r for r in data["results"] if r["runs"] == 2)
    assert original["mean"] == -3 and original["max"] == -2
    assert community.stats([record("a", reward=1e308), record("b", reward=1e308)])["results"][0]["mean"] == 1e308


@pytest.mark.parametrize("value", [None, "1", True, {}, float("nan"), float("inf"), 10**1000])
def test_invalid_rewards_cannot_break_or_enter_public_feed(value, monkeypatch):
    monkeypatch.setattr(settings, "get", lambda *a: [])
    assert not main._shareable(record("bad", reward=value))


def test_pagination_sort_and_negative_reward_band(records):
    records.extend(record(f"r{i:02}", created_at=i, reward=-i, cost={"total": None}) for i in range(20))
    all_rows = client.get("/api/community?sort=reward_asc").json()["runs"]
    pages = [client.get("/api/community", params={"sort": "reward_asc", "offset": i, "limit": 5}).json()["runs"] for i in range(0, 24, 5)]
    assert [r["id"] for page in pages for r in page] == [r["id"] for r in all_rows]
    negative = client.get("/api/community?reward=zero").json()
    assert negative["total"] == 21 and all(r["reward"] <= 0 for r in negative["runs"])
    assert client.get("/api/community?sort=cost_asc").status_code == 200


@pytest.mark.parametrize("runner,domain,events", [("harbor", None, False), ("mimo", "code", True), ("nemo-gym", "nemo-gym", True), (None, "code", True)])
def test_trace_routes_match_runner_and_keep_public_scrubbing(monkeypatch, runner, domain, events):
    from app.mimo.runner import core
    r = record("trace", runner, domain=domain)
    monkeypatch.setattr(store, "get", lambda id: r)
    monkeypatch.setattr(settings, "get", lambda *a: [])
    monkeypatch.setattr(auth, "current_user", lambda request: None)
    monkeypatch.setattr(store, "read_artifact", lambda *a: b'[{"content":"hidden-user"}]')
    monkeypatch.setattr(main.runner, "is_live", lambda *a: False)
    monkeypatch.setattr(core, "live", lambda *a: None)
    monkeypatch.setattr(core, "events", lambda *a: [{"content": "hidden-user"}])
    response = client.get("/api/runs/trace")
    assert response.status_code == 200 and "hidden-user" not in response.text
    assert ("events" in response.json()) == events
    assert response.json()["run"]["runner"] == (runner or "mimo")
