"""The MiMo release behind the environment contract (app/envs/mimo.py, app/mimo): its tasks read the way the MiMo RL
Environment Explorer reads them, run as their Harbor conversion or on the release's own harness, and linked to their
Harbor twins both ways (no network: the release's rows and task views are stubbed; its browsing index is the real one).

What is checked: the MiMo adapter claims the release and nothing else; tiles, scoped facets and the map are coherent;
every task is a card; a raw row's ref opens the same task; the Harbor twin is the default unless its grader can't score
here; a rubric task's judge is required and must come from the offered pool; the MiMo runner gets the task and the
explorer's fields, never a token in its record; a twin links back to the release; a MiMo rollout's page serves its
events, scrubbed for the public; custom data refuses paths outside the workspace, and raw files are sandboxed.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from app import auth, main, models, settings, store
from app.envs import contract as c, mimo, registry
from app.mimo import catalog as mcat

client = TestClient(main.app, base_url="https://testserver")
TOKEN = "hf_mimo_secret_token_0004"
SPEC = mimo.SPEC
GENERAL = "s3k_0052_accounting_audit_tax_en_t2_rl_008"


def cookie() -> dict[str, str]:
    session = {"token": TOKEN, "name": "erin", "avatar": None, "via": "token", "exp": time.time() + 3600}
    return {"Cookie": f"{auth.COOKIE}={auth._box.encrypt(json.dumps(session).encode()).decode()}"}


def view(tid: str) -> dict:
    rec = mcat.record(tid)
    d = rec["d"]
    v = {"id": tid, "domain": d, "title": rec["t"], "short_title": rec["t"], "facets": rec["f"], "brief": "Do the thing.", "meta": [],
         "image": "docker.io/x:y", "runnable": True, "environment": {"cwd": "/work", "image": "docker.io/x:y"}, "prompt": {"parts": [["task", "Do the thing."]], "task_is_brief": True}}
    if d == "general":
        v["verify"] = {"kind": "rubric", "summary": "Checked.", "needs_judge": "text", "formula": "Σ",
                       "checks": [{"id": "c1", "tier": "critical", "method": "llm", "weight": 2, "question": "Is it right?", "files": []}]}
    else:
        v["verify"] = {"kind": "tests", "summary": "Tests.", "files": [{"path": "t.py", "added": 1, "removed": 0, "new": True}], "formula": "1 or 0"}
    return v


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    ids = [e["id"] for e in mcat.index()["envs"]]
    by_domain: dict[str, list[str]] = {}
    for e in mcat.index()["envs"]:
        by_domain.setdefault(e["d"], []).append(e["id"])
    monkeypatch.setattr(mcat, "rows", lambda: {tid: {"domain": d} for d, tids in by_domain.items() for tid in tids})
    monkeypatch.setattr(mcat, "view", lambda tid: view(tid) if tid in set(ids) else None)
    mimo._order.cache_clear(); mimo._position.cache_clear()
    monkeypatch.setattr(registry, "_meta", lambda kind, spec, token: {"id": spec, "sha": "abc", "restricted": False})
    monkeypatch.setattr(registry, "_choice", {})
    monkeypatch.setattr("app.mimo.runner.domains.run_defaults", lambda tid, d: {"steps": 100, "timeout_min": 30, "max_tokens": None})
    monkeypatch.setattr(models, "get", lambda mid: {"id": mid, "tools": True, "provider": "x"})
    monkeypatch.setattr(models, "catalog", lambda: {"text_judges": [{"id": "judge/text"}], "vision_judges": [{"id": "judge/vision"}], "agents": []})
    monkeypatch.setattr(settings, "_org_members", lambda: set())
    yield
    mimo._order.cache_clear(); mimo._position.cache_clear()


def twin_opt(monkeypatch, warnings=()):
    """A Harbor twin for every task, whose grader may need a key it won't get here."""
    from app.envs import harbor

    monkeypatch.setattr(mimo.MiMoAdapter, "_twin", lambda self, tid, d: {"dataset": f"FineEnvs/MiMo-V2.6-RL-harbor-{d}", "path": f"tasks/{tid}"})
    toml = "[verifier.env]\n" + "".join(f'{k} = "${{{k}}}"\n' for k in warnings)
    monkeypatch.setattr("app.catalog.task", lambda d, p, t=None: {"runnable": {"ok": True, "how": "image"}, "verifier": {"env": list(warnings)},
                                                                   "env": {"image": "img"}, "bytes": 1, "toml": toml})
    return harbor


def test_the_release_is_read_by_the_mimo_adapter_only():
    a = mimo.MiMoAdapter()
    assert a.detect("dataset", {"id": SPEC}) == 1.0 and a.detect("dataset", {"id": "someone/else"}) == 0
    assert registry.resolve("dataset", SPEC).adapter.id == "mimo"


def test_tiles_facets_and_map_are_coherent():
    s = client.get(f"/api/env/{SPEC}").json()
    keys = [f["key"] for f in s["facets"]]
    assert len(keys) == len(set(keys)), "one facet per key"
    tiles = {t["value"] for t in s["tiles"]["items"]}
    assert tiles == {"Code", "Webdev", "Cyber", "Music", "General"} and s["tiles"]["key"] == "domain"
    for f in s["facets"]:
        assert set(f["scope"]) <= tiles | {"*"}
    assert set(s["map"]["by_tile"]) == tiles and all(k in keys for k in s["map"]["by_tile"].values())
    assert s["order"] == "shuffle" and s["overview"][0]["collapsed"] is True


def test_every_task_is_a_card():
    d = client.get(f"/api/env/{SPEC}/tasks?all=1").json()
    assert d["total"] == len(mcat.index()["envs"]) == len(d["cards"])
    assert all(x["facets"]["domain"] and x["color"] for x in d["cards"])


def test_a_raw_rows_ref_opens_the_same_task(monkeypatch):
    twin_opt(monkeypatch)
    first = mimo._order()["code"][6]
    v = client.get(f"/api/env/{SPEC}/task?ref=code/train/6").json()
    assert v["ref"] == first and v["nav"]["index"] == 6
    assert client.get(f"/api/env/{SPEC}/task?ref=code/train/999999").status_code == 404
    assert client.get(f"/api/env/{SPEC}/task?ref=nope").status_code == 404
    # by its own ref, its row is an alias (rollouts recorded against the row are listed with it)
    assert {"env": SPEC, "ref": "code/train/6"} in client.get(f"/api/env/{SPEC}/task?ref={first}").json()["aliases"]


def test_harbor_is_the_default_and_a_model_graded_twin_gets_a_judge(monkeypatch):
    twin_opt(monkeypatch)
    code = mimo._order()["code"][0]
    opts = client.get(f"/api/env/{SPEC}/task?ref={code}").json()["run"]["options"]
    assert [(o["runner"], o["default"]) for o in opts] == [("harbor", True), ("mimo", False)]
    assert not any(f["key"] == "judge" for f in opts[0]["fields"])
    twin_opt(monkeypatch, warnings=["GA_JUDGE_KEY"])   # its grader calls a model: Harbor still runs it, through the relay
    opts = client.get(f"/api/env/{SPEC}/task?ref={GENERAL}").json()["run"]["options"]
    assert [(o["runner"], o["default"]) for o in opts] == [("harbor", True), ("mimo", False)]
    for o in opts:
        judge = next(f for f in o["fields"] if f["key"] == "judge")
        assert judge["type"] == "model" and judge["pool"] == "text_judges" and judge["required"]


def test_mimo_runs_get_the_task_and_checked_fields(monkeypatch):
    twin_opt(monkeypatch)
    from app.mimo.runner import core

    seen = {}

    def fake_submit(user, token, task, model, provider, judge, **kw):
        seen.update(user=user, token=token, task=task, judge=judge, **kw)
        return store.create({"id": "20260103-000000-mimo01", "user": user, "task_id": task["id"], "status": "queued", **kw["extra"]})

    monkeypatch.setattr(core, "submit", fake_submit)
    post = lambda **kw: client.post("/api/runs", json={"dataset": SPEC, "path": GENERAL, "runner": "mimo", "model": "m/x", **kw}, headers=cookie())
    assert post().status_code == 400                                       # a rubric task needs a judge
    assert post(fields={"judge": "not/offered"}).status_code == 400        # from the pool only
    assert post(fields={"judge": "judge/text", "thinking": "loud"}).status_code == 400
    r = post(fields={"judge": "judge/text", "thinking": "high", "temperature": 0.5})
    assert r.status_code == 200, r.text
    assert seen["task"]["id"] == GENERAL and seen["judge"] == "judge/text" and seen["params"] == {"thinking": "high", "temperature": 0.5}
    assert seen["extra"] == {"dataset": SPEC, "path": GENERAL, "env": SPEC, "adapter": "mimo", "runner": "mimo", "restricted": False, "sha": "abc"}
    assert TOKEN not in json.dumps(store.get("20260103-000000-mimo01"))


def test_a_harbor_twin_links_back_to_the_release():
    tid = mimo._order()["code"][3]
    from app.envs.processors import _slug

    env = c.Env("dataset", "FineEnvs/MiMo-V2.6-RL-harbor-code", {}, None)
    links = mimo.harbor_to_mimo(env, f"tasks/{_slug(tid)}", {})
    assert links and links[0]["rel"] == "same" and links[0]["href"] == f"/t/{SPEC}/{tid}"
    assert (SPEC, tid) in mimo.harbor_aliases(env, f"tasks/{_slug(tid)}")
    assert mimo.harbor_to_mimo(c.Env("dataset", "someone/else", {}, None), f"tasks/{_slug(tid)}", {}) == []


def test_a_mimo_rollouts_page_serves_its_events_scrubbed_for_the_public():
    rid = "20260103-000000-mimo02"
    store.create({"id": rid, "user": "erin", "task_id": GENERAL, "dataset": SPEC, "path": GENERAL, "runner": "mimo", "status": "done", "reward": 1.0,
                  "visibility": "public", "title": "T", "model": "m/x", "endpoint": {"base_url": "https://erin.example/v1", "host": "erin.example"}})
    store.append_events(rid, [{"i": 0, "t": 0, "kind": "text", "text": "hello from erin at https://erin.example/v1"}])
    mine = client.get(f"/api/runs/{rid}", headers=cookie()).json()
    assert mine["events"][0]["text"].startswith("hello from erin") and mine["run"]["is_owner"]
    pub = client.get(f"/api/runs/{rid}").json()
    assert "erin" not in json.dumps(pub["events"]) and "erin" not in json.dumps(pub["run"])
    assert client.get(f"/api/runs/{rid}?after=1").json()["events"] == []


def test_custom_data_stays_inside_the_workspace(monkeypatch):
    def outside(tid, rel):
        raise PermissionError(rel)

    monkeypatch.setattr(mcat, "workspace_path", outside)
    assert client.get(f"/api/env/{SPEC}/data?ref={GENERAL}&part=preview&path=../../etc/passwd").status_code == 404
    assert client.get(f"/api/env/{SPEC}/raw?ref={GENERAL}&f=../../etc/passwd").status_code == 404
    assert client.get(f"/api/env/{SPEC}/data?ref={GENERAL}&part=nothing").status_code == 404


def test_raw_files_are_sandboxed(monkeypatch, tmp_path):
    page = tmp_path / "index.html"
    page.write_text("<script>alert(1)</script>")
    monkeypatch.setattr(mcat, "workspace_path", lambda tid, rel: page)
    r = client.get(f"/api/env/{SPEC}/raw?ref={GENERAL}&f=index.html")
    assert r.status_code == 200 and r.headers["content-security-policy"] == "sandbox" and r.headers["x-content-type-options"] == "nosniff"
