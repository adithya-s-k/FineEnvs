"""Security properties of the explorer, end to end through the HTTP API (no network: the Hub is stubbed).

    uv run pytest tests -q

What is checked: who may see and change what (admins, owners, everyone), that sessions can't be forged, that
cross-site requests are refused, that private datasets and private rollouts stay private, that answers and tokens
never reach a page, that a visitor's endpoint can't point the server at a private address, and the headers.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

_TMP = os.environ.get("RLX_TEST_TMP") or tempfile.mkdtemp(prefix="rlx-test-")   # conftest.py's, shared by every module
os.environ.update(STORAGE_DIR=_TMP, RLX_CACHE_DIR=f"{_TMP}/cache", OAUTH_CLIENT_ID="test-client", SESSION_SECRET="test-secret",
                  RLX_WARM="0", RLX_ADMIN_ORG="FineEnvs")

from fastapi.testclient import TestClient  # noqa: E402

from app import admin_app, auth, catalog, config, endpoints, main, models, runner, settings, store  # noqa: E402

TOKENS = {"alice": "hf_alice_secret_token_0001", "bob": "hf_bob_secret_token_0002"}


def cookie(name: str, exp: float | None = None) -> dict[str, str]:
    session = {"token": TOKENS[name], "name": name, "avatar": None, "via": "token", "exp": exp or time.time() + 3600}
    return {"Cookie": f"{auth.COOKIE}={auth._box.encrypt(json.dumps(session).encode()).decode()}"}


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    """No Hub: alice is a FineEnvs member, bob isn't; the catalog is empty."""
    monkeypatch.setattr(settings, "_org_members", lambda: {"alice"})
    monkeypatch.setattr(catalog, "environments", lambda include_hidden=False: [])
    monkeypatch.setattr(models, "get", lambda mid: {"id": mid, "tools": True, "provider": "x"})


client = TestClient(main.app, base_url="https://testserver")
admin = TestClient(admin_app.app, base_url="https://testserver")   # its own app, on its own private Space


def runnable_task(monkeypatch, **row):
    """A task any environment's adapter says runs with the Harbor runner (app/envs/registry.run_task, stubbed)."""
    from types import SimpleNamespace

    from app.envs import registry

    row = {"title": "T", "restricted": False, "sha": "abc", "runnable": {"ok": True, "how": "image"}, "image": "python:3.12", **row}
    opts = [{"runner": "harbor", "label": "Harbor", "ok": True, "why": "", "fields": [], "harnesses": None}]
    monkeypatch.setattr(registry, "run_task", lambda d, p, t=None: (SimpleNamespace(key=d, adapter=SimpleNamespace(id="test")), row, opts))
    monkeypatch.setattr(runner.Rollout, "execute", lambda self: None)


def run(**kw) -> dict:
    rid = kw.pop("id", f"20260101-000000-{os.urandom(3).hex()}")
    base = {"id": rid, "user": "alice", "dataset": "org/ds", "path": "tasks/t1", "task_id": "org/ds:tasks/t1", "title": "T",
            "model": "m/x", "harness": "opencode", "status": "done", "reward": 1.0, "visibility": "public"}
    return store.create({**base, **kw})


# ── sessions ─────────────────────────────────────────────────────────────────
def test_signed_out_and_forged_sessions_are_nobody():
    assert client.get("/api/runs").status_code == 401
    assert client.get("/api/runs", headers={"Cookie": f"{auth.COOKIE}=gAAAAABforged"}).status_code == 401
    assert client.get("/api/runs", headers=cookie("alice", exp=time.time() - 10)).status_code == 401
    assert client.get("/api/runs", headers=cookie("alice")).status_code == 200


def test_me_never_returns_the_token():
    body = client.get("/api/me", headers=cookie("alice")).text
    assert TOKENS["alice"] not in body
    assert json.loads(body)["user"]["name"] == "alice"


# ── admin ────────────────────────────────────────────────────────────────────
ADMIN_GETS = ["/api/admin/overview", "/api/admin/rollouts", "/api/admin/environments", "/api/admin/collections",
              "/api/admin/indexes", "/api/admin/settings", "/api/admin/audit"]


@pytest.mark.parametrize("path", ADMIN_GETS)
def test_admin_is_for_org_members_only(path):
    assert admin.get(path).status_code == 401
    assert admin.get(path, headers=cookie("bob")).status_code == 403
    assert admin.get(path, headers=cookie("alice")).status_code == 200


@pytest.mark.parametrize("path", ADMIN_GETS)
def test_the_public_explorer_has_no_admin(path):
    r = client.get(path, headers=cookie("alice"))
    assert r.status_code == 404 or "Admin" not in r.text   # the static fallback serves nothing there


def test_admin_writes_refused_for_others_and_audited():
    body = {"rollouts_enabled": True, "max_active": 5, "max_per_user": 2, "agents": ["opencode"], "announcement": "hi"}
    assert admin.put("/api/admin/settings", json=body).status_code == 401
    assert admin.put("/api/admin/settings", json=body, headers=cookie("bob")).status_code == 403
    assert admin.post("/api/admin/environments", json={"key": "org/ds", "action": "hide"}, headers=cookie("bob")).status_code == 403
    r = admin.put("/api/admin/settings", json=body, headers=cookie("alice"))
    assert r.status_code == 200 and r.json()["max_active"] == 5
    entry = settings.audit(1)[0]
    assert entry["user"] == "alice" and entry["after"]["max_active"] == 5


def test_admin_input_is_validated():
    h = cookie("alice")
    assert admin.put("/api/admin/settings", json={"rollouts_enabled": True, "max_active": 0, "max_per_user": 2, "agents": []}, headers=h).status_code == 422
    assert admin.put("/api/admin/settings", json={"rollouts_enabled": True, "max_active": 3, "max_per_user": 2, "agents": ["rm -rf"]}, headers=h).status_code == 400
    bad_key = {"collections": [{"id": "x1", "group": "X", "icon": "grid", "color": "code", "ids": ["not a key"]}]}
    assert admin.put("/api/admin/collections", json=bad_key, headers=h).status_code == 400
    bad_id = {"collections": [{"id": "Bad Id", "group": "X", "icon": "grid", "color": "code", "ids": []}]}
    assert admin.put("/api/admin/collections", json=bad_id, headers=h).status_code == 422
    assert admin.post("/api/admin/environments", json={"key": "../../etc", "action": "pin"}, headers=h).status_code == 400


def test_session_name_cannot_be_claimed_by_a_header():
    # the admin check reads the name from the encrypted session only
    assert admin.get("/api/admin/settings", headers={**cookie("bob"), "X-User": "alice", "X-Forwarded-User": "alice"}).status_code == 403


# ── cross-site requests ──────────────────────────────────────────────────────
def test_cross_site_posts_are_refused():
    h = cookie("alice")
    r = client.post("/api/runs", json={"dataset": "org/ds"}, headers={**h, "Origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/runs", json={"dataset": "org/ds"}, headers={**h, "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    r = admin.put("/api/admin/settings", json={}, headers={**h, "Origin": "https://evil.example"})
    assert r.status_code == 403


# ── rollouts: who sees what ──────────────────────────────────────────────────
def test_private_rollouts_are_owner_only():
    r = run(visibility="private")
    assert client.get(f"/api/runs/{r['id']}").status_code == 404
    assert client.get(f"/api/runs/{r['id']}", headers=cookie("bob")).status_code == 404
    assert client.get(f"/api/runs/{r['id']}", headers=cookie("alice")).status_code == 200
    assert client.post(f"/api/runs/{r['id']}/visibility", json={"visibility": "public"}, headers=cookie("bob")).status_code == 404
    assert client.post(f"/api/runs/{r['id']}/cancel", headers=cookie("bob")).status_code == 404


def test_public_rollouts_hide_who_ran_them():
    r = run(endpoint={"base_url": "https://llm.alice-corp.example/v1", "host": "llm.alice-corp.example", "model": "m"})
    store.write_artifact(r["id"], "trajectory.json", json.dumps([{"role": "user", "content": "hello alice, use https://llm.alice-corp.example/v1"}]))
    body = client.get(f"/api/runs/{r['id']}").text
    data = json.loads(body)
    assert "user" not in data["run"] and data["run"]["endpoint"] == {"custom": True}
    assert "alice" not in body and "alice-corp" not in body


def test_ungraded_unfinished_and_restricted_rollouts_stay_off_community():
    keep = run(dataset="org/community-check")
    run(dataset="org/community-check", status="running", reward=None)
    run(dataset="org/community-check", status="failed", reward=None)
    run(dataset="org/community-check", restricted=True)
    run(dataset="org/community-check", visibility="private")
    ids = [x["id"] for x in client.get("/api/community?dataset=org/community-check").json()["runs"]]
    assert ids == [keep["id"]]


def test_rollouts_on_private_datasets_cannot_go_public():
    r = run(visibility="private", restricted=True)
    assert client.post(f"/api/runs/{r['id']}/visibility", json={"visibility": "public"}, headers=cookie("alice")).status_code == 400


def test_tokens_never_reach_the_store(monkeypatch):
    runnable_task(monkeypatch)
    r = client.post("/api/runs", json={"dataset": "org/ds", "path": "tasks/t1", "model": "m/x"}, headers=cookie("alice"))
    assert r.status_code == 200, r.text
    assert TOKENS["alice"] not in r.text
    for f in Path(_TMP).rglob("*"):
        if f.is_file():
            assert TOKENS["alice"].encode() not in f.read_bytes(), f


def test_paused_rollouts_are_refused(monkeypatch):
    runnable_task(monkeypatch)
    settings.save("alice", {"rollouts_enabled": False})
    try:
        r = client.post("/api/runs", json={"dataset": "org/ds", "path": "tasks/t1", "model": "m/x"}, headers=cookie("alice"))
        assert r.status_code == 429 and "paused" in r.json()["detail"]
    finally:
        settings.save("alice", {"rollouts_enabled": True})


# ── private datasets ─────────────────────────────────────────────────────────
def test_private_datasets_need_a_token_that_can_read_them(monkeypatch):
    calls = []

    def fetch(spec, token):   # the Hub: only alice's token can read it
        calls.append(token)
        if token == TOKENS["alice"]:
            return {"id": spec, "sha": "abc", "restricted": True}
        raise PermissionError("private")

    monkeypatch.setattr(catalog, "_fetch_info", fetch)
    catalog._memo.clear()
    with pytest.raises(PermissionError):
        catalog.info("org/private-ds")
    assert catalog.info("org/private-ds", TOKENS["alice"])["restricted"]
    with pytest.raises(PermissionError):   # bob's token is checked on its own, never served alice's cached answer
        catalog.info("org/private-ds", TOKENS["bob"])
    with pytest.raises(PermissionError):   # and signed out, still nothing
        catalog.info("org/private-ds")
    assert TOKENS["bob"] in calls


# ── answers stay out ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("rel,text,withheld", [
    ("solution/solve.sh", None, True),
    ("tests/test.sh", "pytest /tests", False),
    ("tests/grade.py", "import json", False),
    ("tests/expected_output.json", None, True),
    ("tests/verifier/verifier_meta.json", None, True),
    ("tests/grade.json", None, True),
    ("tests/check.py", "GOLD = {'gold_answer': 42}", True),
    ("environment/setup/files/expected_func.json", None, True),
    ("instruction.md", "find the gold_answer", False),
])
def test_answers_are_withheld(rel, text, withheld):
    assert catalog._withheld(rel, text) is withheld


def test_task_toml_masks_answer_keys():
    out = catalog._masked_toml('[metadata]\ngold_answer = "42"\nexpected_output = "x"\ndifficulty = "easy"')
    assert '"42"' not in out and '"x"' not in out and 'difficulty = "easy"' in out


def test_task_files_stay_inside_the_task(monkeypatch):
    monkeypatch.setattr(catalog, "_task_row", lambda s, p, t=None: ({}, {"sha": "abc"}))
    monkeypatch.setattr(catalog, "_task_texts", lambda s, sha, p, t: ({"tests/test.sh": 10, "solution/solve.sh": 5}, {"tests/test.sh": "echo"}, set()))
    assert catalog.task_file("org/ds", "tasks/t1", "../t2/task.toml")["error"]
    assert catalog.task_file("org/ds", "tasks/t1", "/etc/passwd")["error"]
    assert catalog.task_file("org/ds", "tasks/t1", "solution/solve.sh").get("withheld")
    assert catalog.task_file("org/ds", "tasks/t1", "tests/test.sh")["text"] == "echo"


def test_dataset_ids_are_checked():
    for bad in ("../../etc", "a/b/c", "org/..", "", "org/ name"):
        with pytest.raises(ValueError):
            catalog.check_spec(bad)


# ── a visitor's endpoint ─────────────────────────────────────────────────────
@pytest.mark.parametrize("url", ["http://example.com/v1", "https://127.0.0.1/v1", "https://localhost/v1", "https://169.254.169.254/latest",
                                 "https://10.0.0.5/v1", "https://[::1]/v1", "https://user:pass@example.com/v1"])
def test_endpoints_must_be_public_https(url):
    with pytest.raises(endpoints.EndpointError):
        endpoints.check_url(url)


def test_space_hosts_are_pinned_to_hf_space():
    import re

    pat = re.compile(r"^https://[a-z0-9-]+\.hf\.space$")
    assert pat.match("https://openenv-coding-env.hf.space")
    for bad in ("https://evil.com", "https://x.hf.space.evil.com", "https://x.hf.space/../", "http://x.hf.space"):
        assert not pat.match(bad)


# ── headers ──────────────────────────────────────────────────────────────────
def test_security_headers():
    r = client.get("/")
    csp = r.headers["content-security-policy"]
    assert "object-src 'none'" in csp and "frame-ancestors 'self' https://huggingface.co" in csp and "script-src 'self'" in csp
    assert r.headers["x-content-type-options"] == "nosniff"


def test_local_mode_refuses_other_hosts(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_MODE", True)
    assert client.get("/api/me", headers={"Host": "evil.example"}).status_code == 403
    assert client.get("/api/me", headers={"Host": "localhost:8060"}).status_code == 200


# ── what the admin Space changes reaches the explorer through the bucket ─────
def test_admin_hides_a_rollout_from_community():
    r = run(dataset="org/moderation-check")
    assert [x["id"] for x in client.get("/api/community?dataset=org/moderation-check").json()["runs"]] == [r["id"]]
    assert admin.post(f"/api/admin/rollouts/{r['id']}/moderate", json={"hidden": True}, headers=cookie("alice")).status_code == 200
    assert client.get("/api/community?dataset=org/moderation-check").json()["runs"] == []
    assert client.get(f"/api/runs/{r['id']}").status_code == 404                        # gone for everyone
    assert client.get(f"/api/runs/{r['id']}", headers=cookie("alice")).status_code == 200   # its owner still has it
    admin.post(f"/api/admin/rollouts/{r['id']}/moderate", json={"hidden": False}, headers=cookie("alice"))
    assert client.get(f"/api/runs/{r['id']}").status_code == 200


def test_admin_stop_requests_reach_running_rollouts(monkeypatch):
    stopped = []
    monkeypatch.setattr(runner, "is_live", lambda rid: True)
    monkeypatch.setattr(runner, "cancel", lambda rid: stopped.append(rid) or True)
    r = run(status="running", reward=None, updated_at=time.time())
    assert admin.post(f"/api/admin/rollouts/{r['id']}/cancel", headers=cookie("alice")).status_code == 200
    assert admin.post(f"/api/admin/rollouts/{r['id']}/cancel", headers=cookie("bob")).status_code == 403
    import threading

    threading.Thread(target=runner.watch_cancel_requests, daemon=True).start()
    for _ in range(40):
        if r["id"] in stopped:
            break
        time.sleep(0.1)
    assert r["id"] in stopped


def test_answers_written_into_grader_scripts_are_masked():
    sh = "expected_answer=$(cat <<'ANSWER_EOF'\n138236\nANSWER_EOF\n)\nEXPECTED=\"42\"\nagent=$(cat /workdir/answer.txt)\n"
    out = catalog._mask_answers("tests/test.sh", sh)
    assert "138236" not in out and '"42"' not in out and "cat /workdir/answer.txt" in out
    py = 'GOLD_ANSWER = {"x": 1}\nresult = run()\n'
    assert "{\"x\": 1}" not in catalog._mask_answers("tests/grade.py", py)
    assert catalog._mask_answers("instruction.md", "expected = 5") == "expected = 5"   # only grader files


def test_no_network_tasks_are_refused_with_a_reason():
    r = catalog.runnable({"env": {"network": "no-network", "image": "python:3.12"}}, {})
    assert not r["ok"] and "internet" in r["why"]


def test_indexing_is_capped(monkeypatch):
    monkeypatch.setattr(catalog, "info", lambda spec, token=None: {"sha": "s1", "restricted": False})
    monkeypatch.setattr(catalog, "_read_index", lambda spec, sha=None: None)
    monkeypatch.setattr(catalog, "_build", lambda *a, **k: None)
    with catalog._jobs_lock:
        saved = dict(catalog._jobs)
        catalog._jobs.clear()
        catalog._jobs.update({f"o/busy{i}": {"state": "reading", "sha": "x"} for i in range(config.MAX_INDEX_JOBS)})
    try:
        with pytest.raises(RuntimeError):
            catalog.index_status("o/new-one")
    finally:
        with catalog._jobs_lock:
            catalog._jobs.clear()
            catalog._jobs.update(saved)


def test_huge_tasks_are_refused(monkeypatch):
    runnable_task(monkeypatch, bytes=50 * 1024**3)
    r = client.post("/api/runs", json={"dataset": "org/ds", "path": "tasks/t1", "model": "m/x"}, headers=cookie("alice"))
    assert r.status_code == 400 and "GB" in r.json()["detail"]


# ── the run list agrees with the run files ───────────────────────────────────
def test_list_follows_records_written_elsewhere():
    r = run(status="running", reward=None)
    store.list_runs()   # index it
    doc = json.loads((store.RUNS / r["id"] / "run.json").read_text())
    time.sleep(0.01)
    doc.update(status="done", reward=1.0, updated_at=time.time())
    (store.RUNS / r["id"] / "run.json").write_text(json.dumps(doc))   # another process finishes it
    store._synced_at = 0
    assert next(x for x in store.list_runs() if x["id"] == r["id"])["status"] == "done"


def test_only_silent_rollouts_are_marked_interrupted():
    busy = run(status="running", reward=None)                    # touched just now: someone is running it
    lost = run(status="running", reward=None)
    doc = json.loads((store.RUNS / lost["id"] / "run.json").read_text())
    doc["updated_at"] = time.time() - 3600                        # silent for an hour: its worker is gone
    (store.RUNS / lost["id"] / "run.json").write_text(json.dumps(doc))
    store._synced_at = 0
    store.mark_interrupted(set())
    assert store.get(busy["id"])["status"] == "running"
    assert store.get(lost["id"])["status"] == "interrupted"


# ── a task's ${VAR}s never reach this server's environment ───────────────────
def test_task_variables_never_come_from_this_servers_environment(monkeypatch):
    monkeypatch.setenv("SESSION_SECRET_PROBE", "s3cret-value")
    monkeypatch.setenv("HF_TOKEN", "hf_operator_token")
    with pytest.raises(runner.MissingTaskEnv):
        runner.resolve_task_env({"X": "${SESSION_SECRET_PROBE}"})
    with pytest.raises(runner.MissingTaskEnv):
        runner.resolve_task_env({"KEY": "${HF_TOKEN}"})
    assert runner.resolve_task_env({"A": "${HF_TOKEN:-none}", "B": "literal"}) == {"A": "none", "B": "literal"}
    token = runner._TASK_ENV.set({"environment": {"OPENAI_API_KEY": "cap-123"}, "verifier": {"JUDGE": "cap-456"}})
    try:
        assert runner.resolve_task_env({"K": "${OPENAI_API_KEY}"}) == {"K": "cap-123"}
        with pytest.raises(runner.MissingTaskEnv):   # the grader's values never reach the agent's phase
            runner.resolve_task_env({"J": "${JUDGE}"})
        assert runner._resolver("verifier")({"J": "${JUDGE}"}) == {"J": "cap-456"}
    finally:
        runner._TASK_ENV.reset(token)


def test_harbor_and_its_agents_are_cut_off_from_this_servers_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_operator_token")
    monkeypatch.setenv("OAUTH_CLIENT_SECRET", "oauth-secret")
    monkeypatch.setenv("PATH_LIKE_SETTING", "fine")
    runner._patch_harbor_env()
    import harbor.environments.base as env_base
    import harbor.trial.trial as trial
    import harbor.verifier.verifier as verifier
    from harbor.agents import base as agent_base
    from harbor.utils import env as harbor_env
    for mod, phase in ((env_base, "environment"), (trial, "any"), (verifier, "verifier"), (harbor_env, "environment")):
        assert getattr(mod.resolve_env_vars, "_rlx_phase", None) == phase, mod.__name__
    host = runner._HostEnv()
    assert "HF_TOKEN" not in host and "OAUTH_CLIENT_SECRET" not in host and host.get("PATH_LIKE_SETTING") == "fine"
    fake = type("A", (), {"_extra_env": {}})()
    sources = agent_base.BaseAgent._env_sources(fake)
    assert all("HF_TOKEN" not in s and "OAUTH_CLIENT_SECRET" not in s for s in sources)


def test_a_public_rollouts_files_are_images_only_to_others():
    r = run(user="alice", status="done", reward=1.0, visibility="public")
    store.write_artifact(r["id"], "trajectory.json", '[{"role": "user", "content": "from alice"}]')
    store.write_artifact(r["id"], "screenshot.jpg", b"\xff\xd8\xff")
    assert client.get(f"/api/runs/{r['id']}/artifacts/trajectory.json").status_code == 404
    assert client.get(f"/api/runs/{r['id']}/artifacts/screenshot.jpg").status_code == 200
    assert client.get(f"/api/runs/{r['id']}/artifacts/trajectory.json", headers=cookie("alice")).status_code == 200
    assert client.get(f"/api/runs/{r['id']}/artifacts/run.json", headers=cookie("alice")).status_code == 404


# ── sign-in ──────────────────────────────────────────────────────────────────
def _hub_signin(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(auth.httpx, "post", lambda *a, **k: SimpleNamespace(status_code=200, json=lambda: {"access_token": "hf_oauth_x", "scope": "openid inference-api jobs", "expires_in": 3600}))
    monkeypatch.setattr(auth.httpx, "get", lambda *a, **k: SimpleNamespace(json=lambda: {"preferred_username": "carol"}))


def test_oauth_state_lives_in_a_signed_cookie_for_one_browser(monkeypatch):
    _hub_signin(monkeypatch)
    c = TestClient(main.app, base_url="https://testserver")
    r = c.get("/login", follow_redirects=False)
    assert r.status_code in (302, 307)
    state = dict(x.split("=", 1) for x in r.headers["location"].split("?", 1)[1].split("&"))["state"]
    sent = r.cookies.get(auth.STATE_COOKIE)
    assert sent and state not in sent   # encrypted, not the state itself
    other = TestClient(main.app, base_url="https://testserver")   # another browser, the same callback URL: refused
    assert other.get(f"/login/callback?code=x&state={state}", follow_redirects=False).status_code == 400
    assert c.get("/login/callback?code=x&state=forged", cookies={auth.STATE_COOKIE: sent}, follow_redirects=False).status_code == 400
    ok = c.get(f"/login/callback?code=x&state={state}", cookies={auth.STATE_COOKIE: sent}, follow_redirects=False)
    assert ok.status_code in (302, 307) and auth.COOKIE in ok.headers.get("set-cookie", "")


def test_an_expired_sign_in_is_refused(monkeypatch):
    _hub_signin(monkeypatch)
    old = auth._box.encrypt_at_time(b"s1", int(time.time()) - auth.STATE_TTL - 5).decode()
    r = client.get("/login/callback?code=x&state=s1", cookies={auth.STATE_COOKIE: old}, follow_redirects=False)
    assert r.status_code == 400


def test_a_cancelled_sign_in_goes_home():
    r = client.get("/login/callback?error=access_denied", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/"


def test_pasting_tokens_is_rate_limited(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(auth, "TOKEN_LIMIT", auth.Limiter(3, 600))
    monkeypatch.setattr(auth.httpx, "get", lambda *a, **k: SimpleNamespace(status_code=401))
    codes = [client.post("/api/login/token", json={"token": "hf_" + "x" * 30}).status_code for _ in range(5)]
    assert codes == [400, 400, 400, 429, 429]


def test_two_submits_at_once_cant_both_take_the_last_place(monkeypatch):
    import threading as th

    runnable_task(monkeypatch)
    runner._live.clear()
    real_create = store.create

    def slow_create(doc):   # the record takes a moment to write: the window two submits used to race through
        time.sleep(0.3)
        return real_create(doc)

    monkeypatch.setattr(store, "create", slow_create)
    before = {k: settings.get(k) for k in ("max_per_user", "max_active")}
    settings.save("alice", {"rollouts_enabled": True, "max_per_user": 1, "max_active": 5})
    codes: list[int] = []
    try:
        ts = [th.Thread(target=lambda: codes.append(client.post("/api/runs", json={"dataset": "org/ds", "path": "tasks/t1", "model": "m/x"},
                                                                headers=cookie("bob")).status_code)) for _ in range(3)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert sorted(codes) == [200, 429, 429]
        assert runner._reserved == {}   # every reservation let go, the refused ones too
    finally:
        runner._live.clear()
        settings.save("alice", {"rollouts_enabled": True, **{k: v for k, v in before.items() if v is not None}})


def test_a_live_rollouts_events_stay_local_until_flushed(monkeypatch):
    r = run(user="alice", status="running")
    store.append_events(r["id"], [{"kind": "turn", "n": 1}])
    on_store = (store._dir(r["id"]) / "events.jsonl")
    assert store.read_events(r["id"]) == [{"kind": "turn", "n": 1}]   # the page reads the local copy at once
    assert on_store.read_text() == ""                                  # the store (a bucket) wasn't rewritten per event
    monkeypatch.setattr(store, "FLUSH_EVERY", 0.0)
    store.flush_due()
    assert json.loads(on_store.read_text().splitlines()[0]) == {"kind": "turn", "n": 1}
    store.append_events(r["id"], [{"kind": "turn", "n": 2}])
    store.update(r["id"], status="done")                               # it ended: everything is on the store, no local copy
    assert len(on_store.read_text().splitlines()) == 2 and not (store.LOCAL / r["id"]).exists()
    assert store.read_events(r["id"], after=1) == [{"kind": "turn", "n": 2}]


def test_events_a_stopped_process_left_on_local_disk_are_kept():
    r = run(user="alice", status="running")
    store.append_events(r["id"], [{"kind": "turn", "n": 1}])
    store._local.pop(r["id"])   # the process stopped: nothing in memory, the local file left behind
    store._write_json(store._dir(r["id"]) / "run.json", {**store.get(r["id"]), "updated_at": time.time() - store.STALE - 10})
    store._index_put(store.get(r["id"]))
    assert store.mark_interrupted(set()) >= 1
    assert store.get(r["id"])["status"] == "interrupted"
    assert store.read_events(r["id"]) == [{"kind": "turn", "n": 1}] and not (store.LOCAL / r["id"]).exists()


def test_a_visitors_endpoint_cant_rebind_to_a_private_address_later(monkeypatch):
    import socket

    answers = {"model.example": "93.184.216.34", "internal.example": "10.0.0.5"}

    def fake(host, port, *a, **k):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (answers[host], port))]

    monkeypatch.setattr(endpoints, "_real_getaddrinfo", fake)
    real = socket.getaddrinfo
    try:
        assert endpoints.check_url("https://model.example/v1") == "https://model.example/v1"
        assert socket.getaddrinfo("model.example", 443)[0][4][0] == "93.184.216.34"
        answers["model.example"] = "169.254.169.254"   # the name now points at the metadata service
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("model.example", 443)     # any client in this process (the capture proxy's too)
        assert socket.getaddrinfo("internal.example", 443)[0][4][0] == "10.0.0.5"   # names nobody gave us: untouched
    finally:
        socket.getaddrinfo = real
        endpoints._untrusted.discard("model.example")


def test_a_rollout_never_downloads_reference_solutions(monkeypatch):
    import huggingface_hub

    seen = {}
    monkeypatch.setattr(catalog, "info", lambda spec, token=None: {"sha": "abc", "restricted": False})
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: seen.update(k))
    catalog.task_dir("org/ds", "tasks/t1")
    from fnmatch import fnmatch
    for f in ("tasks/t1/solution/solve.sh", "tasks/t1/steps/s1/solution/solve.sh", "tasks/t1/steps/two/solution/deep/x.py"):
        assert any(fnmatch(f, p) for p in seen["ignore_patterns"]), f
    assert not any(fnmatch("tasks/t1/steps/s1/instruction.md", p) for p in seen["ignore_patterns"])


def test_the_admin_space_has_health_checks_and_nothing_else_open():
    assert admin.get("/healthz").json() == {"ok": True}
    r = admin.get("/readyz")
    assert r.status_code == 200 and r.json()["store"] == "ok"
    assert admin.get("/api/admin/settings").status_code == 401
