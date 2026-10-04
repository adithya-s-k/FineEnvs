"""Regression coverage for the security review, with no Hub or model requests."""

import json
import os
import tempfile
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

os.environ.update(OAUTH_CLIENT_ID="test-client", SESSION_SECRET="test-secret",
                  STORAGE_DIR=tempfile.mkdtemp(prefix="mimo-security-"))

from fastapi.testclient import TestClient

from app import auth, config, main, store


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_MODE", False)
    monkeypatch.setattr(store, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(store, "_index", {})
    monkeypatch.setattr(auth, "_states", {})
    return TestClient(main.app, base_url="https://testserver")


def test_public_artifacts_do_not_bypass_scrubbing(client):
    rid = "test-run"
    store.create({"id": rid, "user": "alice", "status": "done", "reward": 1,
                  "visibility": "public", "domain": "webdev"})
    store.write_artifact(rid, "trajectory.json", '{"owner":"alice"}')
    store.write_artifact(rid, "screenshot.jpg", b"\xff\xd8\xff")
    store.append_events(rid, [{"text": "alice private endpoint"}])
    for name in ("run.json", "events.jsonl", "trajectory.json"):
        assert client.get(f"/api/runs/{rid}/artifacts/{name}").status_code == 404
    assert client.get(f"/api/runs/{rid}/artifacts/screenshot.jpg").status_code == 200
    session = {"name": "alice", "exp": time.time() + 3600}
    client.cookies.set(auth.COOKIE, auth._box.encrypt(json.dumps(session).encode()).decode())
    assert client.get(f"/api/runs/{rid}/artifacts/trajectory.json").status_code == 200
    for name in ("run.json", "events.jsonl"):
        assert client.get(f"/api/runs/{rid}/artifacts/{name}").status_code == 404
    client.cookies.clear()
    store.update(rid, visibility="private")
    assert client.get(f"/api/runs/{rid}/artifacts/screenshot.jpg").status_code == 404


def test_oauth_callback_requires_initiating_browser_and_is_single_use(client, monkeypatch):
    calls = []

    def exchange(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(status_code=200, json=lambda: {"access_token": "test-token", "expires_in": 3600})

    monkeypatch.setattr(auth.httpx, "post", exchange)
    monkeypatch.setattr(auth.httpx, "get", lambda *args, **kwargs:
                        SimpleNamespace(json=lambda: {"preferred_username": "alice"}))
    response = client.get("/login", follow_redirects=False)
    state = parse_qs(urlparse(response.headers["location"]).query)["state"][0]
    encrypted = response.cookies[auth.STATE_COOKIE]
    assert state not in encrypted
    cookie = response.headers["set-cookie"].lower()
    assert all(flag in cookie for flag in ("httponly", "secure", "samesite=none", "path=/login", "max-age=600"))
    callback = f"/login/callback?code=code&state={state}"
    other = TestClient(main.app, base_url="https://testserver")
    assert other.get(callback).status_code == 400
    assert client.get("/login/callback?code=code&state=forged").status_code == 400
    assert not calls
    result = client.get(callback, follow_redirects=False)
    assert result.status_code == 307 and len(calls) == 1
    assert auth.COOKIE in client.cookies and auth.STATE_COOKIE not in client.cookies
    client.cookies.set(auth.STATE_COOKIE, encrypted, path="/login")
    assert client.get(callback).status_code == 400
    assert len(calls) == 1


def test_expired_state_cookie_cannot_complete_signin(client):
    auth._states["old"] = time.time()
    encrypted = auth._box.encrypt_at_time(b"old", int(time.time()) - auth.STATE_TTL - 10).decode()
    client.cookies.set(auth.STATE_COOKIE, encrypted, path="/login")
    assert client.get("/login/callback?code=code&state=old").status_code == 400


def upstream(monkeypatch, chunks):
    observed = {"chunks": 0, "response_closed": False, "client_closed": False}
    run = SimpleNamespace(run={}, stream=None, upstream=lambda model: (config.ROUTER, ""))
    monkeypatch.setattr(main.core, "by_cap", lambda cap: run)

    async def body():
        for chunk in chunks:
            observed["chunks"] += 1
            yield chunk

    async def close_response():
        observed["response_closed"] = True

    response = SimpleNamespace(status_code=200, headers={"content-type": "text/event-stream"},
                               aiter_bytes=body, aclose=close_response)

    async def send(*args, **kwargs):
        return response

    async def close_client():
        observed["client_closed"] = True

    monkeypatch.setattr(main.httpx, "AsyncClient", lambda **kwargs:
                        SimpleNamespace(build_request=lambda *a, **k: None, send=send, aclose=close_client))
    return observed, run


@pytest.mark.parametrize("mode", ["line", "total"])
def test_unbounded_upstream_is_stopped_and_closed(client, monkeypatch, mode):
    chunk = b"x" * 1023 + (b"\n" if mode == "total" else b"x")
    observed, run = upstream(monkeypatch, [chunk] * 100)
    monkeypatch.setattr(main, "LLM_REPLY_MAX", 8192)
    monkeypatch.setattr(main, "LLM_SSE_LINE_MAX", 4096)
    response = client.post("/api/llm/cap/v1/chat/completions", json={"model": "model"})
    expected = 8192 if mode == "total" else 4096
    assert len(response.content) == expected
    assert observed["chunks"] == expected // 1024 + 1
    assert observed["response_closed"] and observed["client_closed"] and run.stream is None


def test_normal_stream_passes_through_and_nonobject_input_is_rejected(client, monkeypatch):
    payload = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n'
    observed, run = upstream(monkeypatch, [payload[:12], payload[12:]])
    assert client.post("/api/llm/cap/v1/chat/completions", json=[]).status_code == 400
    response = client.post("/api/llm/cap/v1/chat/completions", json={"model": "model"})
    assert response.content == payload
    assert observed["response_closed"] and observed["client_closed"] and run.stream is None
