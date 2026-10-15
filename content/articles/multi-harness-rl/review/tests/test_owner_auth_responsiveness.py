"""Actual HTTP import requests with owned storage and local OAuth test profiles."""
import asyncio
import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import Request

SERVER = Path(__file__).resolve().parents[1] / "server.py"


@pytest.fixture
def api(tmp_path, monkeypatch):
    import huggingface_hub._oauth as oauth

    monkeypatch.delenv("SPACE_ID", raising=False)
    monkeypatch.setenv("REVIEW_MODE", "on")
    monkeypatch.setenv("REVIEW_OWNER", "owner")
    monkeypatch.setenv("REVIEWERS", "reviewer")
    monkeypatch.setenv("COMMENTS_DIR", str(tmp_path))
    # Only replace the external identity provider's local fixture. Actual session
    # middleware, signed cookies, OAuth parsing and all role checks remain active.
    monkeypatch.setattr(oauth, "_get_mocked_oauth_info", lambda: {
        "userinfo": {"preferred_username": "owner", "name": "Owner"},
        "expires_at": time.time() + 3600,
    })
    spec = importlib.util.spec_from_file_location("article_review_test", SERVER)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)

    @server.app.get("/__test_login/{username}")
    async def login(username: str, request: Request):
        request.session["oauth_info"] = {
            "userinfo": {"preferred_username": username, "name": username},
            "expires_at": time.time() + 3600,
        }
        return {"username": username}

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(server.app, log_level="error", lifespan="on")
    runner = uvicorn.Server(config)
    thread = threading.Thread(target=runner.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    end = time.monotonic() + 5
    while not runner.started and thread.is_alive() and time.monotonic() < end:
        time.sleep(0.01)
    assert runner.started
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    yield server, url, tmp_path
    runner.should_exit = True
    thread.join(timeout=5)
    sock.close()
    assert not thread.is_alive()


async def signed_client(url, username):
    client = httpx.AsyncClient(base_url=url, headers={"x-review": "1"}, timeout=10)
    login = await client.get(f"/__test_login/{username}")
    assert login.status_code == 200
    # SessionMiddleware correctly marks OAuth cookies Secure. For our loopback HTTP
    # fixture only, put the actual signed value in the Cookie header unchanged.
    client.headers["cookie"] = "session=" + client.cookies.get("session")
    return client



def backup(tid="imported-1"):
    return {"threads": [{"id": tid, "anchor": {"type": "text", "quote": "paragraph"},
                         "messages": [], "created_at": "2026-10-01T00:00:00Z"}]}


@pytest.mark.parametrize("identity, expected", [("owner", 200), ("outsider", 403)])
def test_hub_identity_wait_does_not_block_other_review_requests(api, monkeypatch, identity, expected):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from huggingface_hub import HfApi

    server, url, _ = api
    entered = threading.Event()
    release = threading.Event()
    lookups = []

    class IdentityProvider(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/api/whoami-v2"
            lookups.append(self.headers.get("authorization"))
            entered.set()
            release.wait(timeout=5)
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"name": identity}).encode())

        def log_message(self, *args):
            pass

    provider = ThreadingHTTPServer(("127.0.0.1", 0), IdentityProvider)
    worker = threading.Thread(target=provider.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{provider.server_address[1]}"
    # Exercise the real SDK against a controlled external identity service.
    # Only its endpoint is changed: no owner helper or event-loop function is mocked.
    monkeypatch.setattr(server, "HfApi", lambda: HfApi(endpoint=endpoint))
    imported = []
    errors = []

    def import_backup():
        try:
            with httpx.Client(base_url=url, timeout=10) as client:
                imported.append(client.post("/api/review/import", json=backup(),
                                            headers={"authorization": "Bearer fake-test-token"}))
        except Exception as error:
            errors.append(error)

    request_thread = threading.Thread(target=import_backup, daemon=True)
    request_thread.start()
    try:
        assert entered.wait(timeout=5), "the actual SDK never reached the identity service"
        # The Hub response remains pending until after this request. A healthy
        # event loop can still answer status while authentication waits elsewhere.
        with httpx.Client(base_url=url, timeout=0.5) as client:
            status = client.get("/api/review/status")
            assert status.status_code == 200
            assert status.json()["enabled"] is True
    finally:
        release.set()
        request_thread.join(timeout=10)
        provider.shutdown()
        provider.server_close()
        worker.join(timeout=5)
    assert not request_thread.is_alive()
    assert not errors
    assert imported[0].status_code == expected
    # Successful and rejected tokens retain the existing five-minute cache.
    with httpx.Client(base_url=url, timeout=5) as client:
        repeated = client.post("/api/review/import", json=backup("second"),
                               headers={"authorization": "Bearer fake-test-token"})
        assert repeated.status_code == expected
    assert lookups == ["Bearer fake-test-token"]
