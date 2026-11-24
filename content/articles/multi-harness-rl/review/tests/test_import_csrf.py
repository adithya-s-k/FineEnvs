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


def test_cookie_import_requires_write_header_before_storage_changes(api):
    _, url, root = api

    async def requests():
        owner = await signed_client(url, "owner")
        try:
            owner.headers.pop("x-review")
            response = await owner.post("/api/review/import", content=json.dumps(backup()),
                                        headers={"content-type": "text/plain", "origin": "https://untrusted.example"})
            assert response.status_code == 403
            assert not (root / "threads" / "imported-1.json").exists()
            assert (await owner.get("/api/review/threads")).json() == []
        finally:
            await owner.aclose()

    asyncio.run(requests())


def test_guarded_owner_import_and_reviewer_rejection(api):
    _, url, _ = api

    async def requests():
        owner = await signed_client(url, "owner")
        reviewer = await signed_client(url, "reviewer")
        try:
            imported = await owner.post("/api/review/import", json=backup())
            assert imported.status_code == 200
            assert imported.json() == {"added": 1, "skipped": 0, "total": 1}
            skipped = await owner.post("/api/review/import", json=backup())
            assert skipped.json() == {"added": 0, "skipped": 1, "total": 1}
            assert (await owner.get("/api/review/export")).status_code == 200
            denied = await reviewer.post("/api/review/import", json=backup("reviewer-write"))
            assert denied.status_code == 401
            assert len((await owner.get("/api/review/threads")).json()) == 1
        finally:
            await owner.aclose()
            await reviewer.aclose()

    asyncio.run(requests())


def test_bearer_owner_import_still_works_without_cookie_or_write_header(api, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from huggingface_hub import HfApi

    server, url, root = api

    class IdentityProvider(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/api/whoami-v2"
            name = "owner" if self.headers.get("authorization") == "Bearer fake-owner-token" else "outsider"
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"name": name}).encode())

        def log_message(self, *args):
            pass

    provider = ThreadingHTTPServer(("127.0.0.1", 0), IdentityProvider)
    worker = threading.Thread(target=provider.serve_forever, daemon=True)
    worker.start()
    # Configure the real Hub client to an owned external identity endpoint.
    endpoint = f"http://127.0.0.1:{provider.server_address[1]}"
    monkeypatch.setattr(server, "HfApi", lambda: HfApi(endpoint=endpoint))
    try:
        with httpx.Client(base_url=url, timeout=5) as client:
            imported = client.post("/api/review/import", json=backup("token-backup"),
                                   headers={"authorization": "Bearer fake-owner-token"})
            assert imported.status_code == 200
            assert imported.json()["added"] == 1
            denied = client.post("/api/review/import", json=backup("wrong-token"),
                                 headers={"authorization": "Bearer fake-outsider-token"})
            assert denied.status_code == 403
            assert not (root / "threads" / "wrong-token.json").exists()
    finally:
        provider.shutdown()
        provider.server_close()
        worker.join(timeout=5)


def test_published_mode_stays_closed(api):
    server, url, _ = api
    server.REVIEW_MODE = False
    with httpx.Client(base_url=url, timeout=5) as client:
        result = client.post("/api/review/import", json=backup())
        assert result.status_code == 404
