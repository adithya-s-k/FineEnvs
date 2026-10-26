"""Real HTTP review requests using an owned store and local OAuth test profiles."""
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


def test_concurrent_successful_replies_survive_reload(api):
    server, url, root = api

    async def requests():
        owner = await signed_client(url, "owner")
        reviewer = await signed_client(url, "reviewer")
        try:
            created = await owner.post("/api/review/threads", json={
                "anchor": {"type": "text", "quote": "shared paragraph"}, "body": "original",
            })
            assert created.status_code == 200
            tid = created.json()["id"]
            replies = await asyncio.gather(*[
                reviewer.post(f"/api/review/threads/{tid}/replies", json={"body": f"reply-{i}"})
                for i in range(20)
            ])
            assert all(r.status_code == 200 for r in replies), [r.status_code for r in replies]
            fetched = await owner.get("/api/review/threads")
            assert fetched.status_code == 200
            persisted = json.loads((root / "threads" / f"{tid}.json").read_text())
            expected = {"original"} | {f"reply-{i}" for i in range(20)}
            assert {m["body"] for m in fetched.json()[0]["messages"]} == expected
            assert {m["body"] for m in persisted["messages"]} == expected
            reloaded = server.Store(root)
            reloaded.load()
            assert {m["body"] for m in reloaded.get(tid)["messages"]} == expected
        finally:
            await owner.aclose()
            await reviewer.aclose()

    asyncio.run(requests())


def test_review_permissions_and_single_update_still_work(api):
    _, url, _ = api

    async def requests():
        owner = await signed_client(url, "owner")
        outsider = await signed_client(url, "outsider")
        try:
            result = await outsider.get("/api/review/threads")
            assert result.status_code == 403
            created = await owner.post("/api/review/threads", json={
                "anchor": {"type": "text", "quote": "paragraph"}, "body": "original",
            })
            assert created.status_code == 200
            tid = created.json()["id"]
            result = await owner.patch(f"/api/review/threads/{tid}", json={"status": "resolved"})
            assert result.status_code == 200
            assert result.json()["status"] == "resolved"
            owner.headers.pop("x-review")
            result = await owner.post(f"/api/review/threads/{tid}/replies", json={"body": "forged"})
            assert result.status_code == 403
        finally:
            await owner.aclose()
            await outsider.aclose()

    asyncio.run(requests())


def test_parallel_reactions_and_replies_keep_both_kinds_of_changes(api):
    _, url, root = api

    async def requests():
        owner = await signed_client(url, "owner")
        reviewer = await signed_client(url, "reviewer")
        try:
            created = await owner.post("/api/review/threads", json={
                "anchor": {"type": "text", "quote": "paragraph"}, "body": "original",
            })
            thread = created.json()
            tid, mid = thread["id"], thread["messages"][0]["id"]
            emojis = ["👍", "❤️", "😄", "🎉", "👀", "➕", "🔥"]
            results = await asyncio.gather(*[
                reviewer.post(f"/api/review/threads/{tid}/messages/{mid}/reactions", json={"emoji": emoji})
                for emoji in emojis
            ], *[
                reviewer.post(f"/api/review/threads/{tid}/replies", json={"body": f"reply-{i}"})
                for i in range(10)
            ])
            assert all(r.status_code == 200 for r in results), [r.status_code for r in results]
            thread = json.loads((root / "threads" / f"{tid}.json").read_text())
            assert len(thread["messages"]) == 11
            assert thread["messages"][0]["reactions"] == {emoji: ["reviewer"] for emoji in emojis}
        finally:
            await owner.aclose()
            await reviewer.aclose()

    asyncio.run(requests())


def test_edit_delete_and_owner_decision_permissions_are_preserved(api):
    _, url, _ = api

    async def requests():
        owner = await signed_client(url, "owner")
        reviewer = await signed_client(url, "reviewer")
        try:
            created = await owner.post("/api/review/threads", json={
                "anchor": {"type": "text", "quote": "paragraph"}, "body": "original",
                "suggestion": {"action": "replace", "text": "replacement"},
            })
            thread = created.json()
            tid, mid = thread["id"], thread["messages"][0]["id"]
            endpoint = f"/api/review/threads/{tid}"
            denied = await reviewer.patch(endpoint, json={"decision": "accepted"})
            assert denied.status_code == 403
            denied = await reviewer.patch(f"{endpoint}/messages/{mid}", json={"body": "not mine"})
            assert denied.status_code == 403
            denied = await reviewer.delete(f"{endpoint}/messages/{mid}")
            assert denied.status_code == 403
            accepted = await owner.patch(endpoint, json={"decision": "accepted"})
            assert accepted.status_code == 200
            assert accepted.json()["decision"] == "accepted"
            edited = await owner.patch(f"{endpoint}/messages/{mid}", json={"body": "edited"})
            assert edited.status_code == 200
            assert edited.json()["messages"][0]["body"] == "edited"
            assert edited.json()["decision"] == "accepted"
            deleted = await owner.delete(f"{endpoint}/messages/{mid}")
            assert deleted.status_code == 200
            assert deleted.json() == {"deleted": tid}
            assert (await owner.get("/api/review/threads")).json() == []
        finally:
            await owner.aclose()
            await reviewer.aclose()

    asyncio.run(requests())
