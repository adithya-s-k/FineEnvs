"""Review comments for the article, stored on a storage bucket mounted in the Space.

The article is a static site served by nginx. This small API runs next to it, behind nginx at
/api/review/, and adds one thing: reviewers can sign in with Hugging Face, select text or a figure,
and leave a comment. Each thread is one JSON file under COMMENTS_DIR, which defaults to a folder on the
storage bucket mounted at /data, so comments survive restarts and redeploys.

Two modes, set with the REVIEW_MODE Space variable:
- off (default): the published article. Every review endpoint answers 404 and the page never loads
  the review layer.
- on: people who open the article with ?review can sign in and comment, if they are on the reviewer
  list (REVIEWERS) or own the Space.

Configuration (Space variables):
    REVIEW_MODE    "on" to accept comments, anything else to stay published
    REVIEWERS      comma-separated Hugging Face usernames allowed to read and write comments
    COMMENTS_DIR   folder for the threads, default /data/review-comments (the mounted bucket)

The Space owner can also export and import every thread with their Hugging Face token in an
Authorization header (see review/pull_comments.py). The token is only used to check who is calling.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from huggingface_hub import HfApi, attach_huggingface_oauth, parse_huggingface_oauth

REVIEW_MODE = os.environ.get("REVIEW_MODE", "off").strip().lower() == "on"
OWNER = (os.environ.get("REVIEW_OWNER") or os.environ.get("SPACE_AUTHOR_NAME") or "").strip()
REVIEWERS = {u.strip().lower() for u in os.environ.get("REVIEWERS", "").split(",") if u.strip()}
if OWNER:
    REVIEWERS.add(OWNER.lower())
COMMENTS_DIR = Path(os.environ.get("COMMENTS_DIR", "").strip() or "/data/review-comments")

MAX_BODY = 5000
MAX_QUOTE = 1200
WRITES_PER_MINUTE = 30

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
try:
    attach_huggingface_oauth(app)
except ValueError as e:
    # In a Space, sign-in needs `hf_oauth: true` in the README metadata. Without it, stay published.
    if REVIEW_MODE:
        print(f"[review] review mode is off: {e} Add `hf_oauth: true` to the README metadata.", flush=True)
    REVIEW_MODE = False


# ---------------------------------------------------------------- storage


class Store:
    """All threads in memory, written through to one JSON file each under COMMENTS_DIR."""

    def __init__(self, root: Path) -> None:
        self.root = root / "threads"
        self.lock = threading.Lock()
        self.threads: dict[str, dict[str, Any]] = {}
        self.error = ""

    def ready(self) -> bool:
        """Create the folder if we can, and refuse a /data that is not a mounted volume."""
        # Without a mounted bucket a root container can still create /data, but on the temporary
        # disk, where every comment would vanish at the next restart.
        if str(self.root).startswith("/data/") and not os.path.ismount("/data") and os.environ.get("ALLOW_EPHEMERAL") != "1":
            self.error = "No storage bucket is mounted at /data on this Space, so comments cannot be saved yet."
            return False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            probe = self.root / ".write-test"
            probe.write_text("ok")
            probe.unlink()
            self.error = ""
            return True
        except OSError as e:
            self.error = (f"Cannot write comments to {self.root} ({e.strerror}). "
                          "Mount a storage bucket at /data (read & write) in the Space settings.")
            return False

    def load(self) -> None:
        threads: dict[str, dict[str, Any]] = {}
        if self.ready():
            for f in self.root.glob("*.json"):
                try:
                    t = json.loads(f.read_text())
                    threads[t["id"]] = t
                except (OSError, ValueError, KeyError) as e:
                    print(f"[review] skipping {f.name}: {e}", flush=True)
        with self.lock:
            self.threads = threads

    def save(self, thread: dict[str, Any]) -> None:
        if not self.ready():
            raise HTTPException(503, self.error)
        # Write to a temporary file and rename, so a crash never leaves half a thread on disk.
        path = self.root / f"{thread['id']}.json"
        body = json.dumps(thread, ensure_ascii=False, indent=1)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(body)
            os.replace(tmp, path)
        except OSError:
            # A mounted bucket may not support rename; write the file in place instead.
            tmp.unlink(missing_ok=True)
            path.write_text(body)
        with self.lock:
            self.threads[thread["id"]] = thread

    def delete(self, tid: str) -> None:
        (self.root / f"{tid}.json").unlink(missing_ok=True)
        with self.lock:
            self.threads.pop(tid, None)

    def all(self) -> list[dict[str, Any]]:
        with self.lock:
            return sorted(self.threads.values(), key=lambda t: t["created_at"])

    def get(self, tid: str) -> dict[str, Any]:
        with self.lock:
            t = self.threads.get(tid)
        if t is None:
            raise HTTPException(404, "Thread not found.")
        return json.loads(json.dumps(t))


store = Store(COMMENTS_DIR)


@app.on_event("startup")
def _startup() -> None:
    if REVIEW_MODE:
        try:
            store.load()
        except Exception as e:  # the page still works, the panel shows the error
            print(f"[review] could not load comments: {e}", flush=True)


# ---------------------------------------------------------------- auth


def _user(request: Request) -> dict[str, str] | None:
    info = parse_huggingface_oauth(request)
    if info is None:
        return None
    u = info.user_info
    return {"username": u.preferred_username, "name": u.name or u.preferred_username, "avatar": u.picture or ""}


def _require_reviewer(request: Request) -> dict[str, str]:
    if not REVIEW_MODE:
        raise HTTPException(404)
    user = _user(request)
    if user is None:
        raise HTTPException(401, "Sign in with Hugging Face to review.")
    if user["username"].lower() not in REVIEWERS:
        raise HTTPException(403, "This account is not on the reviewer list.")
    return user


_writes: dict[str, list[float]] = {}


def _guard_write(request: Request, user: dict[str, str]) -> None:
    # The session cookie is SameSite=None so sign-in works inside the Space iframe. A custom header
    # can't be sent cross-site without a CORS preflight, which this API never grants, so requiring
    # it blocks forged posts from other sites.
    if request.headers.get("x-review") != "1":
        raise HTTPException(403, "Missing review header.")
    now = time.time()
    recent = [t for t in _writes.get(user["username"], []) if now - t < 60]
    if len(recent) >= WRITES_PER_MINUTE:
        raise HTTPException(429, "Too many comments in a minute. Try again shortly.")
    _writes[user["username"]] = recent + [now]


def _clean(text: Any, limit: int) -> str:
    s = str(text or "").replace("\r\n", "\n").strip()
    if not s:
        raise HTTPException(400, "Empty text.")
    return s[:limit]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- routes


@app.get("/api/review/status")
def status(request: Request) -> JSONResponse:
    if not REVIEW_MODE:
        return JSONResponse({"enabled": False})
    user = _user(request)
    return JSONResponse({
        "enabled": True,
        "user": user,
        "reviewer": bool(user and user["username"].lower() in REVIEWERS),
        "owner": OWNER,
        "storage": store.error or "ok",
    })


@app.get("/api/review/threads")
def list_threads(request: Request) -> list[dict[str, Any]]:
    _require_reviewer(request)
    if store.error and not store.ready():
        raise HTTPException(503, store.error)
    return store.all()


@app.post("/api/review/threads")
async def create_thread(request: Request) -> dict[str, Any]:
    user = _require_reviewer(request)
    _guard_write(request, user)
    payload = await request.json()
    a = payload.get("anchor") or {}
    kind = a.get("type")
    if kind == "text":
        anchor = {"type": "text", "quote": _clean(a.get("quote"), MAX_QUOTE),
                  "prefix": str(a.get("prefix") or "")[-80:], "suffix": str(a.get("suffix") or "")[:80]}
    elif kind == "figure":
        anchor = {"type": "figure", "figure": _clean(a.get("figure"), 200), "title": str(a.get("title") or "")[:300]}
    else:
        raise HTTPException(400, "Unknown anchor type.")
    anchor["section"] = str(a.get("section") or "")[:200]
    anchor["chapter"] = str(a.get("chapter") or "")[:200]
    # A suggestion proposes an edit to the selected text, as in Google Docs' suggesting mode: delete
    # it, or replace it with new text. The owner accepts or rejects it; accepted ones are applied to
    # the article source afterwards.
    suggestion = None
    sug = payload.get("suggestion")
    if sug:
        if kind != "text":
            raise HTTPException(400, "Suggestions need selected text.")
        action = sug.get("action")
        if action == "delete":
            suggestion = {"action": "delete"}
        elif action == "replace":
            suggestion = {"action": "replace", "text": _clean(sug.get("text"), MAX_QUOTE)}
        else:
            raise HTTPException(400, "Unknown suggestion.")
    body = str(payload.get("body") or "").replace("\r\n", "\n").strip()[:MAX_BODY]
    if not body and not suggestion:
        raise HTTPException(400, "Empty text.")
    thread = {
        "id": time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3),
        "anchor": anchor,
        "status": "open",
        "created_at": _now(),
        "messages": [{"id": secrets.token_hex(4), "author": user, "body": body, "created_at": _now()}],
    }
    if suggestion:
        thread["kind"] = "suggestion"
        thread["suggestion"] = suggestion
    await run_in_threadpool(store.save, thread)
    return thread


@app.post("/api/review/threads/{tid}/replies")
async def reply(tid: str, request: Request) -> dict[str, Any]:
    user = _require_reviewer(request)
    _guard_write(request, user)
    payload = await request.json()
    thread = store.get(tid)
    thread["messages"].append({"id": secrets.token_hex(4), "author": user, "body": _clean(payload.get("body"), MAX_BODY), "created_at": _now()})
    await run_in_threadpool(store.save, thread)
    return thread


@app.patch("/api/review/threads/{tid}")
async def update_thread(tid: str, request: Request) -> dict[str, Any]:
    user = _require_reviewer(request)
    _guard_write(request, user)
    payload = await request.json()
    thread = store.get(tid)
    if payload.get("status") in ("open", "resolved"):
        thread["status"] = payload["status"]
        thread["resolved_by"] = user["username"] if payload["status"] == "resolved" else None
        if payload["status"] == "open":
            thread.pop("decision", None)
        thread["updated_at"] = _now()
    if payload.get("decision") in ("accepted", "rejected"):
        if thread.get("kind") != "suggestion":
            raise HTTPException(400, "Only suggestions are accepted or rejected.")
        if user["username"].lower() != OWNER.lower():
            raise HTTPException(403, "Only the article's owner can accept or reject suggestions.")
        thread["decision"] = payload["decision"]
        thread["status"] = "resolved"
        thread["resolved_by"] = user["username"]
        thread["updated_at"] = _now()
    await run_in_threadpool(store.save, thread)
    return thread


REACTIONS = {"👍", "❤️", "😄", "🎉", "👀", "➕", "🔥"}


def _own_or_owner(msg: dict[str, Any], user: dict[str, str]) -> bool:
    return msg["author"]["username"] == user["username"] or user["username"].lower() == OWNER.lower()


@app.patch("/api/review/threads/{tid}/messages/{mid}")
async def edit_message(tid: str, mid: str, request: Request) -> dict[str, Any]:
    user = _require_reviewer(request)
    _guard_write(request, user)
    payload = await request.json()
    thread = store.get(tid)
    msg = next((m for m in thread["messages"] if m["id"] == mid), None)
    if msg is None:
        raise HTTPException(404, "Message not found.")
    if msg["author"]["username"] != user["username"]:
        raise HTTPException(403, "You can only edit your own comments.")
    msg["body"] = _clean(payload.get("body"), MAX_BODY)
    msg["edited_at"] = _now()
    await run_in_threadpool(store.save, thread)
    return thread


@app.post("/api/review/threads/{tid}/messages/{mid}/reactions")
async def react(tid: str, mid: str, request: Request) -> dict[str, Any]:
    """Toggle one emoji reaction by the signed-in reviewer."""
    user = _require_reviewer(request)
    _guard_write(request, user)
    emoji = str((await request.json()).get("emoji") or "")
    if emoji not in REACTIONS:
        raise HTTPException(400, "Unknown reaction.")
    thread = store.get(tid)
    msg = next((m for m in thread["messages"] if m["id"] == mid), None)
    if msg is None:
        raise HTTPException(404, "Message not found.")
    who = msg.setdefault("reactions", {}).setdefault(emoji, [])
    if user["username"] in who:
        who.remove(user["username"])
    else:
        who.append(user["username"])
    if not who:
        msg["reactions"].pop(emoji)
    await run_in_threadpool(store.save, thread)
    return thread


@app.delete("/api/review/threads/{tid}/messages/{mid}")
async def delete_message(tid: str, mid: str, request: Request) -> dict[str, Any]:
    user = _require_reviewer(request)
    _guard_write(request, user)
    thread = store.get(tid)
    msg = next((m for m in thread["messages"] if m["id"] == mid), None)
    if msg is None:
        raise HTTPException(404, "Message not found.")
    if not _own_or_owner(msg, user):
        raise HTTPException(403, "You can only delete your own comments.")
    thread["messages"] = [m for m in thread["messages"] if m["id"] != mid]
    if not thread["messages"]:
        await run_in_threadpool(store.delete, tid)
        return {"deleted": tid}
    await run_in_threadpool(store.save, thread)
    return thread


# ---------------------------------------------------------------- owner: export and import

_token_owner: dict[str, tuple[bool, float]] = {}


def _require_owner(request: Request) -> None:
    """The Space owner, signed in on the page or presenting their Hugging Face token."""
    if not REVIEW_MODE:
        raise HTTPException(404)
    user = _user(request)
    if user and OWNER and user["username"].lower() == OWNER.lower():
        return
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer ") or not OWNER:
        raise HTTPException(401, "Owner only.")
    token = auth[7:].strip()
    key = hashlib.sha256(token.encode()).hexdigest()
    ok, at = _token_owner.get(key, (False, 0.0))
    if time.time() - at > 300:  # ask the Hub who this is, at most every five minutes per token
        try:
            ok = HfApi().whoami(token=token).get("name", "").lower() == OWNER.lower()
        except Exception:
            ok = False
        _token_owner[key] = (ok, time.time())
    if not ok:
        raise HTTPException(403, "Owner only.")


@app.get("/api/review/export")
def export_threads(request: Request) -> list[dict[str, Any]]:
    _require_owner(request)
    return store.all()


@app.post("/api/review/import")
async def import_threads(request: Request) -> dict[str, Any]:
    """Add threads from an export or another source. Existing ids are kept unless replace is true."""
    _require_owner(request)
    payload = await request.json()
    items = payload.get("threads") or []
    replace = bool(payload.get("replace"))
    added, skipped = 0, 0
    for t in items:
        if not isinstance(t, dict) or not {"id", "anchor", "messages", "created_at"} <= t.keys():
            raise HTTPException(400, "Each thread needs id, anchor, messages and created_at.")
        tid = str(t["id"])
        if not tid.replace("-", "").replace("_", "").isalnum() or len(tid) > 80:
            raise HTTPException(400, f"Bad thread id: {tid!r}")
        with store.lock:
            exists = tid in store.threads
        if exists and not replace:
            skipped += 1
            continue
        t.setdefault("status", "open")
        await run_in_threadpool(store.save, t)
        added += 1
    return {"added": added, "skipped": skipped, "total": len(store.all())}
