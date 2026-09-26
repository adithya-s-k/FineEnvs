"""Who the user is: Sign in with Hugging Face (OAuth), or an access token they paste.

Either way the session is one encrypted, HttpOnly cookie. The token inside is read per request to start
the user's own sandbox and call Inference Providers as them. It is never written to disk or into a trace,
and the cookie is encrypted rather than only signed, so the token cannot be read back out of it.

Locally there is no OAuth app: your own token (HF_TOKEN or `hf auth login`) is the user, unless you
paste a different one.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from . import config

router = APIRouter()
COOKIE = "mimo_session"
_box = Fernet(base64.urlsafe_b64encode(hashlib.sha256(f"mimo-explorer:{config.SESSION_SECRET}".encode()).digest()))
_states: dict[str, float] = {}
_local: dict = {}
TOKENS_URL = "https://huggingface.co/settings/tokens"


def _redirect_uri(request: Request) -> str:
    host = config.SPACE_HOST or request.url.netloc
    scheme = "https" if config.SPACE_HOST else request.url.scheme
    return f"{scheme}://{host}/login/callback"


def _local_user() -> dict | None:
    if _local:
        return _local
    from huggingface_hub import get_token, whoami

    token = get_token()
    if not token:
        return None
    try:
        me = whoami(token=token)
    except Exception:
        return None
    _local.update(token=token, name=me["name"], avatar=me.get("avatarUrl"), local=True, via="local")
    return _local


def _cookie_user(request: Request) -> dict | None:
    raw = request.cookies.get(COOKIE)
    if not raw:
        return None
    try:
        data = json.loads(_box.decrypt(raw.encode(), ttl=config.SESSION_DAYS * 86400))
    except (InvalidToken, ValueError):
        return None
    return data if data.get("exp", 0) > time.time() else None


def current_user(request: Request) -> dict | None:
    return _cookie_user(request) or (_local_user() if config.LOCAL_MODE else None)


def require_user(request: Request) -> dict:
    u = current_user(request)
    if not u:
        raise HTTPException(401, "Sign in with Hugging Face to run rollouts.")
    return u


def public(u: dict | None) -> dict | None:
    if not u:
        return None
    return {"name": u["name"], "avatar": u.get("avatar"), "local": bool(u.get("local")), "via": u.get("via", "oauth")}


def _issue(resp, session: dict):
    # SameSite=None so it also works inside the huggingface.co iframe. Browsers accept Secure on http://localhost.
    resp.set_cookie(COOKIE, _box.encrypt(json.dumps(session).encode()).decode(), httponly=True, secure=True,
                    samesite="none", max_age=config.SESSION_DAYS * 86400, path="/")
    return resp


# ── OAuth ────────────────────────────────────────────────────────────────────
@router.get("/login")
def login(request: Request):
    if config.LOCAL_MODE:
        return RedirectResponse("/")
    now = time.time()
    for s, t in list(_states.items()):
        if now - t > 600:
            _states.pop(s, None)
    state = secrets.token_urlsafe(24)
    _states[state] = now
    q = urlencode({"client_id": config.OAUTH_CLIENT_ID, "redirect_uri": _redirect_uri(request), "response_type": "code",
                   "scope": " ".join(config.OAUTH_SCOPES), "state": state})
    return RedirectResponse(f"{config.OPENID_PROVIDER_URL}/oauth/authorize?{q}")


@router.get("/login/callback")
def callback(request: Request, code: str = "", state: str = ""):
    if state not in _states:
        raise HTTPException(400, "Sign-in expired or was tampered with. Try again.")
    _states.pop(state, None)
    basic = base64.b64encode(f"{config.OAUTH_CLIENT_ID}:{config.OAUTH_CLIENT_SECRET}".encode()).decode()
    r = httpx.post(f"{config.OPENID_PROVIDER_URL}/oauth/token", timeout=20,
                   headers={"Authorization": f"Basic {basic}"},
                   data={"grant_type": "authorization_code", "code": code, "redirect_uri": _redirect_uri(request),
                         "client_id": config.OAUTH_CLIENT_ID})
    if r.status_code != 200:
        raise HTTPException(400, f"Could not complete sign-in ({r.status_code}).")
    tok = r.json()
    granted = set((tok.get("scope") or "").split())
    info = httpx.get(f"{config.OPENID_PROVIDER_URL}/oauth/userinfo", timeout=20,
                     headers={"Authorization": f"Bearer {tok['access_token']}"}).json()
    return _issue(RedirectResponse("/#/"), {
        "token": tok["access_token"], "name": info.get("preferred_username") or info.get("name"), "avatar": info.get("picture"),
        "exp": time.time() + min(tok.get("expires_in") or 28800, config.SESSION_DAYS * 86400), "via": "oauth",
        "missing_scopes": sorted({"inference-api", "jobs"} - granted)})


# ── access token ─────────────────────────────────────────────────────────────
class TokenLogin(BaseModel):
    token: str


def _token_gaps(access: dict) -> list[str]:
    """What a token can't do that a rollout needs. Write tokens can do everything; read tokens can't start
    sandboxes; fine-grained tokens are checked against their global permissions (a warning, not a refusal)."""
    role = access.get("role")
    if role == "write":
        return []
    if role == "read":
        return ["jobs"]
    perms = " ".join((access.get("fineGrained") or {}).get("global") or [])
    return [n for n, key in (("inference-api", "inference"), ("jobs", "job")) if key not in perms]


@router.post("/api/login/token")
def token_login(body: TokenLogin):
    token = body.token.strip()
    if not token.startswith("hf_") or len(token) < 20 or any(c.isspace() for c in token):
        raise HTTPException(400, "That doesn't look like a Hugging Face token. They start with hf_.")
    r = httpx.get(f"{config.OPENID_PROVIDER_URL}/api/whoami-v2", headers={"Authorization": f"Bearer {token}"}, timeout=20)
    if r.status_code == 401:
        raise HTTPException(400, "Hugging Face didn't accept this token. It may have been revoked or mistyped.")
    if r.status_code != 200:
        raise HTTPException(502, f"Couldn't check the token with Hugging Face ({r.status_code}). Try again.")
    me = r.json()
    if me.get("type") != "user":
        raise HTTPException(400, "Use a token that belongs to your user account, not an organization token.")
    gaps = _token_gaps(((me.get("auth") or {}).get("accessToken")) or {})
    if gaps == ["jobs"] and (me["auth"]["accessToken"].get("role") == "read"):
        raise HTTPException(400, "This is a read token, and rollouts need to start an HF Sandbox. Use a write token, or a "
                                 "fine-grained one with the Inference Providers and Jobs permissions.")
    resp = JSONResponse({"user": {"name": me["name"], "avatar": me.get("avatarUrl"), "local": False, "via": "token"},
                         "missing_scopes": gaps})
    return _issue(resp, {"token": token, "name": me["name"], "avatar": me.get("avatarUrl"), "via": "token",
                         "exp": time.time() + config.SESSION_DAYS * 86400, "missing_scopes": gaps})


@router.get("/logout")
def logout():
    resp = RedirectResponse("/")
    resp.delete_cookie(COOKIE, samesite="none", secure=True, path="/")
    return resp


@router.post("/api/logout")
def logout_api():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, samesite="none", secure=True, path="/")
    return resp
