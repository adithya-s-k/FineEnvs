"""Sign in with Hugging Face (server-side OAuth code flow, as ML Intern does it).

The access token lives in a signed, HttpOnly cookie. It is read per request to start the user's
own sandbox and call Inference Providers as them, and it is never written to disk or into a trace.

Locally there is no OAuth app: your own token (HF_TOKEN or `hf auth login`) is the user.
"""

from __future__ import annotations

import base64
import secrets
import time
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, URLSafeTimedSerializer

from . import config

router = APIRouter()
COOKIE = "mimo_session"
_signer = URLSafeTimedSerializer(config.SESSION_SECRET, salt="mimo-explorer")
_states: dict[str, float] = {}
_local: dict = {}


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
    _local.update(token=token, name=me["name"], avatar=me.get("avatarUrl"), local=True)
    return _local


def current_user(request: Request) -> dict | None:
    if config.LOCAL_MODE:
        return _local_user()
    raw = request.cookies.get(COOKIE)
    if not raw:
        return None
    try:
        data = _signer.loads(raw, max_age=config.SESSION_DAYS * 86400)
    except BadSignature:
        return None
    if data.get("exp", 0) < time.time():
        return None
    return data


def require_user(request: Request) -> dict:
    u = current_user(request)
    if not u:
        raise HTTPException(401, "Sign in with Hugging Face to run rollouts.")
    return u


def public(u: dict | None) -> dict | None:
    return None if not u else {"name": u["name"], "avatar": u.get("avatar"), "local": bool(u.get("local"))}


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
    missing = {"inference-api", "jobs"} - granted
    info = httpx.get(f"{config.OPENID_PROVIDER_URL}/oauth/userinfo", timeout=20,
                     headers={"Authorization": f"Bearer {tok['access_token']}"}).json()
    session = {"token": tok["access_token"], "name": info.get("preferred_username") or info.get("name"),
               "avatar": info.get("picture"), "exp": time.time() + min(tok.get("expires_in") or 28800, config.SESSION_DAYS * 86400),
               "missing_scopes": sorted(missing)}
    resp = RedirectResponse("/#/")
    # SameSite=None so it also works inside the huggingface.co iframe
    resp.set_cookie(COOKIE, _signer.dumps(session), httponly=True, secure=True, samesite="none",
                    max_age=config.SESSION_DAYS * 86400)
    return resp


@router.get("/logout")
def logout():
    resp = RedirectResponse("/")
    resp.delete_cookie(COOKIE, samesite="none", secure=True)
    return resp
