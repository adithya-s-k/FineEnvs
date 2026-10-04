"""What every page of both apps (the explorer and its admin) gets: security headers, a localhost-only rule when run
locally, same-origin state changes, and pages that revalidate so a deploy is never half-cached."""

from __future__ import annotations

import os
import threading
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config

LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"} | {h.strip() for h in os.environ.get("RLX_ALLOWED_HOSTS", "").split(",") if h.strip()}
CSP = ("default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
       "font-src https://fonts.gstatic.com https://cdn.jsdelivr.net; img-src 'self' data: blob: https:; media-src 'self' data: blob: https:; connect-src 'self'; frame-src 'self' https://*.hf.space; object-src 'none'; "
       "base-uri 'none'; form-action 'self'; frame-ancestors 'self' https://huggingface.co https://*.hf.space")


def client_ip(request: Request) -> str:
    """The visitor's address: the last X-Forwarded-For entry is the one the Space's proxy added (earlier ones are
    whatever the client claimed)."""
    fwd = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()] if config.TRUST_PROXY else []
    return fwd[-1] if fwd else (request.client.host if request.client else "?")


class Limiter:
    """At most `n` hits per `window` seconds for one key (an address, an account): a sliding window, in memory, bounded
    so a flood of distinct keys can't grow it without end. Per replica, which is what a Space runs."""

    def __init__(self, n: int, window: float, keys: int = 20000):
        self.n, self.window, self.keys = n, window, keys
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> bool:
        """Count one hit; False when `key` is over its limit."""
        now = time.time()
        with self._lock:
            hits = [t for t in self._hits.get(key, ()) if now - t < self.window]
            if len(hits) >= self.n:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            if len(self._hits) > self.keys:
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window] or list(self._hits)[: self.keys // 5]:
                    self._hits.pop(k, None)
            return True


def same_origin(request: Request) -> bool:
    """Compare complete origins; forwarding headers supplied by a client aren't authority."""
    origin = request.headers.get("origin")
    if not origin:
        return request.headers.get("sec-fetch-site") in (None, "same-origin", "none")

    def canonical(value: str):
        try:
            u = urlsplit(value)
            if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password or u.path or u.query or u.fragment:
                return None
            return u.scheme, u.hostname.lower(), u.port or (443 if u.scheme == "https" else 80)
        except ValueError:
            return None

    expected = canonical(config.PUBLIC_URL) if config.PUBLIC_URL else canonical(f"{request.url.scheme}://{request.headers.get('host', '')}")
    return expected is not None and canonical(origin) == expected


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def _revalidate(request: Request, call_next):
        """Code and styles revalidate on every load (cheap: ETag -> 304), so a deploy is never half-cached."""
        resp = await call_next(request)
        if request.url.path.startswith(("/api/", "/mcp/", "/login")):
            # APIs can contain a visitor's private dataset, run, or account. A shared
            # proxy or the browser's cache must never reuse them for another visitor.
            resp.headers["Cache-Control"] = "private, no-store"
            resp.headers["X-Robots-Tag"] = "noindex"
        else:
            resp.headers.setdefault("Cache-Control", "no-cache")
        return resp

    @app.middleware("http")
    async def _security(request: Request, call_next):
        """Locally, only answer to localhost: a web page could otherwise DNS-rebind its own name to 127.0.0.1 and drive
        this server (and the machine's HF token) from the browser. Everywhere: the usual security headers."""
        if config.LOCAL_MODE:
            raw = request.headers.get("host") or ""
            host = raw.split("]")[0] + "]" if raw.startswith("[") else raw.rsplit(":", 1)[0]
            if host not in LOCAL_HOSTS:
                return JSONResponse({"detail": "host not allowed (set RLX_ALLOWED_HOSTS to serve under another name)"}, 403)
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
        if not request.url.path.startswith("/api/"):
            resp.headers.setdefault("Content-Security-Policy", CSP)
        return resp


    @app.middleware("http")
    async def _same_origin_posts(request: Request, call_next):
        """The session cookie is SameSite=None (it has to work in the huggingface.co iframe), so a state-changing
        request must prove it came from this page: another site could otherwise start rollouts billed to you."""
        if request.method not in ("GET", "HEAD", "OPTIONS") and request.url.path.startswith("/api/"):
            if not same_origin(request):
                return JSONResponse({"detail": "cross-site request refused"}, 403)
        return await call_next(request)
