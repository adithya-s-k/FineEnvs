"""The HF RL Explorer admin: its own app, deployed as its own private Space, for members of the admin organization.

    uv run uvicorn app.admin_app:app --port 8061      # locally, next to the explorer on 8060

It mounts the same bucket as the explorer (/data on the Space) and shares nothing else with it: no process, no
cookies (another origin), no rollout threads. What it changes goes into the bucket's settings file, which the
explorer re-reads (see app/admin.py). The Space being private already limits it to the organization; every API
route checks membership again.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import admin, auth, config, http, settings, version

# where the public explorer is, for links to its datasets, Spaces and rollouts
EXPLORER = os.environ.get("RLX_EXPLORER_URL", "http://localhost:8060" if config.LOCAL_MODE else "https://fineenvs-rl-explorer.hf.space").rstrip("/")

app = FastAPI(title="HF RL Explorer admin", docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(auth.router)
app.include_router(admin.router)
app.add_middleware(GZipMiddleware, minimum_size=2048, compresslevel=5)
http.install(app)
ADMIN_WEB = config.ROOT / "web-admin"


@app.on_event("startup")
def _startup() -> None:
    import threading

    from . import catalog

    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # the Hub listing the dashboard opens with: fetched now, so the first visit doesn't wait on it
    threading.Thread(target=lambda: catalog.environments(include_hidden=True), daemon=True, name="warm").start()


@app.get("/healthz", include_in_schema=False)
def healthz():
    """Alive: the process answers."""
    return {"ok": True}


@app.get("/readyz", include_in_schema=False)
def readyz():
    """Ready: the bucket it keeps the settings in is mounted and writable (this app runs no rollouts or indexes)."""
    from fastapi.responses import JSONResponse

    try:
        settings.DIR.mkdir(parents=True, exist_ok=True)
        (settings.DIR / ".ready").write_text("ok")
        store = "ok"
    except OSError as e:
        store = f"unwritable: {type(e).__name__}"
    return JSONResponse({"ok": store == "ok", "store": store}, status_code=200 if store == "ok" else 503)


@app.get("/api/me")
def me(request: Request):
    u = auth.current_user(request)
    return {"user": auth.public(u), "local": config.LOCAL_MODE, "oauth": not config.LOCAL_MODE, "is_admin": settings.is_admin(u),
            "admin_org": settings.ADMIN_ORG, "explorer": EXPLORER, "version": version.app_version(), "source": version.source_hash()}


@app.get("/")
def index():
    return FileResponse(ADMIN_WEB / "index.html")


app.mount("/admin", StaticFiles(directory=ADMIN_WEB), name="admin-web")
app.mount("/", StaticFiles(directory=config.WEB_DIR), name="web")   # the explorer's stylesheets, icons and helpers
