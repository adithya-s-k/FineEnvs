"""The eval explorer (the FineEnvs/PortSimEnv-Eval Space): the viewer and its read-only JSON, without an environment.

Overview with the eval table, tasks, task pages and model rollouts replayed in 3D, from the task packs
(BERTH_TASKS_DIR) and the rollouts under BERTH_RUNS_DIR. No OpenEnv sessions, no /ws, no live episodes; the viewer
runs in explorer mode, so playing an episode links to the environment Space (BERTH_PLAY_URL).

    BERTH_RUNS_DIR=results/rollouts uvicorn berth_openenv.explorer:app --port 8000   ->  /viewer/
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from berth_core import load_pack

try:
    from .api import mount_viewer
except ImportError:
    from api import mount_viewer


def create_app() -> FastAPI:
    pack = load_pack()
    app = FastAPI(title="PortSimEnv v1 eval", description="Read-only eval explorer for PortSimEnv v1.")
    mount_viewer(app, pack, mode=os.environ.get("BERTH_VIEWER_MODE") or "explorer")

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok", "tasks": {s: pack.count(s) for s in pack.splits()}}

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/viewer/")

    return app


app = create_app()
