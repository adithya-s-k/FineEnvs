"""The server: OpenEnv's app, plus the editor and its API on the same port.

OpenEnv provides `/reset`, `/step`, `/state`, `/schema`, `/ws`, `/mcp`, `/health`,
the Task API under `/design_canvas/...` and the Gradio UI at `/web`. We add the
browser editor (`/editor/`) and the routes it uses (`/api/...`, `/fonts/...`).
One process, one port, one episode store behind all of them.

    uvicorn designgym.server.app:app --port 8000
"""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

try:
    from ..core.pack import load_pack
    from .editor_api import mount_editor
    from .environment import DesignCanvasEnvironment
    from .gradio_ui import build_ui
except ImportError:
    from core.pack import load_pack
    from server.editor_api import mount_editor
    from server.environment import DesignCanvasEnvironment
    from server.gradio_ui import build_ui

ENV_NAME = "design_canvas"  # the env id (Task API /design_canvas/...), kept from the earlier package name


def create_server():
    pack = load_pack()

    def factory():
        return DesignCanvasEnvironment(pack)

    def gradio_builder(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
        return build_ui(pack)

    os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
    app = create_app(
        factory,
        CallToolAction,
        CallToolObservation,
        env_name=ENV_NAME,
        max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "64")),
        gradio_builder=gradio_builder,
        custom_tab_name="DesignGym",
        custom_tab_primary=True,
        show_default_tab=True,
        title_override="DesignGym · OpenEnv",
    )
    mount_editor(app, pack)

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        out = {"status": "ok", "tasks": {s: pack.count(s) for s in pack.splits()}}
        if hasattr(pack, "stats"):  # the bucket-backed corpus: snapshot and cache state
            out["corpus"] = pack.stats()
        return out

    @app.get("/api/info", include_in_schema=False)
    def info():
        """What this server can play and how it grades: an eval runner checks this before it starts."""
        from .episodes import REAL_SKINS, SKINS

        try:
            from ..realapps import available
        except ImportError:  # flat layout inside the Docker image
            from realapps import available
        ok, why = available()
        return {
            "skins": list(SKINS),
            "real_skins": list(REAL_SKINS) if ok else [],
            "real_apps_unavailable": None if ok else why,
            "reward_version": int(os.environ.get("DESIGN_CANVAS_REWARD", "1")),
            "tasks": {s: pack.count(s) for s in pack.splits()},
            "pack": {k: pack.manifest.get(k) for k in ("evalset_id", "corpus_snapshot_id") if k in pack.manifest},
        }

    return app


_app = None


def __getattr__(name: str):
    # `uvicorn designgym.server.app:app` works, but importing this module
    # does not load a task pack as a side effect.
    global _app
    if name == "app":
        if _app is None:
            _app = create_server()
        return _app
    raise AttributeError(name)


def main(host: str | None = None, port: int | None = None, workers: int | None = None) -> None:
    import uvicorn

    host = host or os.environ.get("DESIGN_CANVAS_HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", "8000"))
    workers = workers or int(os.environ.get("DESIGN_CANVAS_WORKERS", "1"))
    if workers > 1:  # N single-process servers behind one URL (server/router.py); never uvicorn's --workers
        try:
            from .router import serve
        except ImportError:
            from server.router import serve

        return serve(host, port, workers)
    os.environ["PORT"] = str(port)
    uvicorn.run(create_server(), host=host, port=port, ws_ping_interval=60,
                ws_ping_timeout=300)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    main(a.host, a.port, a.workers)
