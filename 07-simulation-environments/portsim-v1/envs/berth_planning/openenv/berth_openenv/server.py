"""The server: OpenEnv's app plus the 3D viewer on the same port.

OpenEnv provides `/reset`, `/step`, `/state`, `/schema`, `/ws`, `/mcp`, `/health`, the Task API under
`/berth_planning/...` and the Gradio UI at `/web`: a "Dock planner" tab where a person plays an episode through the
Task API and a `/ws` session (`/viewer/#/play`), next to OpenEnv's own Playground tab. We add the viewer
(`/viewer/`: Play, and the Explorer of tasks, model rollouts and live episodes) and its read-only JSON under `/api/`.

    uvicorn berth_openenv.server:app --port 8000
"""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from berth_core import load_pack

try:
    from .api import mount_viewer
    from .environment import BerthPlanningEnvironment, BerthState
    from .gradio_ui import build_ui
except ImportError:
    from api import mount_viewer
    from environment import BerthPlanningEnvironment, BerthState
    from gradio_ui import build_ui

ENV_NAME = "berth_planning"


def create_server():
    pack = load_pack()

    def factory():
        return BerthPlanningEnvironment(pack)

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
        custom_tab_name="PortSimEnv v1",
        custom_tab_primary=True,
        show_default_tab=True,
        title_override="PortSimEnv v1 · OpenEnv",
        state_cls=BerthState,
    )
    mount_viewer(app, pack)

    @app.get("/healthz", include_in_schema=False)
    def healthz():
        return {"status": "ok", "tasks": {s: pack.count(s) for s in pack.splits()}}

    return app


_app = None


def __getattr__(name: str):
    global _app
    if name == "app":
        if _app is None:
            _app = create_server()
        return _app
    raise AttributeError(name)


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    import uvicorn

    port = port or int(os.environ.get("PORT", "8000"))
    uvicorn.run(create_server(), host=host, port=port, ws_ping_interval=60, ws_ping_timeout=300)


if __name__ == "__main__":
    main()
