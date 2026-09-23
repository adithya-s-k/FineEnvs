import os

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from openenv.core.env_server import create_app

from ..models import AsrAction, AsrObservation
from .environment import AsrEnvironment, configured_catalog
from .gradio_ui import build_ui
from .rewards import ERROR_WEIGHT, EXACT_WEIGHT, GRADING_POLICY


def create_server():
    catalog = configured_catalog()
    os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
    app = create_app(
        lambda: AsrEnvironment(catalog),
        AsrAction,
        AsrObservation,
        env_name="multilingual_asr",
        max_concurrent_envs=int(os.environ.get("ASR_MAX_SESSIONS", "16")),
        gradio_builder=build_ui,
        custom_tab_name="Try it",
        custom_tab_primary=True,
        show_default_tab=False,
        title_override="Multilingual ASR (FLEURS)",
    )

    @app.get("/healthz")
    def health():
        return {"status": "ok", "snapshot_id": catalog.snapshot_id}

    @app.get("/manifest")
    def manifest():
        return {
            **catalog.manifest,
            "splits": catalog.splits() if hasattr(catalog, "splits") else None,
            "eval_splits": {
                name: len(rows)
                for name, rows in getattr(catalog, "eval_splits", {}).items()
            },
            "grading": {
                "policy": GRADING_POLICY,
                "reward": f"{ERROR_WEIGHT} * max(0, 1 - error_rate) + {EXACT_WEIGHT} * exact_match",
                "error_unit": "cer for scripts without word spacing, wer otherwise",
            },
        }

    @app.get("/assets/{sha}")
    def asset(sha: str, task_id: str | None = None):
        try:
            # A snapshot addresses audio by hash alone; the indexed corpus needs the
            # task to know which row group to read.
            if task_id is not None and hasattr(catalog, "audio_bytes"):
                raw, mime = catalog.asset(sha, task_id)
                return Response(
                    content=raw, media_type=mime, headers={"ETag": f'"{sha}"'}
                )
            path, mime = catalog.asset(sha)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return FileResponse(path, media_type=mime, headers={"ETag": f'"{sha}"'})

    return app


def main():
    import uvicorn

    uvicorn.run(
        "multilingual_asr.server.app:create_server",
        factory=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8006")),
    )
