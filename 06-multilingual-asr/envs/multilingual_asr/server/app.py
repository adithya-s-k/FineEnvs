import os

from fastapi import HTTPException
from fastapi.responses import FileResponse
from openenv.core.env_server import create_app

from ..models import AsrAction, AsrObservation
from .environment import AsrEnvironment, configured_catalog
from .rewards import ERROR_WEIGHT, EXACT_WEIGHT, GRADING_POLICY


def create_server():
    catalog = configured_catalog()
    app = create_app(
        lambda: AsrEnvironment(catalog),
        AsrAction,
        AsrObservation,
        env_name="multilingual_asr",
        max_concurrent_envs=int(os.environ.get("ASR_MAX_SESSIONS", "16")),
        title_override="Multilingual ASR (FLEURS)",
    )

    @app.get("/healthz")
    def health():
        return {"status": "ok", "snapshot_id": catalog.snapshot_id}

    @app.get("/manifest")
    def manifest():
        return {
            **catalog.manifest,
            "grading": {
                "policy": GRADING_POLICY,
                "reward": f"{ERROR_WEIGHT} * max(0, 1 - error_rate) + {EXACT_WEIGHT} * exact_match",
                "error_unit": "cer for scripts without word spacing, wer otherwise",
            },
        }

    @app.get("/assets/{sha}")
    def asset(sha: str):
        try:
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
