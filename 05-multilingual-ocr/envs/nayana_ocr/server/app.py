import os

from fastapi import HTTPException
from fastapi.responses import FileResponse
from openenv.core.env_server import create_app

from ..models import NayanaAction, NayanaObservation
from .environment import NayanaEnvironment, configured_catalog
from .gradio_ui import build_ui


def create_server():
    catalog = configured_catalog()
    os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
    app = create_app(
        lambda: NayanaEnvironment(catalog),
        NayanaAction,
        NayanaObservation,
        env_name="nayana_ocr",
        max_concurrent_envs=int(os.environ.get("NAYANA_MAX_SESSIONS", "16")),
        gradio_builder=build_ui,
        custom_tab_name="Try it",
        custom_tab_primary=True,
        show_default_tab=False,
        title_override="Nayana multilingual OCR",
    )

    @app.get("/healthz")
    def health():
        return {"status": "ok", "snapshot_id": catalog.manifest["snapshot_id"]}

    @app.get("/manifest")
    def manifest():
        return {
            key: catalog.manifest[key]
            for key in (
                "status",
                "snapshot_id",
                "schema_version",
                "datasets_version",
                "config",
                "source_license",
                "counts",
                "pages",
                "media_bytes",
            )
        }

    @app.get("/assets/{sha}")
    def asset(sha: str):
        try:
            path, mime = catalog.asset(sha)
        except KeyError as error:
            raise HTTPException(404, "Unknown asset") from error
        return FileResponse(
            path,
            media_type=mime,
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "ETag": f'"{sha}"',
            },
        )

    return app


def main():
    import uvicorn

    uvicorn.run(
        "nayana_ocr.server.app:create_server", factory=True, host="0.0.0.0", port=8000
    )
