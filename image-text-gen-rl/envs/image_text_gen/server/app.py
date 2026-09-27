import os

from openenv.core.env_server import create_app

from ..data.catalog import configured_catalog
from ..models import MAX_IMAGE_BYTES, ImageTextGenAction, ImageTextGenObservation
from .environment import ImageTextGenEnvironment
from .gradio_ui import build_ui
from .images import TARGET_LONG_SIDE
from .scoring import POLICY

ENV_NAME = "image_text_gen"


def create_server():
    catalog = configured_catalog()
    os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")
    app = create_app(
        lambda: ImageTextGenEnvironment(catalog),
        ImageTextGenAction,
        ImageTextGenObservation,
        env_name=ENV_NAME,
        max_concurrent_envs=int(os.environ.get("IMAGE_TEXT_GEN_MAX_SESSIONS", "64")),
        gradio_builder=build_ui,
        custom_tab_name="Try it",
        custom_tab_primary=True,
        show_default_tab=False,
        title_override="Image text generation RL",
    )

    @app.get("/healthz")
    def health():
        return {"status": "ok", "catalog_id": catalog.manifest["catalog_id"]}

    @app.get("/manifest")
    def manifest():
        from .verifier import verifier_info

        return {
            **catalog.manifest,
            "splits": catalog.splits(),
            "action": {
                "image": "base64 PNG/JPEG/WebP",
                "max_bytes": MAX_IMAGE_BYTES,
                "verifier_long_side": TARGET_LONG_SIDE,
            },
            "grading": {"scoring": POLICY, "verifier": verifier_info()},
        }

    return app


def main():
    import uvicorn

    uvicorn.run(
        "image_text_gen.server.app:create_server",
        factory=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
