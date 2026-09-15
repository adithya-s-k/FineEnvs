import os

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response
from openenv.core.env_server import create_app
from pydantic import BaseModel, Field

from ..data.schema import FAMILIES
from ..models import NayanaAction, NayanaObservation
from .environment import NayanaEnvironment, configured_catalog
from .gradio_ui import build_ui


class DataQuery(BaseModel):
    snapshot_id: str
    split: str = "train"
    languages: list[str] | None = None
    families: list[str] | None = None


class BlockQuery(DataQuery):
    block_id: str
    start: int = Field(default=0, ge=0)
    limit: int = Field(default=512, ge=1, le=1000)


class SampleQuery(DataQuery):
    per_group: int = Field(default=4, ge=1, le=1000)
    seed: int = 42


class PrefetchQuery(BaseModel):
    snapshot_id: str
    task_ids: list[str] = Field(default_factory=list, max_length=64)
    block_ids: list[str] = Field(default_factory=list, max_length=64)


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
        from .judge import judge_info
        from .layout import POLICY

        result = {
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
        for key in (
            "storage",
            "index_version",
            "bucket_id",
            "inventory_id",
            "annotation_validation",
        ):
            if key in catalog.manifest:
                result[key] = catalog.manifest[key]
        result["grading"] = {
            "descriptive_vqa": judge_info(),
            "layout_detection": POLICY,
        }
        return result

    def corpus_query(request, operation):
        if not hasattr(catalog, "blocks"):
            raise HTTPException(
                501, "This server uses a small snapshot, not the full corpus index"
            )
        if request.snapshot_id != catalog.manifest["snapshot_id"]:
            raise HTTPException(
                409, "Corpus index changed; reload the pinned training plan"
            )
        try:
            return operation()
        except (ValueError, IndexError, KeyError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/data/blocks")
    def blocks(request: DataQuery):
        return corpus_query(
            request,
            lambda: {
                "snapshot_id": request.snapshot_id,
                "blocks": catalog.blocks(
                    request.split, request.languages, request.families
                ),
            },
        )

    @app.post("/data/block-tasks")
    def block_tasks(request: BlockQuery):
        return corpus_query(
            request,
            lambda: {
                "snapshot_id": request.snapshot_id,
                "tasks": catalog.block_tasks(
                    request.block_id,
                    request.split,
                    request.families,
                    request.start,
                    request.limit,
                ),
            },
        )

    @app.post("/data/sample")
    def sample(request: SampleQuery):
        return corpus_query(
            request,
            lambda: {
                "snapshot_id": request.snapshot_id,
                "tasks": catalog.sample(
                    request.split,
                    request.languages or catalog.languages,
                    request.families or list(FAMILIES),
                    request.per_group,
                    request.seed,
                ),
            },
        )

    @app.post("/data/prefetch")
    def prefetch(request: PrefetchQuery):
        return corpus_query(
            request,
            lambda: catalog.prefetch(
                task_ids=request.task_ids, block_ids=request.block_ids
            ),
        )

    @app.get("/data/cache")
    def cache():
        if not hasattr(catalog, "stats"):
            raise HTTPException(501, "Full-corpus cache is not configured")
        return catalog.stats()

    @app.get("/assets/{sha}")
    def asset(sha: str, task_id: str | None = None):
        try:
            if hasattr(catalog, "asset_bytes"):
                raw, mime = catalog.asset_bytes(sha, task_id)
                return Response(
                    raw,
                    media_type=mime,
                    headers={
                        "Cache-Control": "public, max-age=31536000, immutable",
                        "ETag": f'"{sha}"',
                    },
                )
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
