"""FastAPI app: MCP tools over /ws and /mcp, the Task API, and the playground at /web."""

import os

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from rdkit import RDLogger

from .config import ENV_NAME
from .environment import RetroRouteEnvironment
from .molecule3d import STATIC, viewer_page

# Agents send unparsable SMILES all the time; the verifier reports them, so
# RDKit's per-molecule stderr messages are only noise in the server log.
RDLogger.DisableLog("rdApp.*")
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")


def _build_ui(*args, **kwargs):
    from .ui import build_ui

    return build_ui(*args, **kwargs)


app = create_app(
    RetroRouteEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name=ENV_NAME,
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "64")),
    gradio_builder=_build_ui,
    show_default_tab=False,
    title_override="RetroEnv",
)

# The playground's 3D view: a conformer page per SMILES, and the vendored 3Dmol.js it loads.
app.mount("/molecule/static", StaticFiles(directory=STATIC), name="molecule-static")


@app.get("/molecule/3d", response_class=HTMLResponse, include_in_schema=False)
def molecule_3d(smiles: str) -> HTMLResponse:
    # A given SMILES always yields the same conformer, so browsers may keep it.
    return HTMLResponse(viewer_page(smiles), headers={"Cache-Control": "public, max-age=86400"})
