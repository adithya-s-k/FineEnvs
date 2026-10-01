"""SmolDataEnvs task serving, capture and UI through OpenEnv's Harbor service."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

# Both local jobs and Spaces prepare exactly these train/test manifests.
data = Path(os.environ.get("SMOLDATA_DATA", "prepared")).resolve()
os.environ.setdefault(
    "OPENENV_DATASETS", ",".join(str(data / "datasets" / s) for s in ("train", "test"))
)
os.environ.setdefault("OPENENV_MAX_OUTPUT_TOKENS", "4096")
os.environ.setdefault("MAX_CONCURRENT_ENVS", "40")
os.environ.setdefault("OPENENV_HARBOR_REWARD_KEY", "correctness,reward")
os.environ.setdefault(
    "OPENENV_HARBOR_AGENT_VERSIONS",
    '{"opencode":"1.18.31","claude-code":"2.1.270","codex":"0.154.0","mini-swe-agent":"2.4.6"}',
)

# This public app owns the capture proxy, task API, sandbox lifecycle and trace UI.
from harbor_env.server.app import app as harbor_app


@asynccontextmanager
async def lifespan(app):
    async with harbor_app.router.lifespan_context(harbor_app):
        yield


# Register task metadata before Harbor's UI, which is mounted at the root.
app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/smoldataenv/splits")
def split_paths():
    return {s: str(data / "datasets" / s) for s in ("train", "test")}


@app.get("/smoldataenv/trials/{trial_name}/tool-count")
def tool_count(trial_name: str):
    from fastapi import HTTPException

    from .environment import count_tools

    if Path(trial_name).name != trial_name or trial_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid trial name")
    root = Path(
        os.environ.get("OPENENV_HARBOR_TRIALS_DIR", "/tmp/openenv-harbor-trials")
    )
    return {
        "native_tool_calls": count_tools(root / trial_name / "agent/trajectory.json")
    }


app.mount("/", harbor_app)
