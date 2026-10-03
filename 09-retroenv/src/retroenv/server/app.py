"""FastAPI application factory for RetroEnv."""

from __future__ import annotations

import os
import json
from pathlib import Path

from openenv.core.env_server.http_server import create_app

from ..store import TaskStore
from ..retrieval import PrecedentIndex
from .models import RetroRouteAction, RetroRouteObservation
from .retro_environment import RetroRouteEnvironment


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASKS_DIR = Path(os.getenv("RETROENV_TASKS_DIR", PROJECT_ROOT / "tasks"))
STOCKS_DIR = Path(os.getenv("RETROENV_STOCKS_DIR", PROJECT_ROOT / "stocks"))
DEFAULT_SPLIT = os.getenv("RETROENV_DEFAULT_SPLIT", "train")
MAX_TOOL_CALLS = int(os.getenv("RETROENV_MAX_TOOL_CALLS", "32"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))
PUBCHEM_CACHE_PATH = os.getenv("RETROENV_PUBCHEM_CACHE", "")


def _load_pubchem_cache() -> dict:
    if not PUBCHEM_CACHE_PATH:
        return {}
    path = Path(PUBCHEM_CACHE_PATH)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"PubChem cache must be an object: {path}")
    return {str(key).casefold(): row for key, row in value.items() if isinstance(row, dict)}


PUBCHEM_CACHE = _load_pubchem_cache()
TASK_STORE = TaskStore(TASKS_DIR, STOCKS_DIR)
PRECEDENT_INDEX = PrecedentIndex(
    TASK_STORE.tasks("train") if "train" in TASK_STORE.splits() else ()
)


def create_retro_environment() -> RetroRouteEnvironment:
    return RetroRouteEnvironment(
        TASK_STORE,
        default_split=DEFAULT_SPLIT,
        max_tool_calls=MAX_TOOL_CALLS,
        pubchem_cache=PUBCHEM_CACHE,
        precedent_index=PRECEDENT_INDEX,
    )


app = create_app(
    create_retro_environment,
    RetroRouteAction,
    RetroRouteObservation,
    env_name="retro_route_env",
    max_concurrent_envs=MAX_CONCURRENT,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
