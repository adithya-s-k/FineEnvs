"""The 3D viewer (`/viewer/`) and the read-only JSON it uses (`/api/...`), on the env server's port, or alone in the
eval explorer (`explorer.py`).

    /api/config                       the viewer's mode: "env" (play here) or "explorer" (no sessions; play links out)
    /api/tasks                        task summaries
    /api/tasks/{id}                   a public task (what the agent sees, structured)
    /api/tasks/{id}/reference         naive and optimal plans - for people; no tool exposes these to the agent
    /api/runs                         rollout runs found under BERTH_RUNS_DIR
    /api/runs/{run}                   one run's episode index
    /api/runs/{run}/episode           one rollout (?model=&task_id=)
    /api/episodes                     live episodes on this server (newest first; env mode only)
    /api/episodes/{episode_id}        one live episode: its task id, every plan checked so far, the grade

    BERTH_VIEWER_MODE=explorer        no live episodes, no Play page: playing links to BERTH_PLAY_URL (the env Space)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from berth_core import TaskPack

WEB = Path(__file__).resolve().parent / "web"
PLAY_URL = "https://huggingface.co/spaces/FineEnvs/PortSimEnv"  # the environment Space: Try Environment, the editor
ENV_URL = "https://fineenvs-portsimenv.hf.space"  # its server, for an agent's OpenEnv client
DEFAULT_RUNS = Path(__file__).resolve().parents[4] / "results" / "rollouts"
_SAFE = re.compile(r"^[A-Za-z0-9._@:+-]+$")


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", model)


def runs_dir() -> Path:
    return Path(os.environ.get("BERTH_RUNS_DIR", DEFAULT_RUNS))


def _summary(t) -> dict:
    return {"task_id": t.task_id, "split": t.split, "quay": t.quay, "terminal": t.terminal, "week": t.week,
            "difficulty": t.difficulty, "ships": len(t.ships), "disruptions": [e["type"] for e in t.disruptions]}


def viewer_config(mode: str | None = None) -> dict:
    mode = (mode or os.environ.get("BERTH_VIEWER_MODE") or "env").strip().lower()
    return {"mode": "explorer" if mode == "explorer" else "env",
            "play_url": os.environ.get("BERTH_PLAY_URL") or PLAY_URL,
            "env_url": os.environ.get("BERTH_ENV_URL") or ENV_URL}


def mount_viewer(app: FastAPI, pack: TaskPack, mode: str | None = None) -> None:
    """mode: "env" (default; live episodes, Play) or "explorer" (read-only); None reads BERTH_VIEWER_MODE."""
    config = viewer_config(mode)

    @app.get("/api/config", tags=["Viewer"])
    def viewer_mode():
        return config

    @app.get("/api/tasks", tags=["Viewer"])
    def tasks():
        return [_summary(t) for t in pack.tasks]

    @app.get("/api/tasks/{task_id}", tags=["Viewer"])
    def task(task_id: str):
        try:
            return pack.public(pack.get(task_id))
        except KeyError:
            raise HTTPException(404, "unknown task") from None

    @app.get("/api/tasks/{task_id}/reference", tags=["Viewer"])
    def reference(task_id: str):
        try:
            return pack.get(task_id).reference
        except KeyError:
            raise HTTPException(404, "unknown task") from None

    @app.get("/api/runs", tags=["Viewer"])
    def runs():
        root = runs_dir()
        out = []
        for d in sorted(root.iterdir() if root.is_dir() else [], reverse=True):
            idx = d / "index.json"
            if not idx.is_file():
                continue
            # only episodes on tasks this server serves (runs on other packs are hidden, not broken links)
            eps = [e for e in json.loads(idx.read_text()).get("episodes", []) if e.get("task_id") in pack._by_id]
            if not eps:
                continue
            out.append({"run": d.name, "models": sorted({e["model"] for e in eps}), "episodes": len(eps),
                        "mean_reward": round(sum(e["reward"] for e in eps) / len(eps), 4)})
        return out

    @app.get("/api/runs/{run}", tags=["Viewer"])
    def run(run: str):
        if not _SAFE.match(run) or not (runs_dir() / run / "index.json").is_file():
            raise HTTPException(404, "unknown run")
        body = json.loads((runs_dir() / run / "index.json").read_text())
        body["episodes"] = [e for e in body.get("episodes", []) if e.get("task_id") in pack._by_id]
        return body

    @app.get("/api/runs/{run}/episode", tags=["Viewer"])
    def episode(run: str, model: str, task_id: str):
        if not (_SAFE.match(run) and _SAFE.match(task_id)):
            raise HTTPException(404, "unknown episode")
        path = runs_dir() / run / model_slug(model) / f"{task_id}.json"
        if not path.is_file():
            raise HTTPException(404, "unknown episode")
        return json.loads(path.read_text())

    if config["mode"] == "env":  # live episodes need the env's sessions (and its OpenEnv imports)
        _mount_live(app)

    @app.get("/viewer", include_in_schema=False)
    def viewer_slash():
        return RedirectResponse("/viewer/")

    app.mount("/viewer", StaticFiles(directory=str(WEB), html=True), name="viewer")


def _mount_live(app: FastAPI) -> None:
    try:
        from .environment import STORE
    except ImportError:
        from environment import STORE

    @app.get("/api/episodes", tags=["Viewer"])
    def live_episodes():
        return [{k: v for k, v in ep.summary().items() if k != "steps"} | {"checks": len(ep.steps)}
                for ep in STORE.recent()]

    @app.get("/api/episodes/{episode_id}", tags=["Viewer"])
    def live_episode(episode_id: str):
        ep = STORE.get(episode_id)
        if ep is None:
            raise HTTPException(404, "unknown or expired episode")
        return ep.summary()
