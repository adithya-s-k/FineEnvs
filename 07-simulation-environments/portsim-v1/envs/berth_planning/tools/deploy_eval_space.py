"""Deploy the PortSimEnv v1 eval explorer as a Space (FineEnvs/PortSimEnv-Eval).

Not an OpenEnv environment: a small FastAPI app (`berth_openenv.explorer`) serving the viewer in explorer mode (the
overview with the eval table, tasks, and every model rollout replayed in 3D) and its read-only JSON. No sessions; to
play an episode the viewer links to the environment Space (deploy_space.py). The image installs core/ and the
berth_openenv package without the environment's dependencies (openenv, fastmcp, gradio), downloads the binary assets
from the public bucket at build time (space_common.py), and reads the eval rollouts from the bucket, mounted read-only
at /data.

    openenv/.venv/bin/python tools/publish_bucket.py                      # twin + rollouts into the bucket, first
    openenv/.venv/bin/python tools/deploy_eval_space.py --stage-only DIR  # stage only, no push
    openenv/.venv/bin/python tools/deploy_eval_space.py                   # create the Space, upload, mount the bucket
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from space_common import PACKS, stage_package  # noqa: E402

ENV_SPACE = "FineEnvs/PortSimEnv"
SPACE_VARS = {
    "BERTH_TASKS_DIR": ":".join(f"/app/tasks/{p}" for p in PACKS),
    "BERTH_RUNS_DIR": "/data/rollouts",
    "BERTH_VIEWER_MODE": "explorer",
    "BERTH_PLAY_URL": f"https://huggingface.co/spaces/{ENV_SPACE}",
}

LINKS = {
    "Environment (play an episode, connect an agent)": f"https://huggingface.co/spaces/{ENV_SPACE}",
    "Article": "https://huggingface.co/spaces/FineEnvs/simulation-rl-environments",
    "Dataset (tasks, source calls, eval rollouts)": "https://huggingface.co/datasets/FineEnvs/PortSimEnv",
    "Bucket (3D twin data, eval rollouts)": "https://huggingface.co/buckets/FineEnvs/PortSimEnv",
    "Code": "https://github.com/adithya-s-k/FineEnvs/tree/main/07-simulation-environments/portsim-v1",
    "Discussion": "https://github.com/adithya-s-k/FineEnvs/discussions/36",
}

DOCKERFILE = """\
# PortSimEnv v1 eval explorer: the viewer and its read-only JSON, no environment sessions.
# Build context: this folder (core/, openenv/, tasks/ side by side). Rollouts: the bucket, mounted at /data.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/user PORT=8000 \\
    BERTH_TASKS_DIR={tasks_dir} BERTH_RUNS_DIR=/data/rollouts \\
    BERTH_VIEWER_MODE=explorer BERTH_PLAY_URL={play_url}
WORKDIR /app
COPY core /app/core
COPY openenv /app/openenv
COPY tasks /app/tasks
COPY fetch_assets.py /app/fetch_assets.py
# berth_openenv without the environment's dependencies: the explorer imports only api.py (FastAPI + berth_core)
RUN pip install /app/core "fastapi>=0.110" "uvicorn[standard]>=0.30" && pip install --no-deps /app/openenv \\
    && python /app/fetch_assets.py && useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["sh", "-c", "uvicorn berth_openenv.explorer:app --host 0.0.0.0 --port ${{PORT:-8000}}"]
"""

CARD = """\
---
title: PortSimEnv-Eval
emoji: 🚢
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
base_path: /viewer/
pinned: false
license: cc-by-sa-4.0
tags: [simulation, rl-environment, evaluation, logistics, scheduling, operations-research]
short_description: PortSimEnv v1 eval, every model rollout in 3D
---

# PortSimEnv v1 eval

The eval of [PortSimEnv v1]({env_url}): models re-plan a broken week of container-ship dockings at a Port of
Barcelona quay, on 50 held-out tasks (the dock-v1 eval split) with one graded submit each, scored against the plan a
CP-SAT solver proved optimal. This Space shows the eval table and replays every rollout in 3D, with its dock chart,
grade and transcript. It is read-only: to play an episode or connect an agent, use the
[environment Space]({env_url}).

| path | what |
|---|---|
| `/viewer/` | overview with the eval table; click a model for its rollouts |
| `/viewer/#/tasks`, `#/task/<id>` | tasks, with the published, naive and optimal plans and every model's plan |
| `/viewer/#/run/<run>/<model>/<task>` | one rollout in 3D; add `?embed=1` (before `#`) for an embed without the header |
| `/api/...` | read-only JSON: tasks, reference plans, runs, episodes |

The eval rollouts are read from the public bucket [{bucket}](https://huggingface.co/buckets/{bucket}) (`rollouts/`),
mounted read-only at `/data`; the 3D twin's data is downloaded from it at build time.

## Links

{links}
"""


def stage(dest: Path, bucket: str, with_twin: bool = False) -> Path:
    stage_package(dest, bucket, with_twin=with_twin)  # core/, openenv/, the task packs, fetch_assets.py
    (dest / "Dockerfile").write_text(DOCKERFILE.format(tasks_dir=SPACE_VARS["BERTH_TASKS_DIR"],
                                                       play_url=SPACE_VARS["BERTH_PLAY_URL"]))
    links = "\n".join(f"- {k}: {v}" for k, v in LINKS.items())
    (dest / "README.md").write_text(CARD.format(env_url=SPACE_VARS["BERTH_PLAY_URL"], bucket=bucket, links=links))
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default="FineEnvs/PortSimEnv-Eval")
    ap.add_argument("--bucket", default="FineEnvs/PortSimEnv")
    ap.add_argument("--hardware", default="cpu-basic")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--stage-only", metavar="DIR", help="stage into DIR, no push")
    ap.add_argument("--with-twin", action="store_true", help="with --stage-only: include the twin files (local docker test)")
    args = ap.parse_args(argv)

    if args.stage_only:
        dest = Path(args.stage_only)
        shutil.rmtree(dest, ignore_errors=True)
        stage(dest, args.bucket, with_twin=args.with_twin)
        print(dest)
        return 0

    from huggingface_hub import HfApi, Volume

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", space_hardware=args.hardware,
                    private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="portsimenv-eval-space-") as tmp:
        dest = stage(Path(tmp) / "PortSimEnv-Eval", args.bucket)
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(dest),
                          commit_message="PortSimEnv v1 eval explorer", delete_patterns=["*"])
    api.set_space_volumes(args.space, volumes=[Volume(type="bucket", source=args.bucket, mount_path="/data",
                                                      read_only=True)])
    for key, value in SPACE_VARS.items():
        api.add_space_variable(args.space, key, value)
    print(f"https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
