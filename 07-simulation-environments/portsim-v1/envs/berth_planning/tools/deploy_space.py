"""Deploy PortSimEnv v1 as an OpenEnv Space (FineEnvs/PortSimEnv).

The Space is one OpenEnv package: core/ (the grader), openenv/ (server, tools, rubric, viewer), the dock-v1 task
packs, openenv.yaml (with its validation block) and a Dockerfile at the root. Binary files (the 3D twin's data, the
gzipped train pack) are not in the Space repo: the image downloads them from the public bucket at build time (a Docker Space build copies LFS
pointer files, not the files). The eval rollouts are not served here: they are in the eval Space (deploy_eval_space.py).

    openenv/.venv/bin/python tools/publish_bucket.py           # twin + rollouts into the bucket, first
    openenv/.venv/bin/python tools/deploy_space.py --stage-only DIR  # stage + `openenv validate`, no push
    openenv/.venv/bin/python tools/deploy_space.py               # stage, validate, `openenv push`, mount the bucket
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from space_common import HERE, IGNORE, stage_package  # noqa: E402  (shared with deploy_eval_space.py)

OPENENV_BIN = HERE / "openenv" / ".venv" / "bin" / "openenv"
SPACE_VARS = {
    "BERTH_TASKS_DIR": "/app/tasks/dock-v1-eval:/app/tasks/dock-v1-train",
    "BERTH_RUNS_DIR": "/app/runs",
    "MAX_CONCURRENT_ENVS": "256",
}

LINKS = {
    "Eval (model rollouts in 3D)": "https://huggingface.co/spaces/FineEnvs/PortSimEnv-Eval",
    "Article": "https://huggingface.co/spaces/FineEnvs/simulation-rl-environments",
    "Dataset (tasks, source calls, eval rollouts)": "https://huggingface.co/datasets/FineEnvs/PortSimEnv",
    "Bucket (3D twin data, eval rollouts)": "https://huggingface.co/buckets/FineEnvs/PortSimEnv",
    "Code": "https://github.com/adithya-s-k/FineEnvs/tree/main/07-simulation-environments/portsim-v1",
    "Discussion": "https://github.com/adithya-s-k/FineEnvs/discussions/36",
}

DOCKERFILE = """\
# PortSimEnv v1: OpenEnv server + 3D viewer. Build context: this folder (core/, openenv/, tasks/ side by side).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/user PORT=8000 ENABLE_WEB_INTERFACE=true \\
    BERTH_TASKS_DIR=/app/tasks/dock-v1-eval:/app/tasks/dock-v1-train BERTH_RUNS_DIR=/app/runs \\
    MAX_CONCURRENT_ENVS=256
WORKDIR /app
COPY core /app/core
COPY openenv /app/openenv
COPY tasks /app/tasks
COPY oracle /app/oracle
COPY fetch_assets.py /app/fetch_assets.py
RUN pip install /app/core /app/openenv && python /app/fetch_assets.py \\
    && mkdir -p /app/runs && useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["sh", "-c", "uvicorn berth_openenv.server:app --host 0.0.0.0 --port ${PORT:-8000} --ws-ping-interval 60 --ws-ping-timeout 600"]
"""


def stage(dest: Path, bucket: str, with_twin: bool = False) -> Path:
    stage_package(dest, bucket, with_twin=with_twin)  # core/, openenv/, the task packs, fetch_assets.py
    shutil.copyfile(HERE / "openenv" / "openenv.yaml", dest / "openenv.yaml")
    for f in (HERE / "openenv" / "space_root").iterdir():  # OpenEnv package root: __init__, client, models, pyproject
        if f.is_file():
            shutil.copyfile(f, dest / f.name)
    (dest / "outputs").mkdir(exist_ok=True)
    (dest / "outputs" / ".gitkeep").write_text("")
    shutil.copytree(HERE / "openenv" / "oracle", dest / "oracle", ignore=IGNORE)
    (dest / "Dockerfile").write_text(DOCKERFILE)
    card = (HERE / "openenv" / "README.md").read_text()
    card = card.replace("tags: [openenv, ", "tags: [openenv, simulation, rl-environment, ", 1)
    card += "\n## Links\n\n" + "\n".join(f"- {k}: {v}" for k, v in LINKS.items()) + "\n"
    card += (f"\nThe 3D twin's data is downloaded from the public bucket [{bucket}]"
             f"(https://huggingface.co/buckets/{bucket}) at build time. The eval (model rollouts in 3D) is a separate "
             "Space: [FineEnvs/PortSimEnv-Eval](https://huggingface.co/spaces/FineEnvs/PortSimEnv-Eval).\n")
    (dest / "README.md").write_text(card)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default="FineEnvs/PortSimEnv")
    ap.add_argument("--bucket", default="FineEnvs/PortSimEnv")
    ap.add_argument("--hardware", default="cpu-basic")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--stage-only", metavar="DIR", help="stage and validate into DIR, no push")
    ap.add_argument("--with-twin", action="store_true", help="with --stage-only: include the twin files (local docker test)")
    args = ap.parse_args(argv)

    if args.stage_only:
        dest = Path(args.stage_only)
        shutil.rmtree(dest, ignore_errors=True)
        stage(dest, args.bucket, with_twin=args.with_twin)
        subprocess.run([str(OPENENV_BIN), "validate", str(dest)], check=True)
        print(dest)
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="portsimenv-space-") as tmp:
        dest = stage(Path(tmp) / "PortSimEnv", args.bucket)
        subprocess.run([str(OPENENV_BIN), "validate", str(dest)], check=True)
        push = [str(OPENENV_BIN), "push", str(dest), "--repo-id", args.space, "--hardware", args.hardware]
        subprocess.run(push + (["--private"] if args.private else []), check=True)
    # A plain environment: no eval rollouts here (they are served by the eval Space), so no bucket mount.
    runtime = api.space_info(args.space).runtime
    if runtime and runtime.volumes:
        api.delete_space_volumes(args.space)
    # `openenv push` copies openenv.yaml's (empty, local) variable defaults onto the Space, where they would
    # override the image's ENV; set the Space's real values.
    for key, value in SPACE_VARS.items():
        api.add_space_variable(args.space, key, value)
    print(f"https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
