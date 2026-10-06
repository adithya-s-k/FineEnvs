"""Deploy PortSimEnv v1 as an OpenEnv Space (FineEnvs/PortSimEnv).

The Space is one OpenEnv package: core/ (the grader), openenv/ (server, tools, rubric, viewer), the dock-v1 task
packs, openenv.yaml (with its validation block) and a Dockerfile at the root. Binary files (the 3D twin's data, the
gzipped train pack) are not in the Space repo: the image downloads them from the public bucket at build time (a Docker Space build copies LFS
pointer files, not the files). Eval rollouts are read from the bucket, mounted read-only at /data.

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

HERE = Path(__file__).resolve().parents[1]
OPENENV_BIN = HERE / "openenv" / ".venv" / "bin" / "openenv"
PACKS = ["dock-v1-eval", "dock-v1-train"]
TWIN_BINARIES = ["twin.json.gz", "terrain.png", "cover.png", "scenery.json.gz", "surface.webp"]
BUCKET_TASKS = ["tasks/dock-v1-train/tasks.jsonl.gz"]
SPACE_VARS = {
    "BERTH_TASKS_DIR": "/app/tasks/dock-v1-eval:/app/tasks/dock-v1-train",
    "BERTH_RUNS_DIR": "/data/rollouts",
    "MAX_CONCURRENT_ENVS": "256",
}  # gzipped packs: binary, so from the bucket too
IGNORE = shutil.ignore_patterns("space_root", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info", "tests",
                                "node_modules", ".DS_Store", "build_log.jsonl", "*.gz", *TWIN_BINARIES)

LINKS = {
    "Article": "https://huggingface.co/spaces/FineEnvs/simulation-rl-environments",
    "Dataset (tasks, source calls, eval rollouts)": "https://huggingface.co/datasets/FineEnvs/PortSimEnv",
    "Bucket (3D twin data, eval rollouts)": "https://huggingface.co/buckets/FineEnvs/PortSimEnv",
    "Code": "https://github.com/adithya-s-k/FineEnvs",
    "Discussion": "https://github.com/adithya-s-k/FineEnvs/discussions/36",
}

FETCH = """\
\"\"\"Build step: download the binary assets from the public bucket (twin data into the installed viewer, the
gzipped train pack into /app/tasks). A Docker Space build copies LFS pointer files, so binaries can't come from the
Space repo.\"\"\"
import pathlib
import urllib.request

import berth_openenv

twin = pathlib.Path(berth_openenv.__file__).parent / "web" / "twin"
files = {{"twin/" + n: twin / n for n in {twin_files!r}}}
files.update({{p: pathlib.Path("/app") / p for p in {bucket_tasks!r}}})
for remote, local in files.items():
    if local.is_file():
        continue
    local.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen("https://huggingface.co/buckets/{bucket}/resolve/" + remote, timeout=120) as r:
        body = r.read()
    if body.startswith(b"version https://git-lfs") or len(body) < 1000:
        raise SystemExit("bad download for " + remote)
    local.write_bytes(body)
    print(remote, len(body))
"""

DOCKERFILE = """\
# PortSimEnv v1: OpenEnv server + 3D viewer. Build context: this folder (core/, openenv/, tasks/ side by side).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/user PORT=8000 ENABLE_WEB_INTERFACE=true \\
    BERTH_TASKS_DIR=/app/tasks/dock-v1-eval:/app/tasks/dock-v1-train BERTH_RUNS_DIR=/data/rollouts \\
    MAX_CONCURRENT_ENVS=256
WORKDIR /app
COPY core /app/core
COPY openenv /app/openenv
COPY tasks /app/tasks
COPY oracle /app/oracle
COPY fetch_assets.py /app/fetch_assets.py
RUN pip install /app/core /app/openenv && python /app/fetch_assets.py \\
    && useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["sh", "-c", "uvicorn berth_openenv.server:app --host 0.0.0.0 --port ${PORT:-8000} --ws-ping-interval 60 --ws-ping-timeout 600"]
"""


def stage(dest: Path, bucket: str, with_twin: bool = False) -> Path:
    shutil.copytree(HERE / "core", dest / "core", ignore=IGNORE)
    shutil.copytree(HERE / "openenv", dest / "openenv", ignore=IGNORE)
    for pack in PACKS:
        shutil.copytree(HERE / "tasks" / pack, dest / "tasks" / pack, ignore=IGNORE)
    if with_twin:  # local image tests only; the Space downloads these from the bucket
        for name in TWIN_BINARIES:
            shutil.copyfile(HERE / "openenv" / "berth_openenv" / "web" / "twin" / name,
                            dest / "openenv" / "berth_openenv" / "web" / "twin" / name)
        for path in BUCKET_TASKS:
            shutil.copyfile(HERE / path, dest / path)
    shutil.copyfile(HERE / "openenv" / "openenv.yaml", dest / "openenv.yaml")
    for f in (HERE / "openenv" / "space_root").iterdir():  # OpenEnv package root: __init__, client, models, pyproject
        if f.is_file():
            shutil.copyfile(f, dest / f.name)
    (dest / "outputs").mkdir(exist_ok=True)
    (dest / "outputs" / ".gitkeep").write_text("")
    shutil.copytree(HERE / "openenv" / "oracle", dest / "oracle", ignore=IGNORE)
    (dest / "Dockerfile").write_text(DOCKERFILE)
    (dest / "fetch_assets.py").write_text(FETCH.format(bucket=bucket, twin_files=TWIN_BINARIES, bucket_tasks=BUCKET_TASKS))
    card = (HERE / "openenv" / "README.md").read_text()
    # The Space opens on the viewer (overview, play, explorer); OpenEnv's own web UI stays at /web.
    card = card.replace("base_path: /web", "base_path: /viewer/", 1)
    card = card.replace("tags: [openenv, ", "tags: [openenv, simulation, rl-environment, ", 1)
    card += "\n## Links\n\n" + "\n".join(f"- {k}: {v}" for k, v in LINKS.items()) + "\n"
    card += (f"\nThe 3D twin's data and the eval rollouts live in the public bucket [{bucket}]"
             f"(https://huggingface.co/buckets/{bucket}), mounted read-only at `/data`.\n")
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

    from huggingface_hub import HfApi, Volume

    api = HfApi()
    with tempfile.TemporaryDirectory(prefix="portsimenv-space-") as tmp:
        dest = stage(Path(tmp) / "PortSimEnv", args.bucket)
        subprocess.run([str(OPENENV_BIN), "validate", str(dest)], check=True)
        push = [str(OPENENV_BIN), "push", str(dest), "--repo-id", args.space, "--hardware", args.hardware]
        subprocess.run(push + (["--private"] if args.private else []), check=True)
    api.set_space_volumes(args.space, volumes=[Volume(type="bucket", source=args.bucket, mount_path="/data",
                                                      read_only=True)])
    # `openenv push` copies openenv.yaml's (empty, local) variable defaults onto the Space, where they would
    # override the image's ENV; set the Space's real values.
    for key, value in SPACE_VARS.items():
        api.add_space_variable(args.space, key, value)
    print(f"https://huggingface.co/spaces/{args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
