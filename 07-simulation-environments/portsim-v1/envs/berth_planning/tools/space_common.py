"""Staging shared by the two Spaces: the environment (deploy_space.py) and the eval explorer (deploy_eval_space.py).

Both images install core/ (the grader) and openenv/ (the berth_openenv package, with the viewer) and serve the dock-v1
task packs. Binary files (the 3D twin's data, the gzipped train pack) are not in a Space repo: a Docker Space build
copies LFS pointer files, so the image's fetch_assets.py downloads them from the public bucket at build time.
"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PACKS = ["dock-v1-eval", "dock-v1-train"]
TWIN_BINARIES = ["twin.json.gz", "terrain.png", "cover.png", "scenery.json.gz", "surface.webp"]
BUCKET_TASKS = ["tasks/dock-v1-train/tasks.jsonl.gz"]  # gzipped packs: binary, so from the bucket too
IGNORE = shutil.ignore_patterns("space_root", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "*.egg-info", "tests",
                                "node_modules", ".DS_Store", "build_log.jsonl", "*.gz", *TWIN_BINARIES)

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


def stage_package(dest: Path, bucket: str, with_twin: bool = False) -> Path:
    """core/, openenv/ and the task packs (no tests, venvs or binaries) plus fetch_assets.py, into dest."""
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
    (dest / "fetch_assets.py").write_text(FETCH.format(bucket=bucket, twin_files=TWIN_BINARIES, bucket_tasks=BUCKET_TASKS))
    return dest
