"""Fetch immutable runtime dependencies without copying the experiment archive into Git."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"


def run(*args):
    subprocess.run(list(map(str, args)), check=True)


def bootstrap(local_archive=None, local_trl=None, local_openenv=None):
    lock = json.loads((ROOT / "configs/runtime-lock.json").read_text())
    archive = lock["archive"]
    destination = RUNTIME / "archive"

    def fetch(item):
        name, expected = item
        path = destination / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return
        if local_archive:
            data = (Path(local_archive) / "04-data-agent" / name).read_bytes()
        else:
            url = f'https://raw.githubusercontent.com/{archive["repo"]}/{archive["revision"]}/04-data-agent/{name}'
            with urllib.request.urlopen(url, timeout=60) as stream:
                data = stream.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Runtime source hash mismatch: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, archive["files"].items()))
    for name, local in (("trl", local_trl), ("openenv", local_openenv)):
        entry = lock[name]
        target = RUNTIME / name
        if not target.exists():
            run("git", "clone", "--no-checkout", "--filter=blob:none", local or f'https://github.com/{entry["repo"]}.git', target)
        try:
            run("git", "-C", target, "cat-file", "-e", entry["revision"])
        except subprocess.CalledProcessError:
            run("git", "-C", target, "fetch", "origin", entry["revision"])
        run("git", "-C", target, "checkout", "--detach", entry["revision"])
    packages = RUNTIME / "packages"
    packages.mkdir(exist_ok=True)
    for name, folder in (("data_agent_env", "blackbox-opencode"), ("whitebox_bash", "whitebox-bash")):
        link = packages / name
        if link.is_symlink():
            link.unlink()
        shutil.copytree(destination / "envs" / folder, link, dirs_exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from runtime.patches import apply, apply_trl
    apply(packages)
    apply_trl(RUNTIME / "trl")
    print(f"Verified {len(archive['files'])} runtime files; pinned TRL and OpenEnv ready.")


def activate():
    paths = [ROOT, RUNTIME / "packages", RUNTIME / "archive/train",
             RUNTIME / "archive/hf/runtime", RUNTIME / "archive/model_runs/lfm25",
             RUNTIME / "openenv/src", RUNTIME / "openenv/envs", RUNTIME / "trl"]
    if not (RUNTIME / "trl/trl").is_dir():
        raise RuntimeError("Run python runtime/bootstrap.py first")
    sys.path[:0] = list(map(str, paths))
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-archive", type=Path)
    p.add_argument("--local-trl", type=Path)
    p.add_argument("--local-openenv", type=Path)
    a = p.parse_args()
    bootstrap(a.local_archive, a.local_trl, a.local_openenv)
