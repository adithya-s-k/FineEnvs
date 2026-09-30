"""Portable paths and small process helpers shared by Spaces and Jobs."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("REPRO_ROOT", "/workspace/repro"))
RUN = ROOT / "experiments/daytona_harness_comparison/logs/20260915"
TOOLS = ROOT / "experiments/daytona_harness_comparison/tools"
TRAIN_PY = ROOT / ".venv312/bin/python"
ENV_PY = ROOT / "OpenEnv/.venv/bin/python"
MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def configure():
    paths = [ROOT / "hf/runtime", TOOLS, RUN / "source/tools", RUN / "source/packages",
             RUN / "source/HuggingEnvs/04-data-agent/train", RUN / "source/OpenEnv/src",
             RUN / "source/OpenEnv/envs", RUN / "source/trl", RUN / "eval-source"]
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    os.environ["DAYTONA_COMPARISON_RUN"] = str(RUN)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def verify_bundle():
    manifest = json.loads((ROOT / "bundle_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Bundle file changed: {name}")
    return len(manifest["files"])


def start(command, log, env=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open("a")
    process = subprocess.Popen(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True, cwd=ROOT)
    stream.close()
    return process


def ready(url, process=None, headers=None, seconds=1200):
    import httpx
    deadline = time.monotonic() + seconds
    with httpx.Client(timeout=10, headers=headers, follow_redirects=True) as client:
        while time.monotonic() < deadline:
            if process and process.poll() is not None:
                raise RuntimeError(f"Service exited before readiness: returncode={process.returncode}")
            try:
                if client.get(url).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
    raise TimeoutError(f"Readiness deadline exceeded: {url}")
