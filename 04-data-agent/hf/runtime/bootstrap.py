"""HF Job entry point; only Python stdlib is needed before creating the locked venvs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", default="/bundle")
    p.add_argument("--role", required=True, choices=["preflight", "eval", "train", "coordinator"])
    args, rest = p.parse_known_args()
    bundle = Path(args.bundle_dir)
    archive = bundle / "bundle.tar.gz"
    info = json.loads((bundle / "bundle.json").read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == info["sha256"]
    root = Path("/workspace/repro")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    os.environ["REPRO_ROOT"] = str(root)
    os.environ["BUNDLE_SHA256"] = info["sha256"]
    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
    os.environ.setdefault("UV_CACHE_DIR", "/workspace/uv-cache")
    for name, target in [("env", root / "OpenEnv/.venv"), ("train", root / ".venv312")]:
        if args.role == "coordinator" and name == "train":
            continue
        subprocess.run(["uv", "venv", "--python", "3.12", str(target)], check=True)
        subprocess.run(["uv", "pip", "sync", "--python", str(target / "bin/python"),
                        "--require-hashes", str(root / f"hf/locks/requirements-{name}.lock")], check=True)
    if args.role != "coordinator":
        # Register the vendored TRL project itself, without resolving or changing
        # any locked dependencies. Merely adding its source to PYTHONPATH leaves
        # importlib.metadata empty and fails at the first checkpoint model card.
        project = root / "experiments/daytona_harness_comparison/logs/20260915/source/trl"
        subprocess.run(["uv", "pip", "install", "--python", str(root / ".venv312/bin/python"),
                        "--no-deps", "--no-build-isolation", "--editable", str(project)], check=True)
    script = "preflight.py" if args.role == "preflight" else "job.py"
    python = root / ".venv312/bin/python"
    if args.role == "coordinator":
        python = root / "OpenEnv/.venv/bin/python"
    os.execv(str(python), [str(python), "-u", str(root / "hf/runtime" / script), "--role", args.role, *rest])


if __name__ == "__main__":
    main()
