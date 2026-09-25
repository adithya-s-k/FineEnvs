"""Build a portable runtime from pinned Hub data and reviewed Git sources.

The internal snapshot layout stays compatible with the validated runners. No
local experiments checkout is required and previous outputs are archived.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REL_RUN = Path("experiments/daytona_harness_comparison/logs/20260915")
PORTABLE_ROOT = "/workspace/repro"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preserve(path, archive):
    """Move superseded material into the project-local ignored archive."""
    path = Path(path)
    if path.exists():
        archive = Path(archive)
        archive.mkdir(parents=True, exist_ok=True)
        path.rename(archive / f"{path.name}-{time.time_ns()}")


def source_tree(spec, target, cache):
    """Export only the committed revision, without local untracked files or credentials."""
    revision = spec["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Source revisions must be full Git commit hashes")
    checkout = cache / spec["name"]
    if not checkout.exists():
        subprocess.run(["git", "init", "--bare", str(checkout)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(checkout), "fetch", "--depth=1", spec["url"], revision],
                   check=True, capture_output=True)
    archive = cache / f"{spec['name']}-{revision}.tar"
    with archive.open("wb") as stream:
        subprocess.run(["git", "-C", str(checkout), "archive", revision], stdout=stream, check=True)
    target.mkdir(parents=True)
    with tarfile.open(archive) as stream:
        stream.extractall(target, filter="data")


def verify_runtime_entrypoints(stage):
    required = ["hf/runtime/ui_smoke.py", "hf/runtime/coordinator.py", "hf/runtime/job.py",
                "hf/configs/deployment.json", "hf/locks/requirements-env.lock"]
    missing = [name for name in required if not (stage / name).is_file()]
    if missing:
        raise ValueError(f"Bundle is missing runtime entry points: {missing}")


def _copy(source, target):
    if source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(
            "temp", "__pycache__", "*.pyc", ".git", ".env", ".venv", ".pytest_cache", ".gradio"))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def build(out, *, sources=None, token=None, seed_archive=None, config=None):
    from huggingface_hub import hf_hub_download

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = sources or json.loads((HERE / "configs/sources.json").read_text())
    archived = PROJECT / "temp/build-archive"
    stage = out / "stage"
    preserve(stage, archived)
    stage.mkdir()
    seed = sources["task_bundle"]
    path = Path(seed_archive) if seed_archive else Path(hf_hub_download(
        seed["repo"], "bundle.tar.gz", repo_type="dataset", revision=seed["revision"], token=token))
    if sha(path) != seed["sha256"]:
        raise ValueError("Frozen task bundle hash mismatch")
    with tarfile.open(path) as archive:
        archive.extractall(stage, filter="data")
    # Task names, bytes, catalog indices and curriculum order are inherited from
    # this immutable bundle, rather than regenerated from a changing Hub branch.
    run = stage / REL_RUN
    for name, expected in sources.get("data_hashes", {}).items():
        if sha(run / name) != expected:
            raise ValueError("Frozen task manifest changed: " + name)
    cache = PROJECT / "temp/source-cache"
    cache.mkdir(parents=True, exist_ok=True)
    for spec in sources["repositories"]:
        target = run / "source" / spec["directory"]
        preserve(target, archived)
        source_tree(spec, target, cache)
    replacements = [
        (HERE, stage / "hf"),
        (PROJECT / "train", run / "source/HuggingEnvs/04-data-agent/train"),
        (PROJECT / "envs/blackbox-opencode", run / "source/packages/data_agent_env"),
        (PROJECT / "envs/whitebox-bash", run / "source/packages/whitebox_bash"),
        (PROJECT / "tools", stage / "experiments/daytona_harness_comparison/tools"),
    ]
    for source, target in replacements:
        preserve(target, archived)
        _copy(source, target)
    for name in ("eval_concurrent.py", "baseline_checks.py", "eval_pass_at_k.py"):
        _copy(PROJECT / "eval" / name, run / "eval-source" / name)
    _copy(PROJECT / "serve/vllm.sh", run / "eval-source/serve_vllm_tunnel.sh")
    for name in ("smoke_multiharness_tito.py", "audit_multiharness_training.py"):
        _copy(PROJECT / "tools" / name, run / "source/tools" / name)
        if name == "smoke_multiharness_tito.py":
            _copy(PROJECT / "tools" / name, run / "eval-source" / name)
    # The seed stores the historical operator root. Runtime strings are relocated;
    # original manifest contents remain intact to retain their original hashes.
    original_root = sources["historical_runtime_root"]
    for base in (run / "source", run / "eval-source", stage / "experiments/daytona_harness_comparison/tools"):
        for item in base.rglob("*"):
            if item.is_file() and item.suffix in {".py", ".sh"}:
                text = item.read_text()
                relocated = text.replace(original_root, PORTABLE_ROOT)
                if relocated != text:
                    item.write_text(relocated)
    if config is not None:
        shutil.copy2(config, stage / "hf/configs/deployment.json")
    verify_runtime_entrypoints(stage)
    paths = {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob("*"))
             if p.is_file() and p.name != "bundle_manifest.json"}
    manifest = {"schema": 2, "sources": sources, "files": paths}
    (stage / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = out / "bundle.tar.gz"
    preserve(archive, archived)
    with tarfile.open(archive, "w:gz", compresslevel=6) as stream:
        for path in sorted(stage.iterdir()):
            stream.add(path, arcname=path.name)
    metadata = {"sha256": sha(archive), "bytes": archive.stat().st_size, "files": len(paths), "sources": sources}
    preserve(out / "bundle.json", archived)
    preserve(out / "bootstrap.py", archived)
    (out / "bundle.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(HERE / "runtime/bootstrap.py", out / "bootstrap.py")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROJECT / "temp/reproduction/bundle")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--sources", type=Path, default=HERE / "configs/sources.json")
    parser.add_argument("--config", type=Path, help="Deployment config to freeze in the bundle")
    parser.add_argument("--seed-archive", type=Path, help="Use a previously downloaded, hash-verified task bundle")
    args = parser.parse_args()
    from deploy import credentials
    secrets = credentials(args.env_file)
    print(json.dumps(build(args.out, sources=json.loads(args.sources.read_text()),
                           token=secrets["HF_TOKEN"], seed_archive=args.seed_archive, config=args.config)))


if __name__ == "__main__":
    main()
