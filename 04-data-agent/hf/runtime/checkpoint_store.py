"""Portable full-checkpoint integrity, including optimizer, RNG and rollout cursor.

Native completion markers bind filesystem paths/mtimes. A remote restore first
verifies every file by content, then explicitly rebinds those native markers.
"""
import hashlib
import json
from pathlib import Path

from common import write_json

READY = "checkpoint.hf.ready.json"


def bucket_location(uri):
    prefix = "hf://buckets/"
    if not uri.startswith(prefix):
        raise ValueError("Expected an HF Bucket URI")
    parts = uri[len(prefix):].strip("/").split("/")
    if len(parts) < 3 or any(p in {"", ".", ".."} for p in parts):
        raise ValueError("Invalid artifact URI")
    return "/".join(parts[:2]), "/".join(parts[2:])


def download_json(source, name, target, api=None):
    from huggingface_hub import HfApi
    if Path(name).name != name:
        raise ValueError("Expected a single artifact filename")
    bucket, prefix = bucket_location(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    (api or HfApi()).download_bucket_files(bucket,
        files=[(prefix + "/" + name, str(target))], raise_on_missing_files=True)
    return json.loads(target.read_text())


def restore_model(source, target, *, arm, bundle_sha256, manifest_sha256):
    """Download hash-verified inference files without transferring optimizer state."""
    from huggingface_hub import HfApi
    from checkpoint_artifacts import METADATA, model_files
    from common import MODEL, REVISION
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    api = HfApi()
    manifest = download_json(source, READY, target / READY, api)
    if digest(target / READY) != manifest_sha256:
        raise ValueError("Checkpoint manifest changed after evaluation was queued")
    if (manifest["arm"] != arm or manifest["bundle_sha256"] != bundle_sha256 or
            manifest["base_model"] != MODEL or manifest["base_revision"] != REVISION):
        raise ValueError("Checkpoint evaluation provenance mismatch")
    names = [name for name in manifest["files"] if name in METADATA or name.endswith(".safetensors")]
    if any(Path(name).name != name for name in names):
        raise ValueError("Invalid checkpoint member")
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(names):
        raise ValueError("Checkpoint lacks model or tokenizer metadata")
    bucket, prefix = bucket_location(source)
    api.download_bucket_files(bucket, files=[(prefix + "/" + n, str(target / n)) for n in names],
                              raise_on_missing_files=True)
    for name in names:
        if digest(target / name) != manifest["files"][name]:
            raise ValueError(f"Checkpoint model hash mismatch: {name}")
    if not model_files(target):
        raise ValueError("No model weights found")
    write_json(target / "evaluation_source.json", {"source": source, "manifest_sha256": manifest_sha256,
               "arm": arm, "step": manifest["step"], "bundle_sha256": bundle_sha256,
               "files": {name: manifest["files"][name] for name in names}})
    return manifest


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def seal(checkpoint, *, arm, bundle_sha256):
    from checkpoint_artifacts import REQUIRED, finalize_saved
    checkpoint = Path(checkpoint)
    native = finalize_saved(checkpoint)
    required = set(REQUIRED)
    if arm in {"blackbox", "opencode"}:
        required.add("rollout_state.json")
    for name in required:
        if not (checkpoint / name).is_file():
            raise ValueError(f"Incomplete full checkpoint: {name}")
    paths = sorted(p for p in checkpoint.iterdir() if p.is_file() and p.name != READY)
    if any(p.is_symlink() for p in paths):
        raise ValueError("Checkpoint must contain actual files")
    before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    files = {p.name: digest(p) for p in paths}
    after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    if before != after:
        raise ValueError("Checkpoint changed while hashing")
    manifest = {"schema": 1, "arm": arm, "step": native["step"],
                "base_model": native["base_model"], "base_revision": native["base_revision"],
                "bundle_sha256": bundle_sha256, "files": files,
                "required_training_state": sorted(required), "final": native.get("final", False)}
    write_json(checkpoint / READY, manifest)
    return manifest


def verify(checkpoint):
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / READY).read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name or (checkpoint / name).is_symlink():
            raise ValueError("Invalid checkpoint member")
        if digest(checkpoint / name) != expected:
            raise ValueError(f"Remote checkpoint content mismatch: {name}")
    if not set(manifest["required_training_state"]).issubset(manifest["files"]):
        raise ValueError("Checkpoint lacks complete training-state hashes")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if state["global_step"] != manifest["step"]:
        raise ValueError("Checkpoint optimizer step mismatch")
    return manifest


def restore(source, target, *, arm, bundle_sha256):
    from huggingface_hub import HfApi
    from checkpoint_artifacts import mark_saved, finalize_saved
    target = Path(target)
    if target.exists():
        raise ValueError("Restore target must be new")
    target.mkdir(parents=True)
    HfApi().sync_bucket(source, str(target), quiet=True)
    manifest = verify(target)
    if manifest["arm"] != arm or manifest["bundle_sha256"] != bundle_sha256:
        raise ValueError("Remote checkpoint provenance mismatch")
    # Preserve the verified source manifest separately before rebinding native paths.
    write_json(target.parent / (target.name + ".remote-origin.json"), manifest)
    mark_saved(target, manifest["step"], manifest["base_model"], manifest["base_revision"],
               final=manifest.get("final", False))
    finalize_saved(target)
    # Re-seal the rebound copy so its complete integrity check also remains usable.
    seal(target, arm=arm, bundle_sha256=bundle_sha256)
    return manifest
