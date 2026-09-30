"""Portable checkpoint markers. The evaluator verifies bytes before loading weights."""
import hashlib
import json
from pathlib import Path

from recipe import write_json

REQUIRED = ("config.json", "trainer_state.json", "tokenizer_config.json", "tokenizer.json",
            "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def saved(path, cfg, step):
    path = Path(path)
    for name in REQUIRED:
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise ValueError(f"Incomplete checkpoint: {name}")
    if not list(path.glob("*.safetensors")):
        raise ValueError("Checkpoint has no model weights")
    if json.loads((path / "trainer_state.json").read_text())["global_step"] != step:
        raise ValueError("Checkpoint step mismatch")
    files = {p.name: p.stat().st_size for p in path.iterdir() if p.is_file() and not p.name.startswith(".") and
             p.name not in {"checkpoint.saved.json", "checkpoint.ready.json", "eval.request.json", "recipe.json"}}
    write_json(path / "checkpoint.saved.json", {"step": step, "base_model": cfg["profile"]["id"],
               "base_revision": cfg["profile"]["revision"], "files": files})


def make_ready(path):
    path = Path(path)
    marker = json.loads((path / "checkpoint.saved.json").read_text())
    hashes = {}
    for name, size in marker["files"].items():
        if Path(name).name != name or (path / name).stat().st_size != size:
            raise ValueError("Checkpoint incomplete or changed after save")
        hashes[name] = sha256(path / name)
        if (path / name).stat().st_size != size:
            raise ValueError("Checkpoint changed during hashing")
    write_json(path / "checkpoint.ready.json", {**marker, "sha256": hashes})


def verify(path, *, resume=False):
    path = Path(path)
    marker = json.loads((path / "checkpoint.ready.json").read_text())
    names = marker["sha256"]
    for name, expected in names.items():
        if Path(name).name != name or sha256(path / name) != expected:
            raise ValueError(f"Checkpoint integrity check failed: {name}")
    if resume and any(n not in names for n in REQUIRED):
        raise ValueError("Resume requires a complete optimizer checkpoint")
    return marker
