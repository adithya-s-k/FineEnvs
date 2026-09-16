"""A stop requires exact full-state checkpoint readback, including optimizer state."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stop_hf_after_checkpoint import READY, verify_checkpoint


def make_checkpoint(path):
    files = {"model.safetensors": b"weights", "optimizer.pt": b"optimizer",
             "scheduler.pt": b"scheduler", "rng_state.pth": b"rng",
             "trainer_state.json": b'{"global_step":150}'}
    for name, data in files.items():
        (path / name).write_bytes(data)
    manifest = {"arm": "whitebox", "step": 150, "bundle_sha256": "bundle",
                "base_model": "Qwen/Qwen3.5-2B", "base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
                "files": {n: hashlib.sha256(v).hexdigest() for n, v in files.items()}}
    (path / READY).write_text(json.dumps(manifest))
    return SimpleNamespace(environment={"BUNDLE_SHA256": "bundle"})


def test_complete_state_can_authorize_stop(tmp_path):
    job = make_checkpoint(tmp_path)
    assert verify_checkpoint(tmp_path, job=job, step=150)["step"] == 150


def test_corrupt_optimizer_prevents_stop(tmp_path):
    job = make_checkpoint(tmp_path)
    (tmp_path / "optimizer.pt").write_bytes(b"partial upload")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_checkpoint(tmp_path, job=job, step=150)


def test_wrong_checkpoint_or_bundle_prevents_stop(tmp_path):
    job = make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        verify_checkpoint(tmp_path, job=job, step=200)
    job.environment["BUNDLE_SHA256"] = "other"
    with pytest.raises(ValueError, match="provenance"):
        verify_checkpoint(tmp_path, job=job, step=150)
