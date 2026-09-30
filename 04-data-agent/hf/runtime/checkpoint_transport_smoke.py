"""Tiny labeled checkpoint fixture: real Bucket upload, restore and tamper detection."""
import json
import os
from pathlib import Path

from common import ROOT, configure, write_json


def check():
    configure()
    import torch
    from safetensors.torch import save_file
    from huggingface_hub import HfApi
    from checkpoint_artifacts import REQUIRED, mark_saved, resume_info
    from checkpoint_store import seal, restore, verify
    root = ROOT / "outputs" / os.environ["RUN_OWNER"] / "checkpoint-transport-fixture"
    source = root / "source"
    source.mkdir(parents=True, exist_ok=False)
    for name in REQUIRED:
        (source / name).write_text("{}")
    write_json(source / "trainer_state.json", {"global_step": 2})
    write_json(source / "rollout_state.json", {"prompt_index": 3, "model_version": 2})
    torch.save({"state": {0: {"step": torch.tensor(2), "exp_avg": torch.tensor([0.1])}},
                "param_groups": [{"params": [0]}]}, source / "optimizer.pt")
    save_file({"integration_fixture.weight": torch.tensor([1.0, 2.0])}, source / "model.safetensors")
    mark_saved(source, 2, "integration-fixture-only", "fixture-v1")
    seal(source, arm="blackbox", bundle_sha256=os.environ["BUNDLE_SHA256"])
    destination = ("hf://buckets/" + os.environ["ARTIFACT_BUCKET"] + "/" + os.environ["RUN_ID"]
                   + "/preflight/" + os.environ["RUN_OWNER"] + "/checkpoint-transport-fixture")
    HfApi().sync_bucket(str(source), destination, quiet=True)
    target = root / "restored"
    restore(destination, target, arm="blackbox", bundle_sha256=os.environ["BUNDLE_SHA256"])
    resume = resume_info(target, "integration-fixture-only", "fixture-v1")
    assert resume["step"] == 2 and resume["group_offset"] == 3
    optimizer = torch.load(target / "optimizer.pt", weights_only=True)
    assert optimizer["state"][0]["step"].item() == 2
    original = (target / "optimizer.pt").read_bytes()
    (target / "optimizer.pt").write_bytes(original + b"tampered")
    try:
        verify(target)
    except ValueError as exc:
        assert "optimizer.pt" in str(exc)
    else:
        raise AssertionError("Corrupted optimizer checkpoint was accepted")
    (target / "optimizer.pt").write_bytes(original)
    verify(target)
    result = {"passed": True, "fixture_only": True, "real_bucket_roundtrip": True,
              "optimizer_tamper_rejected": True, "native_cursor_restored": True}
    write_json(root / "result.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(check()), flush=True)
