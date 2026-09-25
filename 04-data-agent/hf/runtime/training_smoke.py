"""Validate real native optimizer/save/remote-resume evidence before an HF long run."""
import json
import math
import os
from pathlib import Path

from common import configure, write_json


def validate(output, arm):
    configure()
    from checkpoint_store import verify
    from checkpoint_artifacts import resume_info
    output = Path(output)
    run = output / "run"
    markers = {step: verify(run / f"checkpoint-{step}") for step in (2, 4)}
    for step, marker in markers.items():
        assert marker["step"] == step and marker["arm"] == arm
        assert marker["bundle_sha256"] == os.environ["BUNDLE_SHA256"]
    origin = json.loads((output / "remote-resume/checkpoint-2.remote-origin.json").read_text())
    assert origin == markers[2], "Resume must come from the remotely verified step-2 checkpoint"
    metric_file = output / "audit/metrics.jsonl" if arm in {"blackbox", "opencode"} else run / "metrics.jsonl"
    rows = [json.loads(line) for line in metric_file.read_text().splitlines() if line.strip()]
    updates = [row for row in rows if "grad_norm" in row]
    assert {int(row["step"]) for row in updates} >= {1, 2, 3, 4}
    assert all(math.isfinite(v) for row in updates for v in row.values() if isinstance(v, float))
    assert any(row["grad_norm"] > 0 for row in updates), "No learning signal observed"
    if arm in {"blackbox", "opencode"}:
        restored = output / "remote-resume/checkpoint-2"
        info = resume_info(restored, origin["base_model"], origin["base_revision"])
        expected = f"resume    checkpoint step=2, next schedule group={info['group_offset']}"
        assert expected in (output / "train-resumed.log").read_text()
        from training_capture_audit import audit_async
        audit = audit_async(output / "audit", arm)
        assert audit and all(v["tito_pass"] == v["completed_results"] and
                             v["retained_tokens"] == v["eligible_tokens"] and
                             v["rows_over_token_budget"] == 0 for v in audit.values())
        weights = [name for name in markers[2]["files"] if name.endswith(".safetensors")]
        assert weights and any(markers[2]["files"][name] != markers[4]["files"][name] for name in weights), "Weights did not change after resume"
    else:
        initial = json.loads((run / "optimizer_evidence_from_0.json").read_text())
        resumed = json.loads((run / "optimizer_evidence_from_2.json").read_text())
        assert initial["final_step"] == 2 and resumed["final_step"] == 4
        assert initial["weights_changed"] or resumed["weights_changed"]
        assert initial["final_weight_digest"] == resumed["initial_weight_digest"]
        audit = [json.loads(line) for line in (run / "token_audit.jsonl").read_text().splitlines() if line.strip()]
        assert {row["step"] for row in audit} >= {0, 1, 2, 3}
        assert all(row["tito_pass"] and row["supervised"] > 0 for record in audit for row in record["rows"])
    import torch
    optimizer = torch.load(run / "checkpoint-4/optimizer.pt", map_location="cpu", weights_only=False)
    assert optimizer["state"] and optimizer["param_groups"]
    report = {"arm": arm, "passed": True, "bundle_sha256": os.environ["BUNDLE_SHA256"],
              "optimizer_steps": [1, 2, 3, 4], "native_optimizer_state_verified": True,
              "remote_restore_verified": True, "tito_pass": True, "weights_updated": True,
              "nonzero_gradient_updates": sum(row["grad_norm"] > 0 for row in updates)}
    write_json(output / "training_smoke_verified.json", report)
    return report
