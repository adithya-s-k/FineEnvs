import json
from pathlib import Path

import pytest

import continue_allocation as continuation


def parent_run(tmp_path):
    root = tmp_path / "parent"
    root.mkdir()
    config = {
        "model": "model", "model_revision": "revision",
        "training": {"max_steps": 1000, "save_steps": 50, "soft_max_train_seconds": 42000,
                     "learning_rate": 3e-6, "gradient_accumulation_steps": 4, "num_generations": 8},
        "resources": {"slurm_walltime": "12:00:00"},
        "dataset": {"schedule_sha256": "fixed"},
        "evaluation": {"interval_optimizer_steps": 100, "concurrency": 100},
        "logging": {"project": "existing-project", "space_id": "existing-space"},
        "monitoring": {"support_job_supervisor": {}},
    }
    continuation.save(root / "run_config.json", config)
    continuation.save(root / "submission.json", {"training": "123"})
    for name in ("manifest.json", "indices.txt", "harness_schedule.json", "pairs.jsonl",
                 "runtime_versions.json", "source_hashes.json"):
        (root / name).write_text("{}")
    (root / "source-snapshot").mkdir()
    (root / "checkpoint-evals/eval-source").mkdir(parents=True)
    return root, config


def test_latest_incomplete_checkpoint_is_never_selected(tmp_path):
    root, config = parent_run(tmp_path)
    for step in (450, 500):
        (root / f"job-123/run/checkpoint-{step}").mkdir(parents=True)

    def validate(path, model, revision):
        if path.name == "checkpoint-500":
            raise ValueError("optimizer save is incomplete")
        return {"step": 450, "checkpoint": str(path)}

    resume, rejected = continuation.select_checkpoint(root, "123", config, validate)
    assert resume["step"] == 450
    assert len(rejected) == 1


def test_prepare_preserves_training_and_frozen_sources(tmp_path):
    root, config = parent_run(tmp_path)
    target = tmp_path / "next"
    resume = {"step": 650, "checkpoint": str(root / "job-123/run/checkpoint-650")}
    prepared = continuation.prepare(root, target, resume)
    for key, value in config["training"].items():
        if key != "soft_max_train_seconds":
            assert prepared["training"][key] == value
    assert prepared["training"]["soft_max_train_seconds"] == 82200
    assert prepared["resources"]["slurm_walltime"] == "24:00:00"
    assert prepared["logging"]["project"] == config["logging"]["project"]
    assert prepared["evaluation"]["interval_optimizer_steps"] == 100
    assert (target / "source-snapshot").resolve() == root / "source-snapshot"
    assert (target / "checkpoint-evals/eval-source").resolve() == root / "checkpoint-evals/eval-source"
    assert continuation.read(root / "run_config.json") == config


def test_cancellation_and_explicit_stop_are_respected(tmp_path):
    root, _ = parent_run(tmp_path)
    assert continuation.may_continue(root, "123", "TIMEOUT")
    assert not continuation.may_continue(root, "123", "CANCELLED")
    assert not continuation.may_continue(root, "123", "RUNNING")
    (root / "job-123").mkdir()
    (root / "job-123/STOP_AFTER_STEP").touch()
    assert not continuation.may_continue(root, "123", "COMPLETED")


def test_partial_submission_requires_reconciliation(tmp_path):
    continuation.save(tmp_path / "submission.json", {"training": "456"})
    with pytest.raises(RuntimeError, match="no duplicate GPU"):
        continuation.existing_submission(tmp_path)


def test_completed_target_submits_nothing(tmp_path, monkeypatch):
    root, _ = parent_run(tmp_path)
    monkeypatch.setattr(continuation, "parent_state", lambda job: "COMPLETED")
    monkeypatch.setattr(continuation, "select_checkpoint", lambda *args: ({"step": 1000}, []))
    result = continuation.execute(root, tmp_path / "next", "123")
    assert result["state"] == "complete"
    assert not (tmp_path / "next").exists()


def test_completed_submission_is_not_duplicated(tmp_path, monkeypatch):
    root, _ = parent_run(tmp_path)
    target = tmp_path / "next"
    resume = {"step": 650, "checkpoint": str(root / "job-123/run/checkpoint-650")}
    continuation.prepare(root, target, resume)
    jobs = dict(zip(continuation.ROLES, map(str, range(456, 461))))
    continuation.save(target / "submission.json", jobs)
    continuation.save(root / "operations/allocation-continuation/status.json", {"supervisor_job": "461"})
    monkeypatch.setattr(continuation, "parent_state", lambda job: "COMPLETED")
    monkeypatch.setattr(continuation, "select_checkpoint", lambda *args: (resume, []))
    monkeypatch.setattr(continuation.subprocess, "run", lambda *a, **kw: pytest.fail("duplicate submission"))
    monkeypatch.setattr(continuation.subprocess, "check_output", lambda *a, **kw: pytest.fail("duplicate supervisor"))
    assert continuation.execute(root, target, "123")["submission"] == jobs
