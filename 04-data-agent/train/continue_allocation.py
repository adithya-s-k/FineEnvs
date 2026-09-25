"""Resume a frozen multi-harness run after its allocation, without resetting training.

Run ``execute`` in a CPU Slurm job with afterany:<parent> dependency. Preparation
reuses the parent's immutable source and tasks; GPU submission uses its existing
launcher. Explicit cancellation and STOP_AFTER_STEP are respected.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROLES = ("training", "cleanup", "eval_watcher", "monitor", "logging")
RESUMABLE_STATES = {"COMPLETED", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED"}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def checkpoint_validator(parent):
    source = parent / "source-snapshot/HuggingEnvs/04-data-agent/train/checkpoint_artifacts.py"
    spec = importlib.util.spec_from_file_location("continuation_checkpoint_artifacts", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resume_info


def select_checkpoint(parent, job, config, validate=None):
    validate = validate or checkpoint_validator(parent)
    candidates = [p for p in (parent / f"job-{job}/run").glob("checkpoint-*")
                  if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()]
    rejected = []
    for path in sorted(candidates, key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        try:
            resume = validate(path, config["model"], config["model_revision"])
            if resume["step"] != int(path.name.split("-")[-1]):
                raise ValueError("Checkpoint directory and saved step disagree")
            return resume, rejected
        except (ValueError, OSError, KeyError) as exc:
            rejected.append({"checkpoint": str(path), "reason": str(exc)})
    raise ValueError(f"No valid full checkpoint found; rejected={rejected}")


def may_continue(parent, job, state):
    stopped = (parent / f"job-{job}/STOP_AFTER_STEP").exists()
    stopped |= (parent / "operations/allocation-continuation/STOP").exists()
    return not stopped and state in RESUMABLE_STATES


def prepare(parent, output, resume, *, walltime="24:00:00", seconds=82200):
    config = deepcopy(read(parent / "run_config.json"))
    if seconds <= 0 or seconds >= 24 * 3600:
        raise ValueError("The soft training limit must leave room before the 24-hour allocation ends")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("manifest.json", "indices.txt", "harness_schedule.json", "pairs.jsonl",
                 "runtime_versions.json", "source_hashes.json"):
        shutil.copy2(parent / name, output / name)
    if (parent / "schedule_summary.json").exists():
        shutil.copy2(parent / "schedule_summary.json", output / "schedule_summary.json")
    # No working-tree overlays: the continuation runs the identical frozen code.
    snapshot = output / "source-snapshot"
    snapshot.symlink_to((parent / "source-snapshot").resolve(), target_is_directory=True)
    evaluation = output / "checkpoint-evals/eval-source"
    evaluation.parent.mkdir()
    evaluation.symlink_to((parent / "checkpoint-evals/eval-source").resolve(), target_is_directory=True)
    config.update(status="prepared", source_snapshot=str(snapshot), frozen_eval_source=str(evaluation),
                  restart_of=str(parent), resume_state=resume,
                  restart_reason="User-authorized allocation continuation; running-job extension denied by Slurm",
                  initialization="Full checkpoint: model, optimizer, scheduler, RNG and rollout cursor")
    config.pop("job_id", None)
    config.pop("replaced_by", None)
    config.pop("allocation_continuation", None)
    config["training"].update(resume_from_checkpoint=resume["checkpoint"], soft_max_train_seconds=seconds)
    config["resources"]["slurm_walltime"] = walltime
    config["dataset"]["schedule_file"] = str(output / "harness_schedule.json")
    config["evaluation"]["protocol_file"] = str(evaluation / "protocol.json")
    # Preserve the same Trackio project and evaluation curve; training-{job} records
    # distinguish allocation histories while retaining global optimizer step numbers.
    config["logging"].update(local_directory=str(output / "trackio"),
                             collector_file=str(snapshot / "tools/trackio_multi4.py"))
    config["monitoring"]["stable_after_optimizer_step"] = resume["step"] + 3
    config["monitoring"]["support_job_supervisor"].update(
        host="Slurm CPU job", status_file=str(output / "supervisor/status.json"),
        source_file=str(snapshot / "tools/supervise_multi4.py"))
    save(output / "run_config.json", config)
    save(output / "validation.json", {"prepared": True, "resume": resume,
        "parent_validation_file": str(parent / "validation.json"),
        "live_resume_validated": False, "frozen_source_reused": True})
    save(output / "operations/ALLOCATION_CONTINUATION.json", {
        "parent": str(parent), "resume": resume, "frozen_source": str(snapshot.resolve()),
        "changed_training_fields": ["resume_from_checkpoint", "soft_max_train_seconds"],
        "gpu_walltime": walltime, "target_step": config["training"]["max_steps"]})
    return config


def preflight(output):
    env = {**os.environ, "TRAIN_RUN_ROOT": str(output), "MULTI4_PREFLIGHT_ONLY": "1",
           "SLURM_JOB_ID": "preflight"}
    log = output / "operations/continuation-preflight.log"
    with log.open("w") as stream:
        subprocess.run(["bash", str(output / "source-snapshot/tools/launch_multi4_long.sh")],
                       env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def parent_state(job):
    result = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", str(job), "--format=JobIDRaw,State"], text=True)
    for line in result.splitlines():
        fields = line.split("|")
        if fields[0] == str(job):
            return fields[1].split()[0].rstrip("+")
    return "UNKNOWN"


def existing_submission(output):
    path = output / "submission.json"
    if not path.exists():
        return None
    record = read(path)
    if not all(str(record.get(k, "")).isdigit() for k in ROLES):
        raise RuntimeError("Incomplete submission record: reconcile Slurm before retrying; no duplicate GPU job submitted")
    return record


def execute(parent, output, expected_job):
    operations = parent / "operations/allocation-continuation"
    operations.mkdir(parents=True, exist_ok=True)
    status_path = operations / "status.json"
    with (operations / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = read(status_path) if status_path.exists() else {}
        try:
            actual_job = str(read(parent / "submission.json")["training"])
            if actual_job != str(expected_job):
                raise ValueError("Parent training job changed")
            state = parent_state(actual_job)
            if not may_continue(parent, actual_job, state):
                status.update(state="skipped", reason="Parent is not eligible for automatic allocation continuation",
                              parent_state=state)
                save(status_path, status)
                return status
            config = read(parent / "run_config.json")
            resume, rejected = select_checkpoint(parent, actual_job, config)
            if resume["step"] >= config["training"]["max_steps"]:
                status.update(state="complete", reason="Parent already reached target step", resume=resume)
                save(status_path, status)
                return status
            if not output.exists():
                prepare(parent, output, resume)
            else:
                prepared = read(output / "run_config.json")
                if prepared.get("restart_of") != str(parent) or prepared.get("resume_state") != resume:
                    raise ValueError("Existing continuation directory has different provenance")
            status.update(state="prepared", resume=resume, rejected_checkpoints=rejected,
                          output=str(output), parent_job=actual_job)
            save(status_path, status)
            submission = existing_submission(output)
            if submission is None:
                preflight(output)
                env = {k: v for k, v in os.environ.items() if k not in {
                    "MULTI4_PREFLIGHT_ONLY", "TRAIN_RUN_ROOT", "TRAIN_JOB_ID", "CODE_ROOT", "TRAIN_AUDIT_TOOLS"}}
                with (output / "operations/submission.log").open("a") as stream:
                    subprocess.run([sys.executable, str(output / "source-snapshot/tools/submit_multi4_long.py"),
                                    "--run", str(output), "--submit"], env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
                submission = existing_submission(output)
            status.update(state="submitted", submission=submission)
            save(status_path, status)
            if not status.get("supervisor_job"):
                intent = operations / "supervisor-submission.json"
                if intent.exists():
                    raise RuntimeError("Reconcile existing supervisor submission intent before retrying")
                command = [sys.executable, "-u", str(output / "source-snapshot/tools/supervise_multi4.py"),
                           "--run", str(output)]
                save(intent, {"state": "submitting", "command": command})
                job = subprocess.check_output(["sbatch", "--parsable", "--partition=hopper-cpu",
                    "--ntasks=1", "--cpus-per-task=1", "--mem=2G", "--time=48:00:00",
                    "--job-name=multi4-support-supervisor", "--output=/fsx/%u/logs/%x-%j.out",
                    "--error=/fsx/%u/logs/%x-%j.err", "--wrap", shlex.join(command)], text=True).strip().split(";")[0]
                if not job.isdigit():
                    raise RuntimeError("Unrecognized supervisor submission response")
                status["supervisor_job"] = job
                save(intent, {"state": "submitted", "job": job})
            status.update(state="submitted", checked_at=datetime.now(timezone.utc).isoformat())
            save(status_path, status)
            return status
        except Exception as exc:
            status.update(state="error", error=str(exc), checked_at=datetime.now(timezone.utc).isoformat())
            save(status_path, status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "prepare", "execute"])
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parent-job", required=True)
    args = parser.parse_args()
    parent = args.parent.resolve()
    if args.action == "execute":
        if args.output is None:
            parser.error("--output is required")
        result = execute(parent, args.output.resolve(), args.parent_job)
    else:
        config = read(parent / "run_config.json")
        resume, rejected = select_checkpoint(parent, args.parent_job, config)
        result = {"resume": resume, "rejected_checkpoints": rejected}
        if args.action == "prepare":
            if args.output is None:
                parser.error("--output is required")
            prepare(parent, args.output.resolve(), resume)
            preflight(args.output.resolve())
            result.update(prepared=str(args.output.resolve()), preflight_passed=True, submitted=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
