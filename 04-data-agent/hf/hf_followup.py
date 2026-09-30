"""Freeze and run a CPU coordinator for an independently pinned HF Whitebox job.

The trainer/evaluator bundle is never rewritten. The controller is a separately
hashed artifact and carries the qualified Space pin into every evaluation job.
Preparing this controller never starts a long training run.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

HF = Path(__file__).resolve().parent
TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def credentials(path):
    from dotenv import dotenv_values
    values = dotenv_values(path)
    os.environ["HF_TOKEN"] = values.get("HF_API_KEY") or values["HF_TOKEN"]
    for key in ("DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET"):
        if values.get(key):
            os.environ[key] = values[key]


def prepare(args):
    from huggingface_hub import HfApi
    credentials(args.env_file)
    training = HfApi().inspect_job(job_id=args.training_job, namespace="HuggingEnvs")
    if training.labels.get("arm") != "whitebox" or training.labels.get("role") != "train":
        raise ValueError("Expected a native Whitebox HF training job")
    info = read(args.bundle / "bundle.json")
    archive = args.bundle / "bundle.tar.gz"
    if (hashlib.sha256(archive.read_bytes()).hexdigest() != info["sha256"] or
            info["sha256"] != training.environment["BUNDLE_SHA256"]):
        raise ValueError("The archive must match the actual training job")
    volume = next(v for v in training.volumes if v.type == "dataset" and v.mount_path == "/bundle")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = out / "runtime"
    with tarfile.open(archive) as bundle:
        config = json.load(bundle.extractfile("hf/configs/deployment.json"))
        logger_root = out / "logger-runtime"
        for name in ("hf/runtime/common.py", "hf/runtime/checkpoint_store.py", "hf/runtime/logging_sync.py",
                     "experiments/daytona_harness_comparison/logs/20260915/source/tools/trackio_multi4.py",
                     "experiments/daytona_harness_comparison/logs/20260915/source/tools/monitor_multi4.py"):
            target = logger_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.extractfile(name).read())
    config["evaluation"]["concurrency_per_arm"]["whitebox"] = 50
    write(root / "hf/configs/deployment.json", config)
    runtime = root / "hf/runtime"
    runtime.mkdir(parents=True)
    for name in ("common.py", "checkpoint_store.py", "coordinator.py"):
        shutil.copyfile(HF / "runtime" / name, runtime / name)
    shutil.copyfile(Path(__file__), out / "hf_followup.py")
    shutil.copyfile(HF / "late_hf_logging.py", out / "late_hf_logging.py")
    args.coordination_dir.mkdir(parents=True, exist_ok=True)
    plan = {"training_job": training.id, "namespace": "HuggingEnvs", "arm": "whitebox",
        "training_phase": training.labels["phase"], "bundle_sha256": info["sha256"],
        "bundle_repo": volume.source, "bundle_revision": volume.revision,
        "space_bundle_sha256": training.environment.get("SPACE_BUNDLE_SHA256", info["sha256"]),
        "env_file": str(args.env_file.resolve()), "root": str(root),
        "logger_root": str(logger_root), "logging_python": sys.executable,
        "coordination_dir": str(args.coordination_dir.resolve()),
        "environment_python": str(HF.parents[2] / "OpenEnv/.venv/bin/python"),
        "evaluation": {"flavor": config["compute"]["eval_flavor"], "dp": config["compute"]["eval_dp"],
                       "concurrency": 50, "cells": 250, "metric": "pass@1"},
        "controller_walltime": "08:00:00" if training.labels["phase"] == "smoke" else "36:00:00",
        "long_training_submitted": False,
        "files": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in out.rglob("*") if p.is_file()}}
    write(out / "plan.json", plan)
    command = [sys.executable, "-u", str(out / "hf_followup.py"), "watch", "--plan", str(out / "plan.json")]
    script = out / "controller.slurm"
    script.write_text("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n")
    if args.submit:
        job = subprocess.check_output(["sbatch", "--parsable", "--partition=hopper-cpu",
            "--cpus-per-task=2", "--mem=8G", "--time=" + plan["controller_walltime"], "--job-name=cmp-hf-whitebox-followup",
            "--output=" + str(out / "controller-%j.out"), "--error=" + str(out / "controller-%j.err"),
            str(script)], text=True).strip()
        write(out / "submission.json", {"controller_job": job, "training_job": training.id,
              "qualification_only": training.labels["phase"] == "smoke"})
    print(json.dumps({"plan": str(out / "plan.json"), "submitted": args.submit}))


def watch(plan_path):
    from huggingface_hub import HfApi
    plan_path = plan_path.resolve()
    plan = read(plan_path)
    out = plan_path.parent
    for name, expected in plan["files"].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Coordinator source/configuration changed after preparation: " + name)
    credentials(plan["env_file"])
    lock = (Path(plan["coordination_dir"]) / (plan["training_job"] + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ["REPRO_ROOT"] = plan["root"]
    sys.path.insert(0, str(Path(plan["root"]) / "hf/runtime"))
    from checkpoint_store import download_json
    from coordinator import run
    api = HfApi()
    while True:
        training = api.inspect_job(job_id=plan["training_job"], namespace=plan["namespace"])
        write(out / "waiting.json", {"checked_at": time.time(), "job": training.id, "stage": training.status.stage})
        if plan["training_phase"] != "smoke":
            break
        if training.status.stage == "COMPLETED":
            source = ("hf://buckets/" + training.environment["ARTIFACT_BUCKET"] + "/" +
                training.environment["RUN_ID"] + "/jobs/" + training.environment["RUN_OWNER"])
            proof = download_json(source, "training_smoke_verified.json", out / "training_smoke_verified.json", api)
            if (proof.get("arm") != "whitebox" or proof.get("bundle_sha256") != plan["bundle_sha256"] or
                    proof.get("optimizer_steps") != [1, 2, 3, 4] or not all(proof.get(k) for k in
                    ("passed", "weights_updated", "tito_pass", "remote_restore_verified", "native_optimizer_state_verified"))):
                raise ValueError("HF optimizer qualification did not pass")
            os.environ["QUALIFY_CHECKPOINT_STEP"] = "4"
            break
        if training.status.stage in TERMINAL:
            raise RuntimeError("HF optimizer smoke ended without successful qualification: " + training.status.stage)
        time.sleep(60)
    if training.environment["BUNDLE_SHA256"] != plan["bundle_sha256"]:
        raise ValueError("Training runtime identity changed")
    os.environ.update(TRAINING_JOB=training.id, ARTIFACT_BUCKET=training.environment["ARTIFACT_BUCKET"],
        RUN_ID=training.environment["RUN_ID"], RUN_OWNER="cpu-hf-followup-" + training.id,
        BUNDLE_SHA256=plan["bundle_sha256"], BUNDLE_REPO=plan["bundle_repo"],
        BUNDLE_REVISION=plan["bundle_revision"])

    @contextmanager
    def admission():
        with (Path(plan["coordination_dir"]) / "eval-admission.lock").open("a") as capacity:
            fcntl.flock(capacity, fcntl.LOCK_EX)
            slurm = subprocess.check_output(["squeue", "--noheader", "--user", os.environ["USER"], "--format=%j"], text=True)
            active_hf = any(j.status.stage not in TERMINAL for j in api.list_jobs(namespace=plan["namespace"],
                labels={"experiment": "data-agent-daytona", "role": "eval"}))
            if active_hf or any(name.strip().startswith("cmp-eval-") for name in slurm.splitlines()):
                yield False
            else:
                code = "from daytona import Daytona; print(len(list(Daytona().list())))"
                occupied = int(subprocess.check_output([plan["environment_python"], "-c", code], text=True).strip())
                yield occupied + 50 + 24 <= 125

    run(out / "output", "whitebox", admission=admission)
    if plan.get("logger_root"):
        subprocess.run([plan["logging_python"], str(out / "late_hf_logging.py"),
                        "--plan", str(plan_path)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--training-job", required=True)
    preparation.add_argument("--bundle", type=Path, required=True)
    preparation.add_argument("--env-file", type=Path, required=True)
    preparation.add_argument("--out", type=Path, required=True)
    preparation.add_argument("--coordination-dir", type=Path, required=True)
    preparation.add_argument("--submit", action="store_true")
    monitoring = commands.add_parser("watch")
    monitoring.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.action == "prepare" else watch(args.plan)
