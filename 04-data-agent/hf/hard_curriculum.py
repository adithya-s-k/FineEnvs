"""Prepare and explicitly launch the hard-task continuation on HF Jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
DEFAULT_RUN = WORKSPACE / "experiments/async_grpo_harbor_data_agent/logs/multi4-hard500-2epochs-from500-20260917"
PORTABLE = Path("/workspace/repro")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(run, out):
    if out.exists():
        raise ValueError("Preserve the existing bundle; use a new --out directory")
    c = read(run / "config.proposed.json")
    if c["dataset"]["total_groups"] != 1000 or c["dataset"]["difficulty_counts"] != {"easy": 0, "medium": 0, "hard": 500}:
        raise ValueError("Expected the reviewed 500-hard-task, two-pass configuration")
    stage = out / "stage"
    stage.mkdir(parents=True)
    logs = WORKSPACE / "experiments/async_grpo_harbor_data_agent/logs"
    source = logs / "multi4-long-prod-20260915/source-snapshot"
    ignore = shutil.ignore_patterns(".git", ".env", "*.env", ".venv", "__pycache__", "*.pyc", "node_modules")
    for name in ("trl", "OpenEnv", "HuggingEnvs"):
        shutil.copytree(source / name, stage / "source" / name, ignore=ignore)
    for name in ("pyproject.toml", "VERSION", "README.md", "LICENSE"):
        if not (stage / "source/trl" / name).exists():
            shutil.copy2(WORKSPACE / "trl" / name, stage / "source/trl" / name)
    train = stage / "source/HuggingEnvs/04-data-agent/train"
    shutil.copy2(HERE.parent / "train/atomic_rollouts.py", train / "atomic_rollouts.py")
    for name in ("common.py", "artifacts.py", "checkpoint_store.py", "telemetry.py",
                 "hard_curriculum_train.py", "hard_curriculum_job.py", "hard_curriculum_bootstrap.py", "hard_curriculum_logging.py", "hard_kernel_diagnostic.py", "training_capture_audit.py"):
        target = stage / "hf/runtime" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / "runtime" / name, target)
    for name in ('trackio_multi4.py','monitor_multi4.py'):
        shutil.copy2(WORKSPACE / 'experiments/async_grpo_harbor_data_agent/tools' / name, stage / 'hf/runtime' / name)
    shutil.copytree(HERE / "locks", stage / "hf/locks")
    shutil.copy2(HERE / "hard_curriculum.py", stage / "hf/hard_curriculum.py")
    # Reuse the evaluated capture transport while retaining the parent trainer.
    evaluation = logs / "multi4-long-prod-cont-20260915/checkpoint-evals/eval-source"
    for name in ("eval_concurrent.py", "eval_pass_at_k.py", "harnesses_supported.txt",
                 "smoke_multiharness_tito.py", "baseline_checks.py"):
        target = stage / "eval" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(evaluation / name, target)
    for name in ("server.py", "sse.py"):
        rel = Path("OpenEnv/src/openenv/core/harness/capture") / name
        shutil.copy2(evaluation / rel, stage / "source" / rel)
    # vLLM returns an empty body for successful cache-reset control requests.
    client = stage / "source/trl/trl/generation/vllm_client.py"
    before = '        self._post(f"{self.base_url}/reset_prefix_cache")'
    replacement = '        response = self.session.post(f"{self.base_url}/reset_prefix_cache")\n        response.raise_for_status()'
    if before in client.read_text():
        client.write_text(client.read_text().replace(before, replacement))
    inputs = stage / "inputs"
    inputs.mkdir()
    for name in ("manifest.json", "selection.json", "overlap_validation.json", "harness_schedule.json", "execution_schedule.jsonl"):
        shutil.copy2(run / name, inputs / name)
    shutil.copy2(run / "task_bundle/indices.txt", inputs / "train_indices.txt")
    shutil.copytree(Path(c["dataset"]["split"]), stage / "data/train")
    protocol = read(c["evaluation"]["protocol_file"])
    baseline = Path(protocol["split"]).parent
    shutil.copytree(Path(protocol["split"]), stage / "data/test")
    shutil.copy2(baseline / "manifest.json", inputs / "test_manifest.json")
    shutil.copy2(baseline / "indices.txt", inputs / "test_indices.txt")
    c["dataset"].update(split=str(PORTABLE / "data/train"), manifest=str(PORTABLE / "inputs/manifest.json"),
        schedule_file=str(PORTABLE / "inputs/harness_schedule.json"),
        execution_schedule_file=str(PORTABLE / "inputs/execution_schedule.jsonl"))
    c["resources"] = {"provider": "hf_jobs", "namespace": "HuggingEnvs", "flavor": "h200x2", "timeout": "72h",
                      "trainer_gpus": 1, "training_inference_gpus": 1}
    c["evaluation"].update(partition=None, flavor="a100-large", timeout="4h", concurrency=50, tp=1, dp=1)
    c["hf"] = {"namespace": "HuggingEnvs", "bundle_repo": "HuggingEnvs/data-agent-hard500-repro",
        "artifact_bucket": "HuggingEnvs/data-agent-daytona-artifacts", "run_id": run.name,
        "image": read(HERE / "configs/deployment.json")["compute"]["bootstrap_image"],
        "coordinator_flavor": "cpu-upgrade", "coordinator_timeout": "96h",
        "parent_prefix": "hf://buckets/HuggingEnvs/data-agent-daytona-artifacts/" + run.name + "/parent-500",
        "secret_names": ["HF_TOKEN", "E2B_API_KEY"], "environment_service": "job_local_harbor",
        "existing_spaces_modified": False, "train_hourly_usd": 10.0, "eval_hourly_usd": 2.5}
    c["status"] = "HF setup prepared; live GPU qualification and launch approval required"
    c['logging']['project'] = 'qwen35-2b-harbor-vs-opencode-20260916'
    write(stage / "config.json", c)
    parent = Path(c["initialization"]["checkpoint"])
    for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "rollout_state.json"):
        if not (parent / name).is_file():
            raise ValueError(f"Parent checkpoint is incomplete: {name}")
    parent_manifest = {"step": 500, "source": str(parent), "model": c["model"], "model_revision": c["model_revision"],
        "files": {p.name: digest(p) for p in sorted(parent.iterdir()) if p.is_file()},
        "bytes": sum(p.stat().st_size for p in parent.iterdir() if p.is_file())}
    write(inputs / "parent_manifest.json", parent_manifest)
    paths = {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    write(stage / "bundle_manifest.json", {"schema_version": 1, "files": paths})
    archive = out / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(stage.iterdir()):
            tar.add(path, arcname=path.name)
    write(out / "bundle.json", {"sha256": digest(archive), "files": len(paths), "bytes": archive.stat().st_size})
    shutil.copy2(HERE / "runtime/hard_curriculum_bootstrap.py", out / "bootstrap.py")
    write(out / "config.json", c)
    print(json.dumps({"prepared": str(out), "bundle": read(out / "bundle.json"), "parent_bytes": parent_manifest["bytes"]}))


def upload(api, out):
    c = read(out / "config.json")
    parent_manifest = read(out / "stage/inputs/parent_manifest.json")
    parent = Path(parent_manifest["source"])
    for name, expected in parent_manifest["files"].items():
        if digest(parent / name) != expected:
            raise ValueError(f"Parent changed after packaging: {name}")
    api.sync_bucket(str(parent), c["hf"]["parent_prefix"], quiet=True)
    repo = c["hf"]["bundle_repo"]
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    commit = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=out,
        allow_patterns=["bundle.tar.gz", "bundle.json", "bootstrap.py"], commit_message="Freeze hard500 continuation runtime and task manifests")
    receipt = {**read(out / "bundle.json"), "repo": repo, "revision": commit.oid,
               "parent_prefix": c["hf"]["parent_prefix"], "uploaded_at": time.time()}
    write(out / "uploaded.json", receipt)
    return receipt


def request(out, phase, uploaded=None):
    c, b = read(out / "config.json"), read(out / "bundle.json")
    role = "preflight" if phase == "preflight" else "train"
    flavor = "cpu-upgrade" if role == "preflight" else c["resources"]["flavor"]
    timeout = "2h" if phase != "long" else "72h"
    owner = f"{c['hf']['run_id']}-{phase}-{b['sha256'][:8]}"
    return {"namespace": c["hf"]["namespace"], "image": c["hf"]["image"], "flavor": flavor,
        "timeout": timeout, "name": owner,
        "command": ["python", "/bundle/bootstrap.py", "--role", role, "--phase", phase],
        "env": {"RUN_OWNER": owner, "RUN_ID": c["hf"]["run_id"], "BUNDLE_SHA256": b["sha256"],
            "ARTIFACT_BUCKET": c["hf"]["artifact_bucket"], "COMPARISON_ARM": "blackbox",
            "BUNDLE_REPO": c["hf"]["bundle_repo"], "BUNDLE_REVISION": (uploaded or {}).get("revision", "UPLOAD_REQUIRED"),
            "TRACKIO_SPACE": c["logging"]["space_id"], "PYTHONUNBUFFERED": "1"},
        "labels": {"experiment": "harbor-hard500", "role": role, "phase": phase, "run": c["hf"]["run_id"]}}


def submit(api, secrets, out, phase):
    from huggingface_hub import Volume
    uploaded = read(out / "uploaded.json")
    if digest(out / "bundle.tar.gz") != uploaded["sha256"]:
        raise ValueError("Uploaded runtime differs from local bundle")
    req = request(out, phase, uploaded)
    if phase == "long":
        proof = read(out / "qualification.json")
        job = api.inspect_job(job_id=proof["job_id"], namespace=req["namespace"])
        if (job.status.stage != "COMPLETED" or job.labels.get("phase") != "smoke"
                or job.environment.get("BUNDLE_SHA256") != uploaded["sha256"]):
            raise ValueError("A completed GPU qualification on this exact bundle is required")
        prefix = req["env"]["RUN_ID"] + "/jobs/" + job.environment["RUN_OWNER"]
        api.download_bucket_files(req["env"]["ARTIFACT_BUCKET"],
            files=[(prefix + "/qualification.json", str(out / "qualification.remote.json"))], raise_on_missing_files=True)
        proof = read(out / "qualification.remote.json")
        required = ["passed", "finite_schedule_exhausted", "remote_optimizer_resume", "tito_pass", "eval_smoke_passed", "training_capture_optimizer_audit", "weights_updated", "trackio_remote_verified", "gpu_kernel_preflight"]
        if proof.get("bundle_sha256") != uploaded["sha256"] or not all(proof.get(k) is True for k in required):
            raise ValueError("GPU qualification is incomplete")
    intent = out / f"submission-{phase}.json"
    if intent.exists():
        raise RuntimeError("Submission already attempted; reconcile the saved intent instead of duplicating jobs")
    write(intent, {"state": "submitting", "request": req, "created_at": time.time()})
    job = api.run_job(**req, secrets={k: secrets[k] for k in ["HF_TOKEN", "E2B_API_KEY"]},
        volumes=[Volume(type="dataset", source=uploaded["repo"], revision=uploaded["revision"], mount_path="/bundle", read_only=True)])
    write(intent, {"state": "submitted", "job_id": job.id, "url": job.url, "request": req})
    if phase == "long":
        control = {**req, "name": req["name"] + "-evals", "flavor": "cpu-upgrade", "timeout": "96h",
            "command": ["python", "/bundle/bootstrap.py", "--role", "coordinator", "--phase", "long"],
            "env": {**req["env"], "TRAINING_JOB": job.id, "RUN_OWNER": req["name"] + "-evals"},
            "labels": {**req["labels"], "role": "coordinator", "training_job": job.id}}
        write(out / "submission-coordinator.json", {"state": "submitting", "request": control})
        cpu = api.run_job(**control, secrets={k: secrets[k] for k in ["HF_TOKEN", "E2B_API_KEY"]},
            volumes=[Volume(type="dataset", source=uploaded["repo"], revision=uploaded["revision"], mount_path="/bundle", read_only=True)])
        write(out / "submission-coordinator.json", {"state": "submitted", "job_id": cpu.id, "url": cpu.url})
    return {"job_id": job.id, "url": job.url}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "preview", "upload", "submit"])
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=["preflight", "smoke", "long"], default="smoke")
    p.add_argument("--env-file", type=Path, default=WORKSPACE / "experiments/.env")
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.run.resolve(), args.out.resolve())
        return
    if args.action == "preview":
        value = request(args.out, args.phase, read(args.out / "uploaded.json") if (args.out / "uploaded.json").exists() else None)
        write(args.out / f"preview-{args.phase}.json", value)
    else:
        from deploy import credentials
        from huggingface_hub import HfApi
        secrets = credentials(args.env_file)
        api = HfApi(token=secrets["HF_TOKEN"])
        value = upload(api, args.out) if args.action == "upload" else submit(api, secrets, args.out, args.phase)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
