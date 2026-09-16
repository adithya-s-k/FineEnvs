"""Reproducible private HuggingEnvs deployment. Never reads credentials into config files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "temp/reproduction"


def credentials(env_file):
    from dotenv import dotenv_values
    values = dotenv_values(env_file) if env_file else {}
    # Explicit file credential takes precedence over the desktop's restricted OAuth token.
    token = values.get("HF_API_KEY") or values.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN or an --env-file containing HF_API_KEY is required")
    result = {"HF_TOKEN": token}
    for key in ("DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET", "E2B_API_KEY", "OPENAI_API_KEY"):
        if values.get(key) or os.environ.get(key):
            result[key] = values.get(key) or os.environ[key]
    return result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def ensure_repo(api, repo, kind, **kwargs):
    api.create_repo(repo, repo_type=kind, private=True, exist_ok=True, **kwargs)
    info = api.repo_info(repo, repo_type=kind)
    if not info.private and not (kind == "space" and getattr(info, "protected", False)):
        raise RuntimeError(f"Existing resource is public: {repo}")


def upload_bundle(api, config, out):
    repo = config["resources"]["repro_dataset_repo"]
    ensure_repo(api, repo, "dataset")
    bundle = out / "bundle"
    metadata = json.loads((bundle / "bundle.json").read_text())
    (bundle / "README.md").write_text("# Daytona data-agent reproduction\n\nPrivate frozen runtime, 1,000 training tasks, 250 evaluation tasks and baseline evidence.\n\nSee the HF launch scripts in the archived `hf/` directory. No credentials are included.\n")
    commit = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=bundle,
                              allow_patterns=["bundle.tar.gz", "bundle.json", "bootstrap.py", "README.md"],
                              commit_message="Package frozen Daytona source, tasks and HF runtime")
    value = {**metadata, "repo": repo, "revision": commit.oid}
    save(out / "bundle_uploaded.json", value)
    return value


def spaces(api, config, secrets, out, selected):
    bundle = out / "bundle"
    info = json.loads((bundle / "bundle.json").read_text())
    for arm in config["resources"]["environment_spaces"]:
        name = arm
        mode = "shared"
        if selected and name not in selected:
            continue
        repo = config["resources"]["environment_spaces"][name]
        ensure_repo(api, repo, "space", space_sdk="docker")
        for key, value in secrets.items():
            api.add_space_secret(repo, key, value)
        variables = {"COMPARISON_ARM": arm, "COMPARISON_MODE": mode,
                     "BUNDLE_SHA256": info["sha256"],
                     "MAX_CONCURRENT_ENVS": "1024",
                     "SANDBOX_CAPACITY": str(config["serving"]["shared_service"]["sandbox_capacity_per_arm"][arm]),
                     "TRAIN_RESERVED_SANDBOXES": str(config["serving"]["shared_service"]["train_reserved_per_arm"][arm]),
                     "OPENENV_HARBOR_AGENT_VERSIONS": json.dumps(config["harness_pins"])}
        for key, value in variables.items():
            api.add_space_variable(repo, key, value)
        stage = out / "spaces" / name
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / "bundle.tar.gz", stage / "bundle.tar.gz")
        # Deploy dependencies from the selected frozen bundle, even if local work
        # has since added a UI dependency for the next deployment.
        with tarfile.open(bundle / "bundle.tar.gz") as archive:
            (stage / "requirements-env.lock").write_bytes(
                archive.extractfile("hf/locks/requirements-env.lock").read())
        oauth = ("hf_oauth: true\nhf_oauth_scopes:\n  - inference-api\n"
                 if config["resources"].get("interactive_hf_oauth", False) else "")
        (stage / "README.md").write_text(f"---\ntitle: Data Agent {'SETA Whitebox' if arm == 'whitebox' else 'Blackbox OpenCode' if arm == 'opencode' else 'Blackbox Harbor'} Env\nsdk: docker\napp_port: 7860\n{oauth}---\n\nInteractive OpenEnv environment for data-agent training and evaluation.\n")
        entrypoint = "opencode_space.py" if arm == "opencode" else "space_app.py"
        dockerfile = f'''FROM {config['compute']['bootstrap_image']}
USER root
RUN useradd -m -u 1000 user
WORKDIR /workspace/repro
COPY requirements-env.lock /tmp/requirements-env.lock
RUN uv venv --python 3.12 /opt/environment && uv pip sync --python /opt/environment/bin/python --require-hashes /tmp/requirements-env.lock
COPY bundle.tar.gz /tmp/bundle.tar.gz
RUN python -c "import tarfile; tarfile.open('/tmp/bundle.tar.gz').extractall('/workspace/repro', filter='data')" && rm /tmp/bundle.tar.gz
RUN chown -R user:user /workspace /opt/environment
USER user
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/workspace/repro/hf/runtime REPRO_ROOT=/workspace/repro PORT=7860
CMD ["/opt/environment/bin/python", "-u", "/workspace/repro/hf/runtime/{entrypoint}"]
'''
        (stage / "Dockerfile").write_text(dockerfile)
        commit = api.upload_folder(repo_id=repo, repo_type="space", folder_path=stage,
                                   commit_message="Deploy frozen Daytona environment service")
        hardware = config["serving"]["shared_service"]["hardware"]
        # CPU Basic uses HF's default sleep policy; disabling sleep requires paid hardware.
        hardware_options = {} if hardware == "cpu-basic" else {"sleep_time": -1}
        runtime = api.request_space_hardware(repo, hardware, **hardware_options)
        api.update_repo_settings(repo, repo_type="space",
                                 visibility=config["resources"].get("space_visibility", "private"))
        save(out / f"space-{name}.json", {"repo": repo, "revision": commit.oid,
             "bundle_sha256": info["sha256"], "stage": str(runtime.stage), "url": f"https://huggingface.co/spaces/{repo}"})
        print(json.dumps({"space": repo, "stage": str(runtime.stage),
                          "visibility": config["resources"].get("space_visibility", "private")}), flush=True)

def submit(api, config, secrets, out, args):
    from huggingface_hub import Volume
    uploaded = json.loads((out / "bundle_uploaded.json").read_text())
    role = args.role
    qualification = (role == "train" and args.phase == "smoke") or (role == "coordinator" and args.phase == "qualify")
    if config.get("training_launch_hold") and role in {"train", "coordinator"} and not qualification:
        raise ValueError(config["training_launch_hold"])
    if role == "train" and args.phase not in {"smoke", "long"}:
        raise ValueError("Training phase must be smoke or long")
    proof = None
    if role == "train" and args.phase == "long":
        from launch_gates import verify
        proof = verify(api, config, uploaded, args.arm, args.baseline_job, args.smoke_job, out)
        if args.arm == "opencode":
            from launch_gates import verify_comparison_baseline
            comparison_id = getattr(args, "comparison_baseline_job", None)
            proof["native_baseline_prefix"] = proof["baseline_prefix"]
            proof["native_baseline_job"] = args.baseline_job
            proof["comparison_baseline_job"] = comparison_id
            proof["baseline_prefix"] = verify_comparison_baseline(api, config, comparison_id, out)
            proof["baseline_job"] = comparison_id
        if getattr(args, "checkpoint_eval_job", None):
            from launch_gates import verify_checkpoint_eval
            proof["checkpoint_eval"] = verify_checkpoint_eval(api, config, uploaded, args.arm,
                args.smoke_job, args.checkpoint_eval_job, out)
        elif getattr(args, "external_checkpoint_coordinator", False):
            raise ValueError("Long training with an external coordinator requires --checkpoint-eval-job")
        save(out / "launch-proofs" / args.arm / "verified.json", proof)
    if role == "coordinator" and args.phase not in {"setup", "qualify"} and not args.training_job:
        raise ValueError("--training-job is required for the checkpoint coordinator")
    if role == "eval" and args.phase == "checkpoint":
        raise ValueError("Checkpoint evaluations are submitted with verified manifests by the coordinator")
    flavor = args.flavor or ("cpu-upgrade" if role in ["preflight", "coordinator"] else
                            config["compute"]["training_flavor"] if role == "train" else "a100-large")
    if role == "train" and flavor not in {"h200x2", "a100x4"}:
        raise ValueError("Training needs separate inference and optimizer GPUs")
    identity = f"{role}-{args.arm}-{int(time.time())}"
    env = {"ARTIFACT_BUCKET": config["resources"]["artifacts_bucket"], "RUN_ID": config["run_id"],
           "RUN_OWNER": identity, "PYTHONUNBUFFERED": "1", "COMPARISON_ARM": args.arm,
           "BUNDLE_SHA256": uploaded["sha256"], "JOB_FLAVOR": flavor,
           "BUNDLE_REPO": uploaded["repo"], "BUNDLE_REVISION": uploaded["revision"],
           "TRACKIO_MODE": "offline", "HF_JOB_NAMESPACE": config["namespace"],
           "EVAL_CONCURRENCY": str(config["evaluation"]["concurrency_per_arm"][args.arm])}
    if args.arm == "opencode":
        limits = config["evaluation"]["opencode_backend_concurrency"]
        env.update(EVAL_DAYTONA_CONCURRENCY=str(limits["daytona"]), EVAL_HF_CONCURRENCY=str(limits["hf"]))
    if getattr(args, "space_bundle_sha", None):
        if role != "train" or args.phase != "smoke":
            raise ValueError("An independently pinned Space is currently supported only for training qualification")
        env["SPACE_BUNDLE_SHA256"] = args.space_bundle_sha
    if proof:
        env.update(SMOKE_PREFIX=proof["smoke_prefix"], BASELINE_JOB=proof.get("comparison_baseline_job") or args.baseline_job,
                   SPACE_BUNDLE_SHA256=proof["space_bundle_sha256"],
                   BASELINE_PREFIX=proof["baseline_prefix"],
                   COORDINATION_PREFIX="hf://buckets/" + config["resources"]["artifacts_bucket"] + "/" +
                       config["run_id"] + "/coordination/" + identity)
        if proof.get("native_baseline_prefix"):
            env["NATIVE_BASELINE_PREFIX"] = proof["native_baseline_prefix"]
    if args.training_job:
        env["TRAINING_JOB"] = args.training_job
    if getattr(args, "baseline_job_map", None):
        if role != "coordinator" or args.phase != "setup":
            raise ValueError("--baseline-job-map is for the two-arm setup pipeline")
        mapping = json.loads(args.baseline_job_map)
        if set(mapping) != {"blackbox", "whitebox"}:
            raise ValueError("Provide blackbox and whitebox baseline IDs")
        env["BASELINE_JOB_MAP"] = json.dumps(mapping)
    if args.resume_eval_owner:
        if args.role != "eval" or args.arm != "whitebox" or args.phase != "baseline":
            raise ValueError("--resume-eval-owner requires a whitebox baseline evaluation")
        if Path(args.resume_eval_owner).name != args.resume_eval_owner:
            raise ValueError("Resume owner must be a single artifact directory name")
        env["RESUME_EVAL_PREFIX"] = ("hf://buckets/" + config["resources"]["artifacts_bucket"] + "/" +
                                     config["run_id"] + "/jobs/" + args.resume_eval_owner)
    key = args.arm
    if key in config["resources"]["environment_spaces"]:
        space = config["resources"]["environment_spaces"][key]
        info = api.space_info(space)
        env["SPACE_URL"] = info.host if str(info.host).startswith("https://") else "https://" + info.host
    if role == "train" and args.arm == "opencode":
        import httpx
        evaluator = api.space_info(config["resources"]["environment_spaces"]["blackbox"])
        url = str(evaluator.host)
        url = url if url.startswith("https://") else "https://" + url
        eval_identity = httpx.get(url + "/deployment", headers={"Authorization": "Bearer " + secrets["HF_TOKEN"]},
                             timeout=60).raise_for_status().json()
        if eval_identity.get("arm") != "blackbox" or eval_identity.get("test_tasks") != 250:
            raise ValueError("The four-harness checkpoint evaluator has the wrong task catalog")
        env.update(CHECKPOINT_EVAL_SPACE_URL=url,
                   CHECKPOINT_EVAL_SPACE_SHA256=eval_identity["bundle_sha256"])
    if role == "train" and args.arm in {"blackbox", "opencode"}:
        from runtime.service_contract import check
        save(out / f"service-contract-{args.arm}.json",
             check(env["SPACE_URL"], secrets["HF_TOKEN"], args.arm))
    if proof and env["SPACE_URL"].rstrip("/") != proof["space_url"].rstrip("/"):
        raise ValueError("Long training must use the environment qualified by its optimizer smoke")
    command = ["python", "/bundle/bootstrap.py", "--role", role, "--arm", args.arm,
               "--phase", args.phase, "--dp", str(args.dp)]
    if args.limit:
        command += ["--limit", str(args.limit)]
    labels = {"experiment": "data-agent-daytona", "role": role, "arm": args.arm,
              "phase": args.phase, "run": config["run_id"]}
    if args.training_job:
        labels["training_job"] = args.training_job
    if getattr(args, "dry_run", False):
        # Validate real proofs and Space identity above, but never allocate a job.
        preview = {"dry_run": True, "submitted": False, "role": role, "arm": args.arm,
            "phase": args.phase, "flavor": flavor, "timeout": args.timeout or "2h",
            "command": command, "labels": labels, "environment": env,
            "bundle": uploaded, "proof": proof,
            "external_checkpoint_coordinator": bool(getattr(args, "external_checkpoint_coordinator", False))}
        save(out / "launch-preview.json", preview)
        print(json.dumps(preview), flush=True)
        return preview
    job = api.run_job(namespace=config["namespace"], image=config["compute"]["bootstrap_image"],
                     command=command, flavor=flavor, timeout=args.timeout or ("1h" if role == "preflight" else "2h"),
                     env=env, secrets=secrets, name=f"daytona-{identity}",
                     labels=labels,
                     volumes=[Volume(type="dataset", source=uploaded["repo"], revision=uploaded["revision"],
                                     mount_path="/bundle", read_only=True)], expose=[8000] if role in ["eval", "train"] else None)
    value = {"id": job.id, "url": job.url, "owner": identity, "role": role, "arm": args.arm,
             "phase": args.phase, "flavor": flavor, "bundle": uploaded, "stage": job.status.stage}
    save(out / "jobs" / f"{job.id}.json", value)
    print(json.dumps(value), flush=True)
    if role == "train" and args.phase == "long" and not getattr(args, "external_checkpoint_coordinator", False):
        # The GPU ID is durable before the CPU submission. A failed coordinator
        # launch can be retried explicitly with --role coordinator --training-job.
        coordinator_args = argparse.Namespace(**{**vars(args), "role": "coordinator", "training_job": job.id,
            "flavor": "cpu-upgrade", "timeout": "36h"})
        submit(api, config, secrets, out, coordinator_args)
    return value


def retire_spaces(api, config, out):
    """Pause superseded Spaces only after all associated HF Jobs have drained."""
    terminal = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"}
    active = [j for j in api.list_jobs(namespace=config["namespace"],
              labels={"experiment": "data-agent-daytona"}) if j.status.stage not in terminal]
    report = []
    for repo in config["resources"].get("retired_spaces", []):
        info = api.space_info(repo)
        needles = [repo, str(info.host)]
        users = [j.id for j in active if any(needle in str(value) for needle in needles
                 for value in j.environment.values())]
        if users:
            row = {"space": repo, "state": "draining", "active_jobs": users}
        else:
            runtime = api.pause_space(repo)
            row = {"space": repo, "state": str(runtime.stage), "repository_preserved": True}
        report.append(row)
        print(json.dumps(row), flush=True)
    save(out / "space-retirement.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["upload", "spaces", "job", "status", "trackio", "retire"])
    p.add_argument("--env-file")
    p.add_argument("--config", type=Path, default=HERE / "configs/deployment.json")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--only", help="Comma-separated arms: blackbox,whitebox")
    p.add_argument("--role", default="preflight", choices=["preflight", "train", "eval", "coordinator"])
    p.add_argument("--arm", default="blackbox", choices=["blackbox", "whitebox", "opencode"])
    p.add_argument("--phase", default="smoke", choices=["smoke", "ramp", "baseline", "long", "checkpoint", "setup", "qualify"])
    p.add_argument("--flavor")
    p.add_argument("--timeout")
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume-eval-owner", help="Restore the immutable first-graded ledger of a whitebox baseline")
    p.add_argument("--smoke-job", help="Completed optimizer/save/resume HF Job for this arm and bundle")
    p.add_argument("--baseline-job", help="Completed HF baseline with matching task and evaluation protocol")
    p.add_argument("--comparison-baseline-job", help="For native OpenCode: completed shared four-harness Harbor baseline")
    p.add_argument("--training-job", help="HF training Job followed by a checkpoint coordinator")
    p.add_argument("--baseline-job-map", help="JSON mapping of both baseline Job IDs for --phase setup")
    p.add_argument("--space-bundle-sha", help="Exact already-deployed Space bundle for a training smoke; avoids restarting active eval services")
    p.add_argument("--external-checkpoint-coordinator", action="store_true",
                   help="Start the separately versioned hf_followup.py coordinator after submitting this training job")
    p.add_argument("--checkpoint-eval-job", help="Completed independent checkpoint evaluation of --smoke-job")
    p.add_argument("--dry-run", action="store_true", help="Validate and save the exact job request without allocating it")
    a = p.parse_args()
    from huggingface_hub import HfApi
    config = json.loads(a.config.read_text())
    secrets = credentials(a.env_file)
    api = HfApi(token=secrets["HF_TOKEN"])
    if a.action == "upload":
        api.create_bucket(config["resources"]["artifacts_bucket"], private=True, exist_ok=True)
        print(json.dumps(upload_bundle(api, config, a.out)))
    elif a.action == "spaces":
        spaces(api, config, secrets, a.out, set(a.only.split(",")) if a.only else None)
    elif a.action == "job":
        submit(api, config, secrets, a.out, a)
    elif a.action == "retire":
        retire_spaces(api, config, a.out)
    elif a.action == "status":
        for path in sorted((a.out / "jobs").glob("*.json")):
            local = json.loads(path.read_text())
            j = api.inspect_job(job_id=local["id"], namespace=config["namespace"])
            print(json.dumps({"id": j.id, "role": local["role"], "arm": local["arm"], "stage": j.status.stage, "message": j.status.message}))
        for name, repo in config["resources"]["environment_spaces"].items():
            try:
                s = api.get_space_runtime(repo)
                print(json.dumps({"space": repo, "stage": str(s.stage)}))
            except Exception as e:
                print(json.dumps({"space": repo, "error": type(e).__name__}))
    else:
        if not config["resources"].get("trackio_space"):
            raise SystemExit("Trackio is configured in training jobs only; no Trackio Space is deployed")
        os.environ["HF_TOKEN"] = secrets["HF_TOKEN"]
        from trackio.deploy import create_space_if_not_exists
        create_space_if_not_exists(config["resources"]["trackio_space"],
                                   bucket_id=config["resources"]["trackio_bucket"], private=config["logging"].get("space_private", False))
        api.update_repo_settings(config["resources"]["trackio_space"], repo_type="space",
                                 private=config["logging"].get("space_private", False))
        print(json.dumps({"trackio_space": config["resources"]["trackio_space"]}))


if __name__ == "__main__":
    main()
