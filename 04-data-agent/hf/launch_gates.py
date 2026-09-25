"""Verify completed HF baseline and optimizer evidence before allocating a long run."""
import hashlib
import json
from pathlib import Path
import tarfile


def protocol_identity(config):
    # Admission limits are deployment controls. Changing another arm's concurrency
    # must not invalidate an otherwise identical completed Whitebox baseline.
    evaluation = {key: config["evaluation"].get(key) for key in (
        "metric", "max_output_tokens_per_call", "max_model_calls", "episode_timeout_seconds",
        "first_graded_result_immutable", "retry_only_ungraded_infrastructure_failures")}
    return {"model": config["model"], "data": config["data"]["manifest_sha256"],
            "pins": config["harness_pins"], "evaluation": evaluation,
            "sampling": config["training"]["sampling"]}


def validate_proofs(config, uploaded, arm, baseline_config, baseline, smoke):
    if protocol_identity(config) != protocol_identity(baseline_config):
        raise ValueError("Baseline task/sampling/harness protocol differs from this deployment")
    if arm == "opencode":
        baseline = baseline.get("daytona", baseline)
        if baseline.get("implementation") != "standalone-opencode":
            raise ValueError("Native training requires a native OpenCode baseline")
        baseline = {**baseline, "arm": "opencode"}
    expected = 1000 if arm == "blackbox" else 250
    if not (baseline.get("comparison_ready") and baseline.get("arm") == arm and
            baseline.get("graded_cells") == expected and baseline.get("tito_pass")):
        raise ValueError("Full baseline coverage/TiTO/version evidence is required")
    if not (smoke.get("passed") and smoke.get("arm") == arm and smoke.get("bundle_sha256") == uploaded["sha256"]
            and smoke.get("remote_restore_verified") and smoke.get("tito_pass") and smoke.get("weights_updated")
            and smoke.get("native_optimizer_state_verified")):
        raise ValueError("Optimizer/save/resume smoke must pass with this exact runtime bundle")


def validate_comparison_baseline(score):
    expected = {"opencode", "claude-code", "codex", "mini-swe-agent"}
    harnesses = score.get("harnesses", {})
    if (not score.get("comparison_ready") or not score.get("tito_pass")
            or score.get("graded_cells") != 1000 or set(harnesses) != expected
            or any(value.get("graded") != 250 for value in harnesses.values())):
        raise ValueError("Checkpoint comparisons require the complete four-harness baseline, not the native diagnostic")


def archived_config(api, job):
    volume = next(v for v in job.volumes if v.mount_path == "/bundle" and v.type == "dataset")
    from huggingface_hub import hf_hub_download
    archive = Path(hf_hub_download(volume.source, "bundle.tar.gz", repo_type="dataset",
                                  revision=volume.revision, token=api.token))
    if hashlib.sha256(archive.read_bytes()).hexdigest() != job.environment["BUNDLE_SHA256"]:
        raise ValueError("Baseline's archived runtime checksum differs from the executed bundle")
    with tarfile.open(archive) as tar:
        return json.load(tar.extractfile("hf/configs/deployment.json"))


def verify_comparison_baseline(api, config, job_id, out):
    if not job_id:
        raise ValueError("Native training also needs --comparison-baseline-job from the shared four-harness evaluation")
    job = api.inspect_job(job_id=job_id, namespace=config["namespace"])
    if (job.status.stage != "COMPLETED" or job.labels.get("role") != "eval"
            or job.labels.get("phase") != "baseline" or job.labels.get("arm") != "blackbox"):
        raise ValueError("Comparison baseline must be a completed Harbor baseline Job")
    if protocol_identity(config) != protocol_identity(archived_config(api, job)):
        raise ValueError("Comparison baseline task/model/sampling protocol differs")
    prefix = job.environment["RUN_ID"] + "/jobs/" + job.environment["RUN_OWNER"]
    target = Path(out) / "launch-proofs/opencode/comparison-baseline.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    api.download_bucket_files(job.environment["ARTIFACT_BUCKET"],
        files=[(prefix + "/canonical_scores.json", str(target))], raise_on_missing_files=True)
    validate_comparison_baseline(json.loads(target.read_text()))
    return "hf://buckets/" + job.environment["ARTIFACT_BUCKET"] + "/" + prefix


def verify(api, config, uploaded, arm, baseline_id, smoke_id, out):
    if not baseline_id or not smoke_id:
        raise ValueError("Long training requires --baseline-job and --smoke-job")
    jobs = []
    for job_id, role, phase in [(baseline_id, "eval", "baseline"), (smoke_id, "train", "smoke")]:
        j = api.inspect_job(job_id=job_id, namespace=config["namespace"])
        if j.status.stage != "COMPLETED" or any(j.labels.get(k) != v for k, v in {"arm": arm, "role": role, "phase": phase}.items()):
            raise ValueError(f"Job {job_id} is not a completed {arm} {role}/{phase}")
        jobs.append(j)
    out = Path(out) / "launch-proofs" / arm
    out.mkdir(parents=True, exist_ok=True)
    proofs = []
    sources = []
    for j, filename in zip(jobs, ["canonical_scores.json", "training_smoke_verified.json"]):
        prefix = j.environment["RUN_ID"] + "/jobs/" + j.environment["RUN_OWNER"]
        api.download_bucket_files(j.environment["ARTIFACT_BUCKET"],
            files=[(prefix + "/" + filename, str(out / filename))], raise_on_missing_files=True)
        proofs.append(json.loads((out / filename).read_text()))
        sources.append("hf://buckets/" + j.environment["ARTIFACT_BUCKET"] + "/" + prefix)
    baseline_config = archived_config(api, jobs[0])
    validate_proofs(config, uploaded, arm, baseline_config, proofs[0], proofs[1])
    smoke_job = jobs[1]
    prefix = smoke_job.environment["RUN_ID"] + "/jobs/" + smoke_job.environment["RUN_OWNER"]
    api.download_bucket_files(smoke_job.environment["ARTIFACT_BUCKET"],
        files=[(prefix + "/space_identity.json", str(out / "space_identity.json"))], raise_on_missing_files=True)
    service = json.loads((out / "space_identity.json").read_text())
    service_pin = smoke_job.environment.get("SPACE_BUNDLE_SHA256", uploaded["sha256"])
    if service.get("bundle_sha256") != service_pin:
        raise ValueError("Optimizer smoke did not verify its declared environment bundle")
    report = {"passed": True, "baseline_job": baseline_id, "smoke_job": smoke_id,
              "baseline_prefix": sources[0], "smoke_prefix": sources[1], "bundle_sha256": uploaded["sha256"],
              "space_bundle_sha256": service_pin, "space_url": smoke_job.environment["SPACE_URL"]}
    (out / "verified.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def validate_checkpoint_eval(uploaded, arm, smoke_id, job, score, evidence):
    if (job.status.stage != "COMPLETED" or
            any(job.labels.get(k) != v for k, v in
                {"role": "eval", "phase": "checkpoint", "arm": arm, "training_job": smoke_id}.items())):
        raise ValueError("Checkpoint evaluation must have completed for the qualified smoke")
    expected = 250 if arm == "whitebox" else 1000
    if (not score.get("comparison_ready") or not score.get("tito_pass") or
            score.get("graded_cells") != expected or score.get("arm") != arm or
            evidence.get("step") != 4 or job.environment.get("CHECKPOINT_STEP") != "4" or
            evidence.get("source") != job.environment.get("CHECKPOINT_PREFIX") or
            evidence.get("bundle_sha256") != uploaded["sha256"] or
            evidence.get("manifest_sha256") != job.environment.get("CHECKPOINT_MANIFEST_SHA") or
            job.environment.get("BUNDLE_SHA256") != uploaded["sha256"]):
        raise ValueError("Checkpoint evaluation coverage, TiTO or source identity differs")


def verify_checkpoint_eval(api, config, uploaded, arm, smoke_id, eval_id, out):
    job = api.inspect_job(job_id=eval_id, namespace=config["namespace"])
    target = Path(out) / "launch-proofs" / arm / "checkpoint-eval"
    target.mkdir(parents=True, exist_ok=True)
    prefix = job.environment["RUN_ID"] + "/jobs/" + job.environment["RUN_OWNER"]
    values = []
    for name in ["canonical_scores.json", "checkpoint_evaluation.json"]:
        api.download_bucket_files(job.environment["ARTIFACT_BUCKET"],
            files=[(prefix + "/" + name, str(target / name))], raise_on_missing_files=True)
        values.append(json.loads((target / name).read_text()))
    validate_checkpoint_eval(uploaded, arm, smoke_id, job, *values)
    return {"passed": True, "job_id": eval_id, "step": 4,
            "manifest_sha256": values[1]["manifest_sha256"]}
