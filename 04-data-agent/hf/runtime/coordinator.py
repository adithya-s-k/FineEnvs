"""Queue independent HF checkpoint evaluations after durable checkpoint publication."""
import hashlib
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

from common import ROOT, write_json
from checkpoint_store import READY, bucket_location, digest, download_json

TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"}


def eligible(manifest, interval, include_final):
    step = manifest["step"]
    return type(step) is int and step > 0 and (step % interval == 0 or include_final and manifest.get("final", False))


def evaluation_key(training_job, manifest_sha, protocol):
    encoded = json.dumps([training_job, manifest_sha, protocol], sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def verified_terminal_result(stage, score, evidence, manifest, sha, source, status=None):
    """Retain a fully published result even if the final artifact flush failed."""
    if not score.get("comparison_ready") or evidence.get("manifest_sha256") != sha:
        raise ValueError("Checkpoint evaluation did not pass its evidence gate")
    if stage == "COMPLETED":
        return
    if stage != "ERROR" or not status or not (
        status.get("passed") is True and status.get("finished_at")
        and status.get("arm") == manifest["arm"] and status.get("phase") == "checkpoint"
        and score.get("complete") is True and score.get("tito_pass") is True
        and score.get("graded_cells", 0) == score.get("expected_cells", -1) > 0
        and evidence.get("step") == manifest["step"]
        and evidence.get("bundle_sha256") == manifest["bundle_sha256"]
        and evidence.get("source") == source
    ):
        raise ValueError("Failed job has no complete, provenance-verified evaluation")


def sync_decisions(api, decisions, destination):
    """Retry transient bucket failures without losing submission intent."""
    from bucket_io import sync_with_retry
    sync_with_retry(api, decisions, destination)


def submit_once(state, key, known, persist, launch):
    """Persist intent before submission; an ambiguous response must never double-submit."""
    matches = [j for j in known if (j.labels or {}).get("evaluation_key") == key]
    if len(matches) > 1:
        raise RuntimeError("Duplicate checkpoint evaluation jobs require reconciliation")
    if matches:
        state[key] = {**state.get(key, {}), "job_id": matches[0].id, "status": "submitted"}
        persist()
        return matches[0]
    if key in state:
        raise RuntimeError("Unresolved submission intent; inspect HF Jobs before retrying")
    state[key] = {"status": "submitting", "created_at": time.time()}
    persist()
    job = launch()
    state[key].update(job_id=job.id, status="submitted")
    persist()
    return job


def validate_requested_final(request, training_id, source, manifest, manifest_sha):
    """Bind an intentional stop to one fully verified, durable checkpoint."""
    if not request:
        return False
    expected = {"training_job": training_id, "checkpoint": source,
                "step": manifest["step"], "manifest_sha256": manifest_sha}
    if (request.get("full_checkpoint_verified") is not True
            or request.get("user_requested_stop") is not True
            or type(request.get("step")) is not int or request["step"] <= 0
            or any(request.get(k) != v for k, v in expected.items())):
        raise ValueError("Requested final evaluation does not match the verified stopped checkpoint")
    return True


def run(output, arm, admission=None, final_checkpoint=None):
    from huggingface_hub import HfApi, Volume
    from huggingface_hub.errors import EntryNotFoundError
    api = HfApi()
    config = json.loads((ROOT / "hf/configs/deployment.json").read_text())
    namespace = config["namespace"]
    training_id = os.environ["TRAINING_JOB"]
    training = api.inspect_job(job_id=training_id, namespace=namespace)
    if training.labels.get("role") != "train" or training.labels.get("arm") != arm:
        raise ValueError("Coordinator must follow the requested arm's training job")
    if training.environment["BUNDLE_SHA256"] != os.environ["BUNDLE_SHA256"]:
        raise ValueError("Coordinator and training bundle differ")
    owner = training.environment["RUN_OWNER"]
    bucket = os.environ["ARTIFACT_BUCKET"]
    base = "hf://buckets/" + bucket + "/" + os.environ["RUN_ID"]
    training_root = base + "/jobs/" + owner
    destination = base + "/coordination/" + owner
    decisions = output / "decisions"
    decisions.mkdir(parents=True, exist_ok=True)
    try:
        state = download_json(destination, "state.json", decisions / "state.json", api)
    except EntryNotFoundError:
        state = {}

    def persist():
        write_json(decisions / "state.json", state)
        sync_decisions(api, decisions, destination)

    protocol = {"evaluation": config["evaluation"], "data": config["data"], "pins": config["harness_pins"]}
    evaluation = config["evaluation"]
    terminal_seen = None
    while True:
        # A second coordinator exits before it can submit work.
        peers = list(api.list_jobs(namespace=namespace, labels={"role": "coordinator", "training_job": training_id}))
        active_peers = sorted((j for j in peers if j.status.stage not in TERMINAL), key=lambda j: j.id)
        if active_peers and active_peers[0].environment.get("RUN_OWNER") != os.environ["RUN_OWNER"]:
            raise RuntimeError("Another coordinator already owns this training job")
        training = api.inspect_job(job_id=training_id, namespace=namespace)
        if training.status.stage in TERMINAL:
            terminal_seen = terminal_seen or time.monotonic()
        else:
            terminal_seen = None
        jobs = list(api.list_jobs(namespace=namespace, labels={"role": "eval", "training_job": training_id}))
        active = [j for j in jobs if j.status.stage not in TERMINAL]
        _, prefix = bucket_location(training_root + "/run")
        try:
            folders = [item.path for item in api.list_bucket_tree(bucket, prefix=prefix, recursive=False)
                       if Path(item.path).name.startswith("checkpoint-")]
        except EntryNotFoundError:
            folders = []
        pending = []
        alerts = []
        checkpoints = []
        for folder in folders:
            source = "hf://buckets/" + bucket + "/" + folder
            target = output / "manifests" / Path(folder).name / READY
            try:
                manifest = download_json(source, READY, target, api)
            except EntryNotFoundError:
                continue  # Files are still being uploaded; only the final marker admits eval.
            if manifest["arm"] != arm or manifest["bundle_sha256"] != os.environ["BUNDLE_SHA256"]:
                raise ValueError("Published checkpoint provenance differs from training")
            checkpoints.append((source, manifest, digest(target)))
        final_step = max((m["step"] for _, m, _ in checkpoints), default=0) if terminal_seen is not None else 0
        if final_checkpoint and not any(m["step"] == final_checkpoint["step"] for _, m, _ in checkpoints):
            raise ValueError("Requested final checkpoint is not durably published")
        for source, manifest, sha in checkpoints:
            if final_checkpoint and manifest["step"] != final_checkpoint["step"]:
                continue
            requested_final = validate_requested_final(final_checkpoint, training_id, source, manifest, sha)
            if os.environ.get("QUALIFY_CHECKPOINT_STEP") and manifest["step"] != int(os.environ["QUALIFY_CHECKPOINT_STEP"]):
                continue
            is_final = requested_final or (evaluation["also_final"] and manifest["step"] == final_step and final_step > 0)
            if not is_final and not eligible(manifest, evaluation["interval_steps"], evaluation["also_final"]):
                continue
            key = evaluation_key(training_id, sha, protocol)
            match = [j for j in jobs if j.labels.get("evaluation_key") == key]
            if len(match) > 1:
                raise RuntimeError("Duplicate evaluation identity")
            if match:
                j = match[0]
                state[key] = {"step": manifest["step"], "job_id": j.id, "stage": j.status.stage,
                              "checkpoint": source, "manifest_sha256": sha}
                score_path = decisions / "scores" / f"step-{manifest['step']:06d}.json"
                if j.status.stage in {"COMPLETED", "ERROR"}:
                    eval_source = base + "/jobs/" + j.environment["RUN_OWNER"]
                    if not score_path.exists():
                        score = download_json(eval_source, "canonical_scores.json", output / "score-cache" / (key + ".json"), api)
                        evidence = download_json(eval_source, "checkpoint_evaluation.json", output / "source-cache" / (key + ".json"), api)
                        status = (download_json(eval_source, "status.json", output / "status-cache" / (key + ".json"), api)
                                  if j.status.stage == "ERROR" else None)
                        verified_terminal_result(j.status.stage, score, evidence, manifest, sha, source, status)
                        write_json(score_path, {"step": manifest["step"], "job_id": j.id,
                                   "source": evidence, "scores": score, "provider_stage": j.status.stage,
                                   "evaluation_status": status})
                    cached = json.loads(score_path.read_text())
                    if cached.get("job_id") != j.id or cached.get("step") != manifest["step"]:
                        raise ValueError("Cached score belongs to a different evaluation")
                    verified_terminal_result(j.status.stage, cached["scores"], cached["source"],
                                             manifest, sha, source, cached.get("evaluation_status"))
                    state[key]["result_verified"] = True
                elif j.status.stage in TERMINAL and j.status.stage != "COMPLETED":
                    alerts.append({"step": manifest["step"], "job_id": j.id, "stage": j.status.stage})
            elif key in state:
                alerts.append({"step": manifest["step"], "reason": "ambiguous submission; reconciliation required"})
            else:
                pending.append((manifest["step"], key, source, sha))
        capacity = admission() if admission is not None else nullcontext(True)
        with capacity as admitted:
            can_dispatch = training.status.stage not in {"CANCELED", "CANCELLED", "DELETED"}
            # A normal cancellation must still suppress new GPU work. An explicit
            # user stop permits exactly the checkpoint bound and verified above.
            can_dispatch = can_dispatch or (final_checkpoint is not None and training.status.stage in {"CANCELED", "CANCELLED"})
            if admitted and pending and not active and not alerts and can_dispatch:
                step, key, source, sha = sorted(pending)[0]
                eval_owner = f"eval-{arm}-step{step}-{key[:12]}"
                env = {"ARTIFACT_BUCKET": bucket, "RUN_ID": os.environ["RUN_ID"], "RUN_OWNER": eval_owner,
                       "COMPARISON_ARM": arm, "BUNDLE_SHA256": os.environ["BUNDLE_SHA256"],
                       "SPACE_URL": training.environment["SPACE_URL"], "JOB_FLAVOR": config["compute"]["eval_flavor"],
                       "SPACE_BUNDLE_SHA256": training.environment.get("SPACE_BUNDLE_SHA256", training.environment["BUNDLE_SHA256"]),
                       "EVAL_CONCURRENCY": str(evaluation["concurrency_per_arm"][arm]),
                       "CHECKPOINT_PREFIX": source, "CHECKPOINT_MANIFEST_SHA": sha, "CHECKPOINT_STEP": str(step),
                       "TRAINING_JOB": training_id, "PYTHONUNBUFFERED": "1", "TRACKIO_MODE": "offline"}
                env["HF_JOB_NAMESPACE"] = namespace
                if arm == "opencode":
                    # Checkpoint provenance remains native OpenCode, while the
                    # comparison evaluates the same weights through four Harbor agents.
                    env.update(EVAL_SUITE="harbor",
                               SPACE_URL=training.environment["CHECKPOINT_EVAL_SPACE_URL"],
                               SPACE_BUNDLE_SHA256=training.environment["CHECKPOINT_EVAL_SPACE_SHA256"])
                secrets = {k: os.environ[k] for k in ["HF_TOKEN", "DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET"] if os.environ.get(k)}
                def launch():
                    return api.run_job(namespace=namespace, image=config["compute"]["bootstrap_image"],
                        command=["python", "/bundle/bootstrap.py", "--role", "eval", "--arm", arm,
                                 "--phase", "checkpoint", "--dp", str(config["compute"]["eval_dp"])],
                        flavor=config["compute"]["eval_flavor"], timeout="4h", env=env, secrets=secrets,
                        name=eval_owner, expose=[8000], labels={"experiment": config["experiment"], "role": "eval",
                        "arm": arm, "phase": "checkpoint", "training_job": training_id, "evaluation_key": key},
                        volumes=[Volume(type="dataset", source=os.environ["BUNDLE_REPO"],
                            revision=os.environ["BUNDLE_REVISION"], mount_path="/bundle", read_only=True)])
                j = submit_once(state, key, jobs, persist, launch)
                active.append(j)
                pending.pop(pending.index((step, key, source, sha)))
        write_json(decisions / "monitor.json", {"checked_at": time.time(), "training_job": training_id,
                   "training_stage": training.status.stage, "active_eval_jobs": [j.id for j in active],
                   "pending_steps": sorted(p[0] for p in pending), "alerts": alerts})
        persist()
        if alerts:
            raise RuntimeError("Checkpoint evaluation failed or has an ambiguous submission; inspect monitor.json")
        if terminal_seen is not None and time.monotonic() - terminal_seen >= 120 and not active and (not pending or not can_dispatch):
            return
        time.sleep(60)
