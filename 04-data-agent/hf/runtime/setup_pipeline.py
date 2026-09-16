"""Finish the two HF baselines, deploy the tested runtime, smoke, then start training."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from common import ROOT, ENV_PY, start, write_json
from checkpoint_store import download_json
from coordinator import TERMINAL
BUNDLE = Path("/bundle")


def run(output):
    from huggingface_hub import HfApi
    import httpx
    sys.path.insert(0, str(ROOT / "hf"))
    import deploy
    api = HfApi()
    config = json.loads((ROOT / "hf/configs/deployment.json").read_text())
    namespace = config["namespace"]
    bundle = json.loads((BUNDLE / "bundle.json").read_text())
    bundle.update(repo=os.environ["BUNDLE_REPO"], revision=os.environ["BUNDLE_REVISION"])
    destination = ("hf://buckets/" + os.environ["ARTIFACT_BUCKET"] + "/" + os.environ["RUN_ID"] +
                   "/pipelines/" + bundle["sha256"])
    output.mkdir(parents=True, exist_ok=True)
    try:
        state = download_json(destination, "pipeline.json", output / "pipeline.json", api)
    except Exception as exc:
        from huggingface_hub.errors import EntryNotFoundError
        if not isinstance(exc, EntryNotFoundError):
            raise
        state = {"bundle_sha256": bundle["sha256"], "jobs": {}, "phase": "waiting_baselines"}

    def persist():
        state["updated_at"] = time.time()
        write_json(output / "pipeline.json", state)
        api.sync_bucket(str(output), destination, include=["pipeline.json", "jobs/*.json", "launch-proofs/**", "ui-smoke.json"], quiet=True)

    def wait_until(check):
        while not check():
            persist()
            time.sleep(60)
        persist()

    secrets = {k: os.environ[k] for k in ["HF_TOKEN", "DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET", "OPENAI_API_KEY"] if os.environ.get(k)}
    args_base = dict(dp=1, limit=0, resume_eval_owner=None, training_job=None, baseline_job=None, smoke_job=None)

    def ensure_job(role, arm, phase, **options):
        key = ":".join([role, arm, phase])
        jobs = [j for j in api.list_jobs(namespace=namespace, labels={"role": role, "arm": arm, "phase": phase,
                "run": config["run_id"]}) if j.environment.get("BUNDLE_SHA256") == bundle["sha256"]]
        if len(jobs) > 1:
            raise RuntimeError(f"Multiple matching {key} jobs require reconciliation")
        if jobs:
            state["jobs"][key] = {"id": jobs[0].id, "stage": jobs[0].status.stage}
            persist()
            return jobs[0].id
        if key in state["jobs"]:
            raise RuntimeError(f"Unresolved {key} submission intent; inspect HF Jobs before retrying")
        state["jobs"][key] = {"stage": "submitting"}
        persist()
        args = argparse.Namespace(**{**args_base, "role": role, "arm": arm, "phase": phase, **options})
        value = deploy.submit(api, config, secrets, output, args)
        state["jobs"][key] = {"id": value["id"], "stage": value["stage"]}
        persist()
        return value["id"]

    def completed(ids):
        done = True
        for arm, job_id in ids.items():
            job = api.inspect_job(job_id=job_id, namespace=namespace)
            state.setdefault("job_status", {})[job_id] = job.status.stage
            if job.status.stage in TERMINAL and job.status.stage != "COMPLETED":
                raise RuntimeError(f"{arm} job {job_id} ended as {job.status.stage}; artifacts require inspection")
            done = done and job.status.stage == "COMPLETED"
        return done

    try:
        peers = [j for j in api.list_jobs(namespace=namespace, labels={"role": "coordinator", "phase": "setup", "run": config["run_id"]})
                 if j.status.stage not in TERMINAL and j.environment.get("BUNDLE_SHA256") == bundle["sha256"]]
        if peers and min(peers, key=lambda j: j.id).environment["RUN_OWNER"] != os.environ["RUN_OWNER"]:
            raise RuntimeError("Another setup pipeline owns this bundle")
        baselines = json.loads(os.environ["BASELINE_JOB_MAP"]) if os.environ.get("BASELINE_JOB_MAP") else config["pipeline"]["baseline_jobs"]
        if set(baselines) != {"blackbox", "whitebox"}:
            raise ValueError("Both baseline job IDs are required")
        state["baseline_jobs"] = baselines
        wait_until(lambda: completed(baselines))
        for arm, job_id in baselines.items():
            job = api.inspect_job(job_id=job_id, namespace=namespace)
            source = "hf://buckets/" + os.environ["ARTIFACT_BUCKET"] + "/" + os.environ["RUN_ID"] + "/jobs/" + job.environment["RUN_OWNER"]
            scores = download_json(source, "canonical_scores.json", output / f"baseline-{arm}.json", api)
            if not scores["comparison_ready"] or scores["arm"] != arm:
                raise ValueError("Baseline coverage, token or version gate failed")
        state["phase"] = "deploying_training_runtime"
        persist()
        bundle_dir = output / "bundle"
        bundle_dir.mkdir(exist_ok=True)
        for name in ["bundle.tar.gz", "bundle.json"]:
            shutil.copy2(BUNDLE / name, bundle_dir / name)
        write_json(output / "bundle_uploaded.json", bundle)

        def deployments():
            result = {}
            for arm, repo in config["resources"]["environment_spaces"].items():
                host = api.space_info(repo).host.rstrip("/")
                if not host.startswith("https://"):
                    host = "https://" + host
                response = httpx.get(host + "/deployment", timeout=30)
                response.raise_for_status()
                result[arm] = response.json()
            return result

        if not state.get("deployed"):
            wait_until(lambda: all(sum(d["admission"]["active"].values()) == 0 for d in deployments().values()))
            current = deployments()
            needed = {arm for arm, d in current.items() if d["bundle_sha256"] != bundle["sha256"]}
            if needed:
                deploy.spaces(api, config, secrets, output, needed)
            def ready():
                try:
                    return all(d["bundle_sha256"] == bundle["sha256"] for d in deployments().values())
                except (httpx.HTTPError, ValueError):
                    return False
            wait_until(ready)
            state["deployed"] = True
        if not state.get("ui_passed"):
            proc = start([ENV_PY, ROOT / "hf/runtime/ui_smoke.py", "--out", output / "ui-smoke.json"], output / "ui-smoke.log")
            if proc.wait(timeout=600) != 0:
                raise RuntimeError("Deployed train/test UI isolation test failed")
            state["ui_passed"] = True
        state["phase"] = "optimizer_smokes"
        persist()
        smokes = {arm: ensure_job("train", arm, "smoke", flavor=config["compute"]["training_flavor"], timeout="2h") for arm in baselines}
        wait_until(lambda: completed(smokes))
        state["phase"] = "starting_training"
        persist()
        trains = {arm: ensure_job("train", arm, "long", flavor=config["compute"]["training_flavor"], timeout="24h", baseline_job=baselines[arm], smoke_job=smokes[arm])
                  for arm in baselines}
        for arm, job_id in trains.items():
            ensure_job("coordinator", arm, "long", flavor="cpu-upgrade", timeout="36h", training_job=job_id)
        state.update(phase="training_submitted", training_jobs=trains, smoke_jobs=smokes, passed=False)
        persist()
        while True:
            finished = True
            progress = {}
            for arm, job_id in trains.items():
                job = api.inspect_job(job_id=job_id, namespace=namespace)
                if job.status.stage in TERMINAL and job.status.stage != "COMPLETED":
                    raise RuntimeError(f"Training job {job_id} ended as {job.status.stage}")
                finished = finished and job.status.stage == "COMPLETED"
                name = "audit/metrics.jsonl" if arm == "blackbox" else "run/metrics.jsonl"
                path = output / "metrics" / (arm + ".jsonl")
                path.parent.mkdir(exist_ok=True)
                prefix = config["run_id"] + "/jobs/" + job.environment["RUN_OWNER"]
                from huggingface_hub.errors import EntryNotFoundError
                try:
                    api.download_bucket_files(os.environ["ARTIFACT_BUCKET"],
                        files=[(prefix + "/" + name, str(path))], raise_on_missing_files=True)
                except EntryNotFoundError:
                    progress[arm] = {"job_id": job_id, "stage": job.status.stage, "step": 0}
                    continue
                rows = []
                for line in path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "step" in row and "grad_norm" in row:
                        rows.append(row)
                if any(not math.isfinite(float(r[k])) for r in rows for k in ["loss", "grad_norm"] if k in r):
                    raise RuntimeError(f"Non-finite optimizer metrics in training job {job_id}")
                latest = rows[-1] if rows else {"step": 0}
                rewards = [r["reward"] for r in rows[-20:] if isinstance(r.get("reward"), (float, int))]
                progress[arm] = {"job_id": job_id, "stage": job.status.stage, "step": latest["step"],
                    "latest_reward": latest.get("reward"), "reward_mean_last20": sum(rewards)/len(rewards) if rewards else None,
                    "nonzero_gradient_updates": sum(r["grad_norm"] > 0 for r in rows)}
            stable = len(progress) == 2 and all(p["step"] >= 10 and p.get("nonzero_gradient_updates", 0) > 0 for p in progress.values())
            state.update(phase="completed" if finished else "training_active" if stable else "training_startup",
                         training_progress=progress, passed=stable)
            persist()
            if finished:
                return
            time.sleep(600 if stable else 60)
    except Exception as exc:
        state.update(phase="needs_attention", error_type=type(exc).__name__, error=str(exc), passed=False)
        persist()
        raise
