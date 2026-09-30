"""Run both disposable optimizer smokes without restarting active eval Spaces."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from common import ROOT, write_json
from checkpoint_store import download_json
from coordinator import TERMINAL


def run(output):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError
    sys.path.insert(0, str(ROOT / "hf"))
    import deploy
    api = HfApi()
    config = json.loads((ROOT / "hf/configs/deployment.json").read_text())
    policy = config["training_smoke"]
    output.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(Path("/bundle/bundle.json").read_text())
    bundle.update(repo=os.environ["BUNDLE_REPO"], revision=os.environ["BUNDLE_REVISION"])
    write_json(output / "bundle_uploaded.json", bundle)
    namespace = config["namespace"]
    destination = "hf://buckets/" + os.environ["ARTIFACT_BUCKET"] + "/" + os.environ["RUN_ID"] + "/jobs/" + os.environ["RUN_OWNER"]
    state = {"phase": "qualifying", "bundle_sha256": bundle["sha256"], "jobs": {}, "passed": False,
             "long_training_launched": False, "standalone_gate": "wait for baseline Daytona cohort to release Space capacity"}

    def persist():
        state["updated_at"] = time.time()
        write_json(output / "qualification.json", state)
        api.sync_bucket(str(output), destination, quiet=True)

    def launch(arm):
        if arm in state["jobs"]:
            return
        matches = [j for j in api.list_jobs(namespace=namespace, labels={"role": "train", "arm": arm, "phase": "smoke"})
                   if j.environment.get("BUNDLE_SHA256") == bundle["sha256"]]
        if len(matches) > 1:
            raise RuntimeError("Multiple smoke jobs for the same arm and bundle")
        if matches:
            state["jobs"][arm] = {"id": matches[0].id, "stage": matches[0].status.stage}
            return
        state["jobs"][arm] = {"stage": "submitting"}
        persist()
        args = argparse.Namespace(role="train", arm=arm, phase="smoke", flavor=config["compute"]["training_flavor"], timeout="2h",
            dp=1, limit=0, resume_eval_owner=None, training_job=None, baseline_job=None, smoke_job=None,
            baseline_job_map=None, space_bundle_sha=policy["space_bundle_pins"][arm])
        secrets = {k: os.environ[k] for k in ["HF_TOKEN", "DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET"] if os.environ.get(k)}
        value = deploy.submit(api, config, secrets, output, args)
        state["jobs"][arm] = {"id": value["id"], "stage": value["stage"]}

    try:
        peers = [j for j in api.list_jobs(namespace=namespace, labels={"role": "coordinator", "phase": "qualify"})
                 if j.status.stage not in TERMINAL and j.environment.get("BUNDLE_SHA256") == bundle["sha256"]]
        if peers and min(peers, key=lambda j: j.id).environment["RUN_OWNER"] != os.environ["RUN_OWNER"]:
            raise RuntimeError("Another coordinator owns qualification for this bundle")
        launch("whitebox")
        while True:
            if "opencode" not in state["jobs"]:
                baseline = api.inspect_job(job_id=policy["opencode_wait_for_daytona_baseline_job"], namespace=namespace)
                source = "hf://buckets/" + baseline.environment["ARTIFACT_BUCKET"] + "/" + baseline.environment["RUN_ID"] + "/jobs/" + baseline.environment["RUN_OWNER"] + "/daytona"
                try:
                    scores = download_json(source, "scores.json", output / "daytona-baseline-scores.json", api)
                except EntryNotFoundError:
                    scores = {}
                state["daytona_baseline_graded"] = scores.get("graded_cells", 0)
                if scores.get("comparison_ready") and scores.get("expected_cells") == 250:
                    launch("opencode")
                elif baseline.status.stage in TERMINAL:
                    raise RuntimeError("Daytona baseline ended without complete TiTO-qualified coverage")
            complete = len(state["jobs"]) == 2
            for arm, item in state["jobs"].items():
                job = api.inspect_job(job_id=item["id"], namespace=namespace)
                item["stage"] = job.status.stage
                if job.status.stage in TERMINAL and job.status.stage != "COMPLETED":
                    raise RuntimeError(f"{arm} optimizer smoke ended {job.status.stage}")
                complete = complete and job.status.stage == "COMPLETED"
                if job.status.stage == "COMPLETED" and "proof" not in item:
                    source = "hf://buckets/" + job.environment["ARTIFACT_BUCKET"] + "/" + job.environment["RUN_ID"] + "/jobs/" + job.environment["RUN_OWNER"]
                    proof = download_json(source, "training_smoke_verified.json", output / f"{arm}-verified.json", api)
                    if not proof.get("passed") or proof.get("bundle_sha256") != bundle["sha256"]:
                        raise RuntimeError("Smoke evidence does not match the qualification bundle")
                    item["proof"] = proof
            persist()
            if complete:
                state.update(phase="complete", passed=True)
                persist()
                return
            time.sleep(60)
    except Exception as exc:
        state.update(phase="needs_attention", error_type=type(exc).__name__, error=str(exc))
        persist()
        raise
