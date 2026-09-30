"""Replay trailing HF checkpoint scores into the finished trainer's Trackio artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def planned_stop_matches(job, plan):
    request = plan.get("final_checkpoint") or {}
    expected_source = ("hf://buckets/" + job.environment["ARTIFACT_BUCKET"] + "/"
                       + job.environment["RUN_ID"] + "/jobs/" + job.environment["RUN_OWNER"]
                       + "/run/checkpoint-" + str(request.get("step")))
    return (job.status.stage in {"CANCELED", "CANCELLED"}
            and job.environment["BUNDLE_SHA256"] == plan["bundle_sha256"]
            and request.get("training_job") == plan["training_job"]
            and request.get("user_requested_stop") is True
            and request.get("full_checkpoint_verified") is True
            and type(request.get("step")) is int and request["step"] > 0
            and request.get("checkpoint") == expected_source
            and len(request.get("manifest_sha256", "")) == 64)


def replay(plan_path):
    from huggingface_hub import HfApi
    from dotenv import dotenv_values
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    base = plan_path.parent
    for name, expected in plan["files"].items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Prepared replay source changed: " + name)
    values = dotenv_values(plan["env_file"])
    os.environ["HF_TOKEN"] = values.get("HF_API_KEY") or values["HF_TOKEN"]
    api = HfApi()
    job = api.inspect_job(job_id=plan["training_job"], namespace=plan["namespace"])
    intentional_stop = planned_stop_matches(job, plan)
    if ((job.status.stage != "COMPLETED" and not intentional_stop)
            or job.environment["BUNDLE_SHA256"] != plan["bundle_sha256"]):
        raise ValueError("Replay requires a completed trainer with the qualified runtime")
    source = ("hf://buckets/" + job.environment["ARTIFACT_BUCKET"] + "/" +
              job.environment["RUN_ID"] + "/jobs/" + job.environment["RUN_OWNER"])
    output = base / "late-training-logs"
    output.mkdir(exist_ok=True)
    api.sync_bucket(source, str(output), quiet=True,
        include=["status.json", "audit/metrics.jsonl", "run/metrics.jsonl", "trackio-events.jsonl"])
    status = json.loads((output / "status.json").read_text())
    if not intentional_stop and (not status.get("finished_at") or not status.get("passed")):
        raise ValueError("Trainer has not published successful final status")
    destination = output / "checkpoint-scores"
    destination.mkdir(exist_ok=True)
    steps = set()
    coordinator_output = Path(plan.get("score_controller_output", base / "output"))
    for path in (coordinator_output / "decisions/scores").glob("step-*.json"):
        record = json.loads(path.read_text())
        evidence, scores = record["source"], record["scores"]
        if intentional_stop and (record["step"] != plan["final_checkpoint"]["step"]
                or evidence.get("manifest_sha256") != plan["final_checkpoint"]["manifest_sha256"]):
            raise ValueError("Stopped trainer replay must use its exact authorized final checkpoint")
        if (not scores.get("comparison_ready") or not scores.get("tito_pass") or
                scores.get("arm") != "whitebox" or scores.get("graded_cells") != 250 or
                evidence.get("bundle_sha256") != plan["bundle_sha256"] or
                record["step"] != evidence["step"] or
                evidence["source"] != source + "/run/checkpoint-" + str(record["step"])):
            raise ValueError("Unverified checkpoint scores cannot enter the training curve")
        target = destination / path.name
        if target.exists() and json.loads(target.read_text()) != record:
            raise ValueError("Conflicting completed scores for the same checkpoint")
        target.write_text(json.dumps(record, indent=2) + "\n")
        steps.add(record["step"])
    if not steps:
        raise ValueError("No completed checkpoint scores to replay")
    for name in ("TRACKIO_SPACE", "TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_BUCKET_ID", "TRACKIO_DATASET_ID"):
        os.environ.pop(name, None)
    os.environ.update(REPRO_ROOT=plan["logger_root"], TRACKIO_DIR=str(output / "trackio"),
        RUN_OWNER=job.environment["RUN_OWNER"], RUN_ID=job.environment["RUN_ID"],
        BUNDLE_SHA256=plan["bundle_sha256"], JOB_FLAVOR=job.flavor,
        TRAINING_SMOKE="1" if job.labels["phase"] == "smoke" else "0")
    for name in ("BASELINE_PREFIX", "BASELINE_JOB", "COORDINATION_PREFIX"):
        if job.environment.get(name):
            os.environ[name] = job.environment[name]
        else:
            os.environ.pop(name, None)
    sys.path.insert(0, str(Path(plan["logger_root"]) / "hf/runtime"))
    from common import configure
    configure()
    import trackio_multi4 as native
    ledger = output / "trackio-events.jsonl"
    # Rebuild a fresh local database from the durable event ledger. Merely
    # copying the dedup ledger would hide old events from the new database.
    previous = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    native.import_events(previous)
    from logging_sync import sync_metrics
    sync_metrics(output, "whitebox")
    events = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    recorded = {r["step"] for r in events if "eval/pass_at_1" in r["metrics"]}
    if not steps.issubset(recorded):
        raise ValueError("Trackio replay did not retain every completed checkpoint")
    # The GPU job is terminal, so its publisher cannot overwrite these late logs.
    api.sync_bucket(str(output), source, quiet=True, include=["checkpoint-scores/**",
        "trackio-events.jsonl", "trackio-backup/**", "trackio_verified.json"])
    receipt = {"passed": True, "training_job": job.id, "steps": sorted(steps),
               "previous_events": len(previous), "current_events": len(events),
               "training_artifacts": source, "offline_trackio_and_bucket": True,
               "provider_training_stage": job.status.stage, "user_requested_stop": intentional_stop}
    (base / "late-trackio-verified.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    replay(parser.parse_args().plan)
