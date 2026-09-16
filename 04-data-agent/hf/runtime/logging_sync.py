"""Durable metrics replay through native Trackio, without optimizer network calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from common import configure, write_json


def checkpoint_scores(output):
    """The logging process polls score artifacts; the optimizer never waits on HF."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError
    from checkpoint_store import bucket_location, download_json
    dest = output / "checkpoint-scores"
    dest.mkdir(exist_ok=True)
    api = HfApi()
    try:
        baseline = os.environ.get("BASELINE_PREFIX")
        if baseline and not (dest / "step-000000.json").exists():
            scores = download_json(baseline, "canonical_scores.json", output / "baseline_scores.json", api)
            write_json(dest / "step-000000.json", {"step": 0, "scores": scores, "source": {"job_id": os.environ.get("BASELINE_JOB")}})
        source = os.environ.get("COORDINATION_PREFIX")
        if source:
            bucket, prefix = bucket_location(source + "/scores")
            for item in api.list_bucket_tree(bucket, prefix=prefix, recursive=False):
                name = Path(item.path).name
                if name.startswith("step-") and name.endswith(".json") and not (dest / name).exists():
                    api.download_bucket_files(bucket, files=[(item.path, str(dest / name))], raise_on_missing_files=True)
    except EntryNotFoundError:
        pass  # No completed checkpoint evaluation yet.
    except Exception as exc:
        write_json(output / "trackio_remote_error.json", {"time": time.time(), "type": type(exc).__name__})
    return [json.loads(p.read_text()) for p in sorted(dest.glob("step-*.json"))]


def sync_metrics(output: Path, arm: str, step=0, smoke=False):
    configure()
    import trackio_multi4 as native
    training_smoke = os.environ.get("TRAINING_SMOKE") == "1"
    project = f"daytona-{arm}-qwen35-2b" + ("-integration" if smoke else "-smoke" if training_smoke else "")
    owner = os.environ.get("RUN_OWNER", "local")
    metadata = {"arm": arm, "sandbox": "daytona", "hf_run": os.environ.get("RUN_ID"),
                "bundle_sha256": os.environ.get("BUNDLE_SHA256"), "flavor": os.environ.get("JOB_FLAVOR"),
                "integration_test": smoke or training_smoke}
    events = []
    if smoke:
        events.append(native.event(project, owner, 0, {"integration/offline_online_roundtrip": 1.0}, metadata))
    else:
        for path in [output / "audit/metrics.jsonl", output / "run/metrics.jsonl"]:
            for row in native.read_metrics(path):
                events.append(native.event(project, owner, row["step"],
                                           native.scalars({k: v for k, v in row.items() if k != "step"}, "train/"), metadata))
        for record in checkpoint_scores(output):
            score = record["scores"]
            if not score.get("comparison_ready"):
                continue
            values = {"eval/pass_at_1": score["average_pass_at_1"], "eval/graded_cells": score["graded_cells"]}
            for harness, item in score["harnesses"].items():
                values[f"eval/{harness}/pass_at_1"] = item["pass_at_1"]
                for level, detail in item["difficulty"].items():
                    values[f"eval/{harness}/{level}/pass_at_1"] = detail["pass_at_1"]
            events.append(native.event(project, owner, record["step"], values, metadata, identity=record["source"]))
        scores = native.read_json(output / "canonical_scores.json", {})
        if scores.get("comparison_ready"):
            values = {"eval/pass_at_1": scores["average_pass_at_1"], "eval/graded_cells": scores["graded_cells"]}
            for h, score in scores["harnesses"].items():
                values[f"eval/{h}/pass_at_1"] = score["pass_at_1"]
                for level, item in score["difficulty"].items():
                    values[f"eval/{h}/{level}/pass_at_1"] = item["pass_at_1"]
            events.append(native.event(project, "evaluation-curve", step, values, metadata))
    if not events:
        return
    ledger = output / "trackio-events.jsonl"
    existing = {json.loads(line)["log_id"] for line in ledger.read_text().splitlines() if line.strip()} if ledger.exists() else set()
    events = [event for event in events if event["log_id"] not in existing]
    if not events:
        return
    native.import_events(events)
    native.backup_project(project, output / "trackio-backup")
    with ledger.open("a") as stream:
        for event in events:
            stream.write(json.dumps(event) + "\n")
    if not os.environ.get("TRACKIO_SPACE"):
        from trackio.sqlite_storage import SQLiteStorage
        write_json(output / "trackio_verified.json", {"passed": True, "project": project,
                   "run": owner, "local_database": str(SQLiteStorage.get_project_db_path(project)),
                   "mode": "offline", "remote_storage": "run artifact bucket", "native_remote_readback": False,
                   "updated_at": time.time(), "unique_events": len(existing) + len(events)})
        return
    config = {"logging": {"project": project, "space_id": os.environ["TRACKIO_SPACE"],
                          "bucket_id": "HuggingEnvs/data-agent-daytona-trackio"}}
    from trackio.deploy import sync_incremental
    from trackio.remote_client import RemoteClient
    sync_incremental(project, os.environ["TRACKIO_SPACE"], private=False, pending_only=False)
    client = RemoteClient(os.environ["TRACKIO_SPACE"], hf_token=os.environ["HF_TOKEN"],
                          httpx_kwargs={"timeout": 60})
    configuration = native.configuration_records(project)
    if configuration:
        client.predict(api_name="/bulk_log", logs=configuration, hf_token=os.environ["HF_TOKEN"])
    if smoke:
        from trackio.remote_client import RemoteClient
        from trackio.sqlite_storage import SQLiteStorage
        client = RemoteClient(os.environ["TRACKIO_SPACE"], hf_token=os.environ["HF_TOKEN"],
                              httpx_kwargs={"timeout": 60})
        # Native read-back, not merely an accepted upload request.
        runs = client.predict(api_name="/get_runs_for_project", project=project)
        assert owner in str(runs), f"Trackio read-back did not contain integration run {owner}"
        logs = client.predict(api_name="/get_logs", project=project, run=owner, run_id=None, scalar_only=True)
        assert logs and "integration/offline_online_roundtrip" in str(logs)
        write_json(output / "trackio_verified.json", {"passed": True, "project": project,
                   "run": owner, "local_database": str(SQLiteStorage.get_project_db_path(project)),
                   "space": os.environ["TRACKIO_SPACE"], "native_remote_readback": True})


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--stop-file", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.watch and a.stop_file is None:
        p.error("--watch requires --stop-file")
    while True:
        sync_metrics(a.out, a.arm, a.step, a.smoke)
        write_json(a.out / "trackio_collector.json", {"last_success": time.time(), "pid": os.getpid()})
        if not a.watch or a.stop_file.exists():
            break
        time.sleep(30)
