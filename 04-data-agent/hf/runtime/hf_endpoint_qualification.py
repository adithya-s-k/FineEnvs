"""Twenty-task, four-harness qualification against an isolated HF capture port."""
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, "/workspace/repro/hf/runtime")
import hard_curriculum_job as runtime
from artifacts import Publisher
from checkpoint_store import restore_model

OVERLAY = Path(__file__).parent
sys.path.insert(0, str(OVERLAY))
os.environ["PYTHONPATH"] = str(OVERLAY) + os.pathsep + os.environ["PYTHONPATH"]


def selected_rows(output):
    selected = {}
    for path in sorted((output / "traces").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("reward") is not None and row.get("n_turns", 0) > 0:
                selected.setdefault((row["harness"], row["index"]), row)
    return selected


def main():
    import httpx
    root = runtime.ROOT
    runtime.verify_bundle()
    manifest = runtime.read(OVERLAY / "overlay_manifest.json")
    for name, expected in manifest.items():
        if hashlib.sha256((OVERLAY / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Qualification overlay changed")
    c = runtime.read(root / "config.json")
    output = root / "outputs" / os.environ["RUN_OWNER"]
    output.mkdir(parents=True, exist_ok=False)
    runtime.write_json(output / "qualification_inputs.json", {
        "bundle_sha256": os.environ["BUNDLE_SHA256"], "overlay_files": manifest,
        "checkpoint_prefix": os.environ["CHECKPOINT_PREFIX"], "indices": list(range(20)),
        "harnesses": c["harnesses"], "concurrency": 8, "full_benchmark": False})
    publisher = Publisher(output)
    publisher.start()
    processes = []
    try:
        command = runtime.eval_command(c, output)
        for flag, value in [("--indices", ",".join(map(str, range(20)))),
                            ("--concurrency", "8"), ("--server-concurrency", "8"),
                            ("--sandbox-concurrency", "8")]:
            command[command.index(flag) + 1] = value
        probe = runtime.start(command + ["--dry-run"], output / "eval-cli-preflight.log")
        if probe.wait(timeout=120):
            raise RuntimeError("Evaluator preflight failed")
        model = output / "inference-model"
        evidence = restore_model(os.environ["CHECKPOINT_PREFIX"], model, arm="blackbox",
            bundle_sha256=os.environ["BUNDLE_SHA256"],
            manifest_sha256=os.environ["CHECKPOINT_MANIFEST_SHA"])
        runtime.write_json(output / "checkpoint_evaluation.json", evidence)
        endpoint = f"https://{os.environ['JOB_ID']}--8201.hf.jobs"
        auth = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]}
        original_start, original_ready, original_get = runtime.start, runtime.ready, httpx.get

        def start(command, log, env=None):
            command = list(command)
            if "openenv.cli" in command:
                i = command.index("-m")
                command[i:i+2] = [OVERLAY / "hf_endpoint_serve.py"]
            return original_start(command, log, env)

        def ready(url, *args, **kwargs):
            if url.startswith(endpoint + "/"):
                kwargs["headers"] = auth
            return original_ready(url, *args, **kwargs)

        def get(url, **kwargs):
            if url.startswith(endpoint + "/"):
                kwargs["headers"] = auth
            return original_get(url, **kwargs)

        with patch.object(runtime, "start", start), patch.object(runtime, "ready", ready), patch.object(httpx, "get", get):
            processes = runtime.services(c, output, model, False)
        # Stop after a bounded first batch, then resume the identical saved cohort.
        first = runtime.start(command + ["--max-new-rollouts", "8"], output / "eval-first.log")
        processes.append(first)
        if first.wait() not in (0, 2):
            raise RuntimeError("First evaluation batch failed")
        before = selected_rows(output)
        runtime.write_json(output / "resume_before.json", {"graded_cells": len(before),
            "cells": [list(key) for key in sorted(before)]})
        if not before:
            raise RuntimeError("No graded cells in first batch; inspect transport before continuing")
        for attempt in range(3):
            process = runtime.start(command + ["--resume"], output / "eval-resume.log")
            processes.append(process)
            if process.wait() not in (0, 2):
                raise RuntimeError("Resumed evaluator failed")
            rows = selected_rows(output)
            if any(rows.get(key) != value for key, value in before.items()):
                raise RuntimeError("Resume replaced an existing graded result")
            # The frozen scorer audits tokens, logprobs and harness versions.
            score = runtime.score_eval(c, output, False)
            expected = {(h, i) for h in c["harnesses"] for i in range(20)}
            complete = set(rows) == expected
            result = {**score, "expected_cells": 80, "complete": complete,
                "comparison_ready": False, "qualification_only": True,
                "resume_preserved_graded_cells": True, "passed": complete and score["tito_pass"]}
            runtime.write_json(output / "endpoint_qualification.json", result)
            publisher.sync()
            if complete:
                break
        if not result["passed"]:
            raise RuntimeError("Endpoint qualification incomplete")
        token = os.environ["HF_TOKEN"].encode()
        for path in output.rglob("*"):
            if path.is_file() and "inference-model" not in path.parts and path.stat().st_size < 100_000_000:
                if token in path.read_bytes():
                    raise RuntimeError("Credential found in local artifact")
        runtime.write_json(output / "credential_audit.json", {"raw_hf_token_in_artifacts": False})
    finally:
        runtime.stop(processes)
        publisher.finish()


if __name__ == "__main__":
    main()
