"""Resumable full-cohort evaluation through the qualified HF endpoint."""
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


def restore_graded(output, c):
    """Restore first grades with their captures; never rerun a graded zero."""
    sources = json.loads(os.environ.get("EVAL_RESUME_PREFIXES", "[]"))
    if not sources and os.environ.get("EVAL_RESUME_PREFIX"):
        sources = [os.environ["EVAL_RESUME_PREFIX"]]
    if not sources:
        return {}
    from huggingface_hub import HfApi
    from checkpoint_store import bucket_location
    api = HfApi()
    selected, origins, old_config = {}, {}, None
    expected_indices = [int(x) for x in (runtime.ROOT / "inputs/test_indices.txt").read_text().replace(",", " ").split()]
    for number, source in enumerate(sources):
        bucket, prefix = bucket_location(source)
        original = output / "recovery-source" / str(number)
        original.mkdir(parents=True)
        names = ["traces/eval_config.json"] + ["traces/model--" + h + ".jsonl" for h in c["harnesses"]]
        api.download_bucket_files(bucket, files=[(prefix + "/" + n, str(original / n)) for n in names],
                                  raise_on_missing_files=False)
        rows = selected_rows(original)
        if not rows:
            continue
        config = runtime.read(original / "traces/eval_config.json")
        if config["indices"] != expected_indices:
            raise ValueError("Recovery source differs from the frozen task order")
        if old_config is not None and config != old_config:
            raise ValueError("Recovery evaluation protocols differ")
        old_config = config
        for key, row in rows.items():
            if key not in selected:
                selected[key] = row
                origins[key] = (bucket, prefix)
    if not selected:
        return {}

    receipts = []
    downloads = {}
    for key, row in selected.items():
        bucket, prefix = origins[key]
        old_capture = Path(row["capture_file"])
        parts = old_capture.parts
        offset = parts.index("captures")
        relative = Path(*parts[offset:])
        target = output / relative
        downloads.setdefault(bucket, []).extend([(prefix + "/" + str(relative), str(target)),
                      (prefix + "/trials/" + row["trial_name"] + "/result.json",
                       str(output / "trials" / row["trial_name"] / "result.json"))])
        receipts.append({"harness": key[0], "index": key[1], "original_capture": str(old_capture),
                         "restored_capture": str(target), "source_prefix": prefix})
        row["capture_file"] = str(target)
    for bucket, files in downloads.items():
        api.download_bucket_files(bucket, files=files, raise_on_missing_files=True)
    (output / "traces").mkdir(exist_ok=True)
    runtime.write_json(output / "traces/eval_config.json", old_config)
    for h in c["harnesses"]:
        rows = [row for (name, _), row in selected.items() if name == h]
        (output / "traces" / ("model--" + h + ".jsonl")).write_text(
            "".join(json.dumps(row) + "\n" for row in rows))
    # Fail closed if a retained capture or native harness version is invalid.
    runtime.score_eval(c, output, False)
    runtime.write_json(output / "recovery_sources.json", {"prefixes": sources, "cells": receipts})
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
        "checkpoint_prefix": os.environ["CHECKPOINT_PREFIX"], "indices": list(range(250)),
        "harnesses": c["harnesses"], "concurrency": 100, "inference_replicas": 4, "full_benchmark": True})
    publisher = Publisher(output)
    publisher.start()
    processes = []
    try:
        command = runtime.eval_command(c, output)
        for flag in ["--concurrency", "--server-concurrency", "--sandbox-concurrency"]:
            command[command.index(flag) + 1] = "100"
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
            if "vllm.entrypoints.openai.api_server" in command:
                command += ["--tensor-parallel-size", "1", "--data-parallel-size", "4"]
                env = {**env, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
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
        before = restore_graded(output, c)
        runtime.write_json(output / "resume_before.json", {"graded_cells": len(before)})
        for attempt in range(4):
            resume = bool(before) or attempt > 0
            process = runtime.start(command + (["--resume"] if resume else []), output / "eval.log")
            processes.append(process)
            if process.wait() not in (0, 2):
                raise RuntimeError("Evaluator failed")
            rows = selected_rows(output)
            if any(rows.get(key) != value for key, value in before.items()):
                raise RuntimeError("Recovery replaced an existing graded result")
            score = runtime.score_eval(c, output, False)
            runtime.write_json(output / "recovery_progress.json", {
                "graded_cells": len(rows), "reused_cells": len(before),
                "resume_preserved_graded_cells": True, "attempt": attempt + 1})
            if score["complete"]:
                runtime.write_json(output / "canonical_scores.json", score)
                logger = runtime.start([root / ".venv312/bin/python", root / "hf/runtime/hard_curriculum_logging.py",
                    "--output", output, "--config", root / "config.json"], output / "trackio-final.log")
                processes.append(logger)
                if logger.wait(timeout=180):
                    raise RuntimeError("Evaluation complete but Trackio publication failed")
                break
            publisher.sync()
        if not score["complete"]:
            raise RuntimeError("Incomplete evaluation; missing cells are not scores")
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
