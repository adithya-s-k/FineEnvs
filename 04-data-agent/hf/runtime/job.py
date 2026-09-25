"""HF GPU task runner using the frozen native evaluators and trainers."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import ROOT, RUN, TOOLS, TRAIN_PY, ENV_PY, MODEL, REVISION, configure, ready, start, write_json


def inference_url():
    return "http://127.0.0.1:" + os.environ.get("LOCAL_INFERENCE_PORT", "8000")


def job_endpoint():
    if os.environ.get("LOCAL_RUNTIME") == "1":
        return os.environ.get("SLURM_JOB_ID", os.environ["RUN_OWNER"]), inference_url()
    from huggingface_hub import HfApi
    api = HfApi()
    for job in api.list_jobs(namespace=os.environ.get("HF_JOB_NAMESPACE", "HuggingEnvs"), labels={"experiment": "data-agent-daytona"}):
        if job.environment.get("RUN_OWNER") == os.environ["RUN_OWNER"]:
            info = api.inspect_job(job_id=job.id, namespace=os.environ.get("HF_JOB_NAMESPACE", "HuggingEnvs"))
            urls = info.status.expose_urls
            if not urls or len(urls) != 1:
                raise RuntimeError("Expected exactly one exposed vLLM endpoint")
            return job.id, urls[0].rstrip("/")
    raise RuntimeError("Could not identify this Job through its unique owner")


def serving(args, output, processes):
    env = dict(os.environ)
    env.update(MODEL=os.environ.get("CHECKPOINT_MODEL", MODEL), TRL_PROD=str(ROOT), VENV=str(ROOT / ".venv312"),
               PORT=os.environ.get("LOCAL_INFERENCE_PORT", "8000"), TP_SIZE="1", DP_SIZE=str(args.dp), MAX_MODEL_LEN="131072",
               GPU_MEMORY_UTILIZATION="0.85" if args.role == "train" else "0.90",
               TOOL_CALL_PARSER="qwen3_xml", REASONING_PARSER="qwen3", ENABLE_THINKING="0",
               ENFORCE_EAGER="0" if args.role == "train" or os.environ.get("LOCAL_RUNTIME") == "1" else "1", TUNNEL="none", SHORT_NAME="daytona-hf",
               VLLM_LOG=str(output / "vllm.log"), VLLM_SERVER_DEV_MODE="1", VLLM_USE_DEEP_GEMM="0",
               VLLM_DEEP_GEMM_WARMUP="skip", VLLM_USE_FLASHINFER_SAMPLER="0", READY_TIMEOUT_SEC="1200")
    extra = ['--dtype bfloat16', '--generation-config vllm', '--logprobs-mode processed_logprobs',
             '--return-tokens-as-token-ids', '--no-enable-prefix-caching',
             '--limit-mm-per-prompt {"image":0,"video":0}', '--gdn-prefill-backend triton',
             '--override-generation-config {"temperature":0.8,"top_p":1.0,"top_k":-1}',
             '--served-model-name Qwen/Qwen3.5-2B']
    if env["MODEL"] == MODEL:
        extra += ["--revision " + REVISION]
    if args.role == "train":
        extra += ['--weight-transfer-config {"backend":"nccl"}']
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("INFERENCE_GPU", "0")
    if args.dp > 1:
        extra += ["--data-parallel-rpc-port " + os.environ.get("VLLM_DP_RPC_PORT", "8950")]
    env["EXTRA_VLLM_ARGS"] = " ".join(extra)
    proc = start(["bash", RUN / "eval-source/serve_vllm_tunnel.sh"], output / "serving.log", env)
    processes.append(proc)
    ready(inference_url() + "/health", proc)
    from openenv.core.harness.capture.validate_llm import validate_llm
    report = validate_llm(inference_url() + "/v1", MODEL)
    if not report.trainable:
        raise RuntimeError("Inference preflight did not establish exact token capture")
    job_id, public = job_endpoint()
    ready(public + "/health", proc, headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"]}, seconds=120)
    return job_id, public


def bridge(output, processes):
    if os.environ.get("LOCAL_RUNTIME") == "1":
        server = "http://127.0.0.1:" + os.environ["LOCAL_ENV_PORT"]
        proc = start([ENV_PY, ROOT / "hf/runtime/local_environment.py"], output / "environment.log")
        processes.append(proc)
        ready(server + "/health", proc, seconds=300)
        import httpx
        info = httpx.get(server + "/deployment", timeout=30).raise_for_status().json()
        if info["bundle_sha256"] != os.environ["BUNDLE_SHA256"] or info["train_tasks"] != 1000 or info["test_tasks"] != 250:
            raise RuntimeError("Local environment task/source identity mismatch")
        write_json(output / "space_identity.json", info)
        os.environ["SPACE_URL"] = server
        return server
    proc = start([ENV_PY, ROOT / "hf/runtime/auth_bridge.py"], output / "bridge.log")
    processes.append(proc)
    ready("http://127.0.0.1:8100/health", proc, seconds=300)
    import httpx
    info = httpx.get("http://127.0.0.1:8100/deployment", timeout=30).raise_for_status().json()
    expected = os.environ.get("SPACE_BUNDLE_SHA256", os.environ["BUNDLE_SHA256"])
    if info["bundle_sha256"] != expected:
        raise RuntimeError("Space and Job runtime bundle hashes differ")
    if os.environ.get("COMPARISON_ARM") == "opencode" and os.environ.get("EVAL_SUITE") != "harbor":
        if (info.get("implementation") != "standalone-opencode" or info.get("train_tasks") != 1000
                or info.get("test_tasks") != 250 or info.get("opencode_version") != "1.18.31"
                or info.get("output_tokens") != {"train": 16384, "test": 4096}):
            raise RuntimeError("Standalone training service contract differs from the comparison")
    write_json(output / "space_identity.json", info)
    return "http://127.0.0.1:8100"


def blackbox_audit(output, server):
    from openenv.harbor.models import HarborRolloutResult
    from smoke_multiharness_tito import audit
    import httpx
    selected = {}
    for path in sorted((output / "traces").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("reward") in (0, 1) and row.get("n_turns", 0) > 0:
                selected.setdefault((row["harness"], row["index"]), row)

    def check(item):
        (harness, index), row = item
        result = HarborRolloutResult.model_validate_json(Path(row["capture_file"]).read_text())
        report, _ = audit(result, token_budget=131072)
        if not report["tito_pass"]:
            raise RuntimeError(f"TiTO failure: {harness}/{index}")
        trial = row["trial_name"]
        path = output / "trials" / trial / "result.json"
        if not path.exists():
            native = httpx.get(server + "/trial/" + trial + "/result", timeout=60).raise_for_status().json()
            write_json(path, native)
        return {"harness": harness, "index": index, **report}

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        reports = list(pool.map(check, selected.items()))
    pins = json.loads((ROOT / "hf/configs/deployment.json").read_text())["harness_pins"]
    counts = {h: sum(r["harness"] == h for r in reports) for h in pins}
    write_json(output / "final_tito.json", {"counts": counts, "tito_pass": bool(reports), "reports": reports})
    from score_comparison import summarize
    scores = summarize("blackbox", output)
    # Require actual harness versions for partial smokes too.
    for harness, versions in scores["harness_versions"].items():
        if versions and set(versions) != {pins[harness]}:
            raise RuntimeError(f"Unverified harness version: {harness}")
    return scores


def evaluate(args, output, server, public):
    suite = "blackbox" if os.environ.get("EVAL_SUITE") == "harbor" else args.arm
    if suite == "opencode":
        command = [ENV_PY, ROOT / "hf/runtime/eval_opencode.py", "--server", server,
            "--vllm-url", public + "/v1", "--model", MODEL, "--out", output,
            "--concurrency", os.environ.get("EVAL_CONCURRENCY", "35"),
            "--limit", str(args.limit or (2 if args.phase == "smoke" else 250)),
            "--backends", os.environ.get("EVAL_BACKENDS", "daytona,hf"),
            "--daytona-concurrency", os.environ.get("EVAL_DAYTONA_CONCURRENCY", "35"),
            "--hf-concurrency", os.environ.get("EVAL_HF_CONCURRENCY", "35")]
        if os.environ.get("EVAL_NO_RAMP") == "1":
            command += ["--no-ramp"]
        if args.phase == "baseline" and os.environ.get("EVAL_NO_RAMP") != "1":
            smoke = list(command)
            smoke[smoke.index("--out")+1] = output / "smoke"
            smoke[smoke.index("--limit")+1] = "2"
            smoke[smoke.index("--concurrency")+1] = "2"
            smoke[smoke.index("--daytona-concurrency")+1] = "2"
            smoke[smoke.index("--hf-concurrency")+1] = "2"
            process = start(smoke, output / "smoke-opencode.log")
            if process.wait() != 0:
                raise RuntimeError("Standalone backend rollout smoke failed; full baseline held")
        process = start(command, output / "eval-opencode.log")
        if process.wait() != 0:
            raise RuntimeError("Standalone OpenCode evaluation failed")
        return
    ceiling = int(os.environ.get("EVAL_CONCURRENCY", "35"))
    phases = [(8, 8), (min(32, ceiling), 32), (ceiling, 100)] if args.phase == "ramp" else [(8 if args.phase == "smoke" else ceiling, args.limit)]
    if args.phase == "baseline":
        # One immutable first-graded ledger spans the ramp and full evaluation.
        # Later passes revisit only infrastructure failures, including scored zeros
        # in the resume set so they can never become best-of-N samples.
        phases = [(8, 8), (min(32, ceiling), 32), (ceiling, 100)] + [(ceiling, 0)] * 4
    elif args.phase == "checkpoint":
        phases = [(ceiling, 0)] * 4
    if args.phase == "smoke":
        phases = [(8, 8 if suite == "blackbox" else 2)]
    records = []
    scores = json.loads((output / "canonical_scores.json").read_text()) if (output / "canonical_scores.json").exists() else {}
    for phase_index, (concurrency, limit) in enumerate(phases):
        if (output / "canonical_scores.json").exists() and json.loads((output / "canonical_scores.json").read_text()).get("comparison_ready"):
            break
        write_json(output / "eval_progress.json", {"stage": phase_index, "concurrency": concurrency,
                   "max_new_rollouts": limit, "started_at": time.time(), "phase": args.phase})
        before = json.loads((output / "canonical_scores.json").read_text()).get("graded_cells", 0) if (output / "canonical_scores.json").exists() else 0
        if suite == "blackbox":
            arms = [{"name": "model", "base_url": public + "/v1", "model": MODEL, "api_key_env": "HF_TOKEN"}]
            write_json(output / "arms.json", arms)
            indices = "@" + str(RUN / "test_indices.txt")
            if args.phase == "smoke":
                indices = ",".join((RUN / "test_indices.txt").read_text().replace(",", " ").split()[:2])
                limit = 0
            cmd = [TRAIN_PY, "-u", RUN / "eval-source/eval_concurrent.py", "--server", server,
                   "--arms", output / "arms.json", "--harnesses", "opencode,claude-code,codex,mini-swe-agent",
                   "--split", str(RUN / "datasets/test"), "--indices", indices, "--repeat", "1",
                   "--temperature", "0.8", "--reward-key", "correctness,reward", "--sandbox", "daytona",
                   "--agent-timeout", "600", "--agent-step-limit", "17", "--max-retries", "3",
                   "--trace-dir", output / "traces", "--capture-dir", output / "captures", "--progress-every", "1",
                   "--concurrency", str(concurrency), "--server-concurrency", str(concurrency),
                   "--sandbox-concurrency", str(concurrency), "--max-new-rollouts", str(limit), "--out", output / "results.json"]
            if (output / "traces/eval_config.json").exists():
                cmd += ["--resume"]
        else:
            cmd = [TRAIN_PY, "-u", TOOLS / "eval_whitebox_native.py", "--run", RUN, "--server", server,
                   "--vllm-url", inference_url() + "/v1", "--out", output, "--concurrency", str(concurrency),
                   "--max-new-rollouts", str(limit)]
        began = time.monotonic()
        proc = start(cmd, output / f"eval-stage{phase_index}-c{concurrency}.log")
        code = proc.wait()
        if code not in (0, 2):
            raise RuntimeError(f"Native evaluator failed: {code}")
        if suite == "blackbox":
            scores = blackbox_audit(output, server)
            scores["training_arm"] = args.arm
            scores["arm"] = args.arm
            scores["evaluation_suite"] = "harbor"
            write_json(output / "canonical_scores.json", scores)
        else:
            from score_comparison import summarize
            scores = summarize("whitebox", output)
        elapsed = time.monotonic() - began
        graded = scores["graded_cells"] - before
        expected = (8 if suite == "blackbox" else 2) if args.phase == "smoke" else limit
        if expected and graded < 0.9 * expected:
            raise RuntimeError(f"Scale gate failed: only {graded}/{expected} graded")
        hourly = {"a100-large": 2.5, "a100x4": 10, "h200x2": 10, "h200": 5}.get(os.environ["JOB_FLAVOR"])
        records.append({"stage": phase_index, "concurrency": concurrency, "new_graded": graded, "elapsed_s": elapsed,
                        "graded_per_minute": graded * 60 / elapsed,
                        "compute_usd_per_1000": hourly * elapsed / 3600 * 1000 / graded if graded and hourly else None})
        write_json(output / "scalability.json", records)
        print(json.dumps(records[-1]), flush=True)
    if args.phase in ("baseline", "checkpoint") and not scores["comparison_ready"]:
        raise RuntimeError("Full eval coverage/TiTO/version gate did not pass")
    if args.phase == "smoke" and scores["graded_cells"] != (8 if suite == "blackbox" else 2):
        raise RuntimeError("Smoke did not grade every requested cell")
    write_json(output / "eval_progress.json", {"finished_at": time.time(), "graded_cells": scores["graded_cells"],
               "comparison_ready": scores["comparison_ready"], "phase": args.phase})


def training_command(args, output, server):
    save = 2 if args.phase == "smoke" else 50
    config = json.loads((ROOT / "hf/configs/deployment.json").read_text())
    harnesses = config["arms"][args.arm].get("training_harnesses", ["opencode"])
    schedule = "reference_schedule.json" if len(harnesses) > 1 else "opencode_schedule.json"
    if args.arm in {"blackbox", "opencode"}:
        entrypoint = "train_standalone_comparison.py" if args.arm == "opencode" else "train_harbor_multi.py"
        cmd = [TRAIN_PY, "-u", RUN / "source/HuggingEnvs/04-data-agent/train" / entrypoint,
               "--server", server, "--vllm-url", inference_url(), "--model", MODEL,
               "--model-revision", REVISION, "--split", RUN / "datasets/train", "--harnesses", ",".join(harnesses),
               "--sandbox", "daytona", "--harness-schedule", RUN / schedule,
               "--task-indices", "@" + str(RUN / "train_indices.txt"), "--learning-rate", "3e-6",
               "--num-generations", "8", "--max-inflight", "32", "--max-staleness", "4", "--grad-accum", "4",
               "--atomic-rollouts", "--max-outstanding-rollouts", "16", "--max-row-tokens", "131072",
               "--per-device-batch-size", "4", "--reward-key", "reward", "--agent-step-limit", "17",
               "--agent-timeout", "600", "--token-budget", "40960", "--max-completion-length", "16384",
               "--dtype", "bfloat16", "--top-p", "1.0", "--temperature", "0.8", "--audit-dir", output / "audit",
               "--project", f"daytona-{args.arm}-qwen35-2b", "--save-steps", str(save), "--output-dir", output / "run"]
    else:
        cmd = [TRAIN_PY, "-u", TOOLS / "train_whitebox_daytona.py", "--run", RUN, "--server", server,
               "--vllm-url", inference_url(), "--output-dir", output / "run", "--save-steps", str(save)]
    return cmd


def train(args, output, server, public, publisher):
    os.environ["ROLLOUT_LLM_URL"] = public
    os.environ["ROLLOUT_LLM_API_KEY"] = os.environ["HF_TOKEN"]
    for key in ["TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_BUCKET_ID", "TRACKIO_DATASET_ID"]:
        os.environ.pop(key, None)
    os.environ["TRACKIO_STORAGE_MODE"] = "jsonl"
    os.environ["TRACKIO_DIR"] = str(output / "trackio")
    cmd = training_command(args, output, server)
    write_json(output / "training_recipe.json", {"arm": args.arm, "phase": args.phase,
               "command": list(map(str, cmd)), "space_bundle_sha256": os.environ.get("SPACE_BUNDLE_SHA256", os.environ["BUNDLE_SHA256"]),
               "job_bundle_sha256": os.environ["BUNDLE_SHA256"], "initialization": "pinned base; then native full-state restore for smoke"})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("TRAIN_GPU", "1")}
    if args.phase == "smoke":
        restored = output / "remote-resume/checkpoint-2"
        for steps, name, extra in [(2, "first", []), (4, "resumed", ["--resume-from-checkpoint", restored])]:
            proc = start(cmd + ["--max-steps", str(steps)] + extra, output / f"train-{name}.log", env)
            if proc.wait() != 0:
                raise RuntimeError(f"{name} training smoke failed")
            publisher.sync()
            if steps == 2:
                from checkpoint_store import restore
                restore(publisher.dest + "/run/checkpoint-2", restored, arm=args.arm,
                        bundle_sha256=os.environ["BUNDLE_SHA256"])
        from training_smoke import validate
        validate(output, args.arm)
    else:
        if not os.environ.get("VERIFIED_SMOKE_MANIFEST"):
            raise RuntimeError("Long training requires a verified optimizer/save/resume manifest")
        proof = json.loads(Path(os.environ["VERIFIED_SMOKE_MANIFEST"]).read_text())
        if not (proof.get("passed") and proof.get("arm") == args.arm and
                proof.get("bundle_sha256") == os.environ["BUNDLE_SHA256"] and
                proof.get("remote_restore_verified") and proof.get("tito_pass") and proof.get("weights_updated")):
            raise RuntimeError("Training smoke provenance or optimizer evidence does not match this run")
        cmd += ["--max-steps", "1000", "--max-train-seconds", "82200", "--checkpoint-max-seconds", "3600"]
        if os.environ.get("RESUME_CHECKPOINT"):
            cmd += ["--resume-from-checkpoint", os.environ["RESUME_CHECKPOINT"]]
        proc = start(cmd, output / "train.log", env)
        if proc.wait() != 0:
            raise RuntimeError("Trainer exited with an error")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=["eval", "train", "coordinator"], required=True)
    p.add_argument("--arm", choices=["blackbox", "whitebox", "opencode"], required=True)
    p.add_argument("--phase", default="smoke")
    p.add_argument("--dp", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    configure()
    output = ROOT / "outputs" / os.environ["RUN_OWNER"]
    output.mkdir(parents=True, exist_ok=True)
    if args.role == "coordinator":
        if args.phase == "qualify":
            from qualify_training import run
            run(output)
        elif args.phase == "setup":
            from setup_pipeline import run
            run(output)
        else:
            from coordinator import run
            run(output, args.arm)
        return
    processes = []
    from artifacts import Publisher
    publisher = Publisher(output)
    publisher.start()
    from telemetry import Telemetry
    telemetry = Telemetry(output, inference_url())
    telemetry.start()
    status = {"arm": args.arm, "phase": args.phase, "started_at": time.time(), "passed": False}
    logger = None
    logger_stop = output / "trackio-stop"
    write_json(output / "status.json", status)
    try:
        if args.role == "eval" and args.phase == "checkpoint":
            from checkpoint_store import restore_model
            source = os.environ["CHECKPOINT_PREFIX"]
            sha = os.environ["CHECKPOINT_MANIFEST_SHA"]
            model = output / "inference-model"
            manifest = restore_model(source, model, arm=args.arm,
                                     bundle_sha256=os.environ["BUNDLE_SHA256"], manifest_sha256=sha)
            if manifest["step"] != int(os.environ["CHECKPOINT_STEP"]):
                raise ValueError("Checkpoint optimizer step differs from the queued evaluation")
            os.environ["CHECKPOINT_MODEL"] = str(model)
            write_json(output / "checkpoint_evaluation.json", {"source": source, "manifest_sha256": sha,
                       "step": manifest["step"], "bundle_sha256": os.environ["BUNDLE_SHA256"]})
        if args.role == "train" and args.phase != "smoke":
            from checkpoint_store import download_json
            proof_path = output / "verified_smoke.json"
            download_json(os.environ["SMOKE_PREFIX"], "training_smoke_verified.json", proof_path)
            os.environ["VERIFIED_SMOKE_MANIFEST"] = str(proof_path)
        server = bridge(output, processes)
        if args.role == "train" and args.arm in {"blackbox", "opencode"}:
            from service_contract import check
            write_json(output / "service_contract.json", check(server, "", args.arm))
        job_id, public = serving(args, output, processes)
        write_json(output / "services.json", {"job_id": job_id, "public_vllm": public, "server": server,
                   "space": os.environ["SPACE_URL"], "tp": 1, "dp": args.dp, "flavor": os.environ["JOB_FLAVOR"]})
        if args.role == "eval":
            if os.environ.get("RESUME_EVAL_PREFIX"):
                if args.arm != "whitebox" or args.phase != "baseline":
                    raise ValueError("This saved baseline restore is for whitebox baseline evaluations")
                from eval_evidence import restore_whitebox
                restore_whitebox(output, os.environ["RESUME_EVAL_PREFIX"])
            evaluate(args, output, server, public)
        else:
            logger_env = {**os.environ, "TRACKIO_DIR": str(output / "trackio"),
                          "TRAINING_SMOKE": "1" if args.phase == "smoke" else "0"}
            logger = start([TRAIN_PY, ROOT / "hf/runtime/logging_sync.py", "--out", output,
                            "--arm", args.arm, "--watch", "--stop-file", logger_stop],
                           output / "trackio-sync.log", logger_env)
            try:
                train(args, output, server, public, publisher)
            finally:
                logger_stop.touch()
                if logger.wait(timeout=180) != 0:
                    raise RuntimeError("Training Trackio persistence failed")
        status["passed"] = True
    except Exception as exc:
        status["error_type"] = type(exc).__name__
        raise
    finally:
        status["finished_at"] = time.time()
        write_json(output / "status.json", status)
        telemetry.finish()
        if logger is not None and logger.poll() is None:
            os.killpg(logger.pid, signal.SIGTERM)
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        publisher.finish()
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
