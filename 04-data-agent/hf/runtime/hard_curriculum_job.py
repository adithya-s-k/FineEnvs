"""Isolated HF training, checkpoint publication and pass@1 evaluation."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(os.environ.get("REPRO_ROOT", "/workspace/repro"))


def configure():
    paths = [ROOT / "hf/runtime", ROOT / "source/HuggingEnvs/04-data-agent/train",
             ROOT / "source/trl", ROOT / "source/OpenEnv/src", ROOT / "source/OpenEnv/envs", ROOT / "eval"]
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    os.environ.update(OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", TRL_EXPERIMENTAL_SILENCE="1",
        OPENENV_E2B_STREAM_UPLOADS="1", HARBOR_SHARED_ENV_NAME="openenv-dataagent-tito-4gb-20260914")


configure()
from common import write_json, start, ready
from checkpoint_store import digest, bucket_location, download_json


def read(path):
    return json.loads(Path(path).read_text())


def verify_bundle():
    files = read(ROOT / "bundle_manifest.json")["files"]
    for name, expected in files.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Runtime input changed: {name}")
    schedule = [json.loads(x) for x in (ROOT / "inputs/execution_schedule.jsonl").read_text().splitlines()]
    if len(schedule) != 1000 or len({g["task_name"] for g in schedule}) != 500:
        raise ValueError("Wrong curriculum size")
    if set(Counter(g["task_name"] for g in schedule).values()) != {2}:
        raise ValueError("Each task must occur exactly twice")
    if any(g["difficulty"] != "hard" for g in schedule):
        raise ValueError("Expected hard-only curriculum")
    return len(files)


def restore_parent(c, output):
    from huggingface_hub import HfApi
    from checkpoint_artifacts import mark_saved, finalize_saved
    manifest = read(ROOT / "inputs/parent_manifest.json")
    target = output / "parent/checkpoint-500"
    target.mkdir(parents=True, exist_ok=False)
    bucket, prefix = bucket_location(c["hf"]["parent_prefix"])
    names = list(manifest["files"])
    if any(Path(n).name != n for n in names):
        raise ValueError("Invalid checkpoint file")
    HfApi().download_bucket_files(bucket, files=[(prefix + "/" + n, str(target / n)) for n in names],
                                  raise_on_missing_files=True)
    for name, expected in manifest["files"].items():
        if digest(target / name) != expected:
            raise ValueError(f"Parent checkpoint changed in transit: {name}")
    state = read(target / "rollout_state.json")
    write_json(output / "parent_origin.json", {"manifest": manifest, "original_rollout_state": state,
               "new_schedule_cursor": 0, "optimizer_unchanged": True})
    # Only the downloaded branch copy changes; original optimizer/model files stay byte-identical.
    write_json(target / "rollout_state.json", {**state, "prompt_index": 0})
    mark_saved(target, 500, c["model"], c["model_revision"])
    finalize_saved(target)
    return target


def services(c, output, model, training):
    python = ROOT / ".venv312/bin/python"
    env_python = ROOT / "OpenEnv/.venv/bin/python"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "VLLM_SERVER_DEV_MODE": "1",
           "VLLM_USE_DEEP_GEMM": "0", "VLLM_DEEP_GEMM_WARMUP": "skip", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    command = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
        "--served-model-name", c["model"], "--port", "8000", "--dtype", "bfloat16",
        "--trust-remote-code", "--generation-config", "vllm", "--max-model-len", "131072",
        "--gpu-memory-utilization", "0.85" if training else "0.90", "--no-enable-prefix-caching",
        "--limit-mm-per-prompt", '{"image":0,"video":0}', "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml", "--reasoning-parser", "qwen3", "--gdn-prefill-backend", "triton",
        "--default-chat-template-kwargs", '{"enable_thinking":false}', "--return-tokens-as-token-ids",
        "--logprobs-mode", "processed_logprobs", "--override-generation-config",
        '{"temperature":0.8,"top_p":1.0,"top_k":-1,"max_tokens":4096}']
    if training:
        command += ["--weight-transfer-config", '{"backend":"nccl"}']
    else:
        command += ["--enforce-eager"]
    processes = []
    try:
        engine = start(command, output / "vllm.log", env)
        processes.append(engine)
        ready("http://127.0.0.1:8000/health", engine, seconds=1800)
        split = ROOT / ("data/train" if training else "data/test")
        # Both datasets are served by this job's isolated Harbor instance.
        server_env = {**os.environ, "OPENENV_HARBOR_TRIALS_DIR": str(output / "trials"),
            "OPENENV_HARBOR_AGENT_VERSIONS": json.dumps(c["harness_versions"]), "MAX_CONCURRENT_ENVS": "128"}
        server = start([env_python, "-m", "openenv.cli", "harbor", "serve", "--dataset", str(ROOT / "data/train"),
            "--dataset", str(ROOT / "data/test"), "--llm-url", "http://127.0.0.1:8000/v1", "--model", c["model"],
            "--port", "8200", "--capture-port", "8201", "--expose", "gradio", "--max-output-tokens", "4096"],
            output / "openenv.log", server_env)
        processes.append(server)
        ready("http://127.0.0.1:8200/health", server, seconds=300)
        from openenv.core.harness.capture.validate_llm import validate_llm
        if not validate_llm("http://127.0.0.1:8000/v1", c["model"]).trainable:
            raise RuntimeError("vLLM did not pass token/logprob qualification")
        import httpx,re
        local = httpx.get("http://127.0.0.1:8201/health", timeout=30).raise_for_status().json()
        match = re.search(r"capture\s+:\d+\s+->\s+(https://\S+)", (output / "openenv.log").read_text())
        if not match:
            raise RuntimeError("Capture tunnel URL missing")
        ready(match[1] + "/health", server, seconds=120)
        remote = httpx.get(match[1] + "/health", timeout=30).raise_for_status().json()
        if not local.get("instance") or local["instance"] != remote.get("instance"):
            raise RuntimeError("Capture tunnel resolves to a different server")
        from openenv.harbor.client import HarborEnv
        client = HarborEnv(base_url="http://127.0.0.1:8200")
        try:
            if client.num_tasks(str(ROOT / "data/train")) != 500 or client.num_tasks(str(ROOT / "data/test")) != 250:
                raise RuntimeError("Wrong task catalog")
        finally:
            client.close()
        write_json(output / "preflight.json", {"tito_probe": True, "capture_instance": local["instance"],
                   "capture_public_matches_local": True, "train_tasks": 500, "test_tasks": 250})
        return processes
    except BaseException:
        stop(processes)
        raise


def stop(processes):
    for p in reversed(processes):
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
    for p in processes:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)


def train_command(c, output, checkpoint, *, smoke=False):
    train = c["training"]
    return [str(ROOT / ".venv312/bin/python"), "-u", str(ROOT / "hf/runtime/hard_curriculum_train.py"),
        "--server", "http://127.0.0.1:8200", "--vllm-url", "http://127.0.0.1:8000", "--model", c["model"],
        "--model-revision", c["model_revision"], "--resume-from-checkpoint", str(checkpoint),
        "--split", str(ROOT / "data/train"), "--harnesses", "+".join(c["harnesses"]), "--sandbox", "e2b",
        "--harness-schedule", str(ROOT / "inputs/harness_schedule.json"),
        "--task-indices", "@" + str(ROOT / "inputs/train_indices.txt"), "--learning-rate", "3e-6",
        "--num-generations", "8", "--max-inflight", "32", "--max-staleness", "4", "--grad-accum", "4",
        "--atomic-rollouts", "--max-outstanding-rollouts", "16", "--max-row-tokens", "131072",
        "--per-device-batch-size", "4", "--reward-key", "correctness,reward", "--agent-step-limit", "17",
        "--agent-timeout", "600", "--token-budget", "40960", "--max-completion-length", "4096",
        "--temperature", "0.8", "--top-p", "1.0", "--top-k", "0", "--seed", "1", "--dtype", "bfloat16",
        "--save-steps", "2" if smoke else "50", "--max-steps", "2501", "--checkpoint-max-seconds", "3600",
        "--audit-dir", str(output / "audit"), "--output-dir", str(output / "run"),
        "--project", c["hf"]["run_id"], "--run-name", os.environ["RUN_OWNER"]]


def train(c, output, parent, publisher, smoke):
    from checkpoint_store import restore
    command = train_command(c, output, parent, smoke=smoke)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "TRACKIO_STORAGE_MODE": "jsonl",
        "PYTHONFAULTHANDLER": "1", "FLA_TILELANG": "1",
        "TRACKIO_DIR": str(output / "trackio"), "CURRICULUM_GROUP_LIMIT": "8" if smoke else "1000",
        "CURRICULUM_AUDIT": str(output / "audit"),
        "CURRICULUM_SCHEDULE_SHA256": digest(ROOT / "inputs/execution_schedule.jsonl")}
    write_json(output / "training_command.json", {"command": command, "group_limit": int(env["CURRICULUM_GROUP_LIMIT"])})
    if smoke:
        first = list(command)
        first[first.index("--max-steps") + 1] = "502"
        process = start(first, output / "train-first.log", env)
        returncode = process.wait()
        write_json(output / 'train-first-exit.json', {'returncode': returncode})
        if returncode != 0:
            raise RuntimeError(f"Initial optimizer smoke failed: exit {returncode}")
        publisher.sync()
        restored = output / "remote-resume/checkpoint-502"
        restore(publisher.dest + "/run/checkpoint-502", restored, arm="blackbox", bundle_sha256=os.environ["BUNDLE_SHA256"])
        command[command.index("--resume-from-checkpoint") + 1] = str(restored)
        env["CURRICULUM_RESUME_STATE"] = str(restored / "curriculum_state.json")
    process = start(command, output / "train.log", env)
    returncode = process.wait()
    write_json(output / 'train-exit.json', {'returncode': returncode})
    if returncode != 0:
        raise RuntimeError(f"Trainer failed: exit {returncode}; checkpoint and receipts remain in artifacts")
    checkpoints = sorted((output / "run").glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]))
    final = checkpoints[-1]
    state = read(final / "curriculum_state.json")
    if not state.get("schedule_exhausted"):
        raise RuntimeError("Trainer stopped before exhausting the finite task schedule")
    publisher.sync()
    write_json(output / "training_complete.json", {"step": int(final.name.split('-')[-1]),
        "schedule_exhausted": True, "admitted_rollouts": len(state["admitted_rollouts"]),
        "partial_groups": state["partial_groups_not_replayed_on_resume"]})
    return final


def eval_command(c, output, smoke=False):
    indices = "0,1" if smoke else "@" + str(ROOT / "inputs/test_indices.txt")
    command = [ROOT / ".venv312/bin/python", ROOT / "eval/eval_concurrent.py", "--server", "http://127.0.0.1:8200",
        "--vllm-url", "http://127.0.0.1:8000/v1", "--model", c["model"],
        "--harnesses", ",".join(c["harnesses"]), "--split", ROOT / "data/test", "--indices", indices,
        "--repeat", "1", "--temperature", "0.8", "--reward-key", "correctness,reward", "--sandbox", "e2b",
        "--agent-timeout", "600", "--agent-step-limit", "17", "--max-retries", "3",
        "--trace-dir", output / "traces", "--capture-dir", output / "captures", "--out", output / "results.json",
        "--concurrency", "4" if smoke else "50", "--server-concurrency", "4" if smoke else "50",
        "--sandbox-concurrency", "4" if smoke else "50"]
    return command


def evaluate(c, output, smoke=False):
    command = eval_command(c, output, smoke)
    for attempt in range(4):
        cmd = command + (["--resume"] if attempt else [])
        process = start(cmd, output / "eval.log")
        if process.wait() not in (0, 2):
            raise RuntimeError("Evaluator failed")
        score = score_eval(c, output, smoke)
        if score["complete"]:
            write_json(output / "canonical_scores.json", score)
            return score
    raise RuntimeError("Incomplete evaluation; missing cells are not scores")


def score_eval(c, output, smoke):
    from openenv.harbor.models import HarborRolloutResult
    from smoke_multiharness_tito import audit
    manifest = read(ROOT / "inputs/test_manifest.json")
    selected = {}
    for path in sorted((output / "traces").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("reward") is not None and row.get("n_turns", 0) > 0:
                selected.setdefault((row["harness"], row["index"]), row)
    reports = []
    for (harness, index), row in selected.items():
        result = HarborRolloutResult.model_validate_json(Path(row["capture_file"]).read_text())
        report, _ = audit(result, token_budget=131072)
        if not report["tito_pass"]:
            raise RuntimeError(f"TiTO audit failed for {harness}/{index}")
        native = read(output / "trials" / row["trial_name"] / "result.json")
        if (native.get("agent_info") or {}).get("version") != c["harness_versions"][harness]:
            raise RuntimeError(f"Harness version differs from the frozen evaluation: {harness}")
        reports.append({"harness": harness, "index": index, "difficulty": manifest["tasks"][index]["difficulty"],
                        "correct": row["reward"] == 1, "tito_pass": True})
    indices = [0, 1] if smoke else [int(x) for x in (ROOT / "inputs/test_indices.txt").read_text().replace(",", " ").split()]
    expected_keys = {(h, i) for h in c["harnesses"] for i in indices}
    if set(selected) - expected_keys:
        raise ValueError("Evaluation contains cells outside the frozen cohort")
    expected = len(expected_keys)
    result = {"metric": "pass@1", "complete": len(reports) == expected, "expected_cells": expected,
              "graded_cells": len(reports), "comparison_ready": len(reports) == expected,
              "tito_pass": bool(reports), "average_pass_at_1": sum(r['correct'] for r in reports) / len(reports) if reports else None,
              "harnesses": {}}
    for h in c["harnesses"]:
        rows = [r for r in reports if r['harness'] == h]
        detail = {"evaluations": len(rows), "pass_at_1": sum(r['correct'] for r in rows) / len(rows) if rows else None, "difficulty": {}}
        for d in ['easy', 'medium', 'hard']:
            tier = [r for r in rows if r['difficulty'] == d]
            detail['difficulty'][d] = {"evaluations": len(tier), "pass_at_1": sum(r['correct'] for r in tier) / len(tier) if tier else None}
        result['harnesses'][h] = detail
    write_json(output / "final_tito.json", {"reports": reports, "tito_pass": result['tito_pass']})
    write_json(output / "eval_progress.json", result)
    return result


def coordinator(c, output):
    from huggingface_hub import HfApi, Volume
    from huggingface_hub.errors import EntryNotFoundError
    from checkpoint_store import READY
    api = HfApi()
    namespace, training_id = c["hf"]["namespace"], os.environ["TRAINING_JOB"]
    training = api.inspect_job(job_id=training_id, namespace=namespace)
    bucket = c["hf"]["artifact_bucket"]
    prefix = c["hf"]["run_id"] + "/jobs/" + training.environment["RUN_OWNER"] + "/run"
    terminal = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"}
    destination = 'hf://buckets/' + bucket + '/' + c['hf']['run_id'] + '/coordinator'
    try:
        api.sync_bucket(destination, str(output), quiet=True)
    except EntryNotFoundError:
        pass
    terminal_seen = None
    while True:
        training = api.inspect_job(job_id=training_id, namespace=namespace)
        terminal_seen = (terminal_seen or time.monotonic()) if training.status.stage in terminal else None
        peers = list(api.list_jobs(namespace=namespace, labels={'experiment':'harbor-hard500',
                     'role':'coordinator','training_job':training_id}))
        peers = sorted([p for p in peers if p.status.stage not in terminal], key=lambda p:p.id)
        if peers and peers[0].environment['RUN_OWNER'] != os.environ['RUN_OWNER']:
            raise RuntimeError('Another coordinator owns this run')
        jobs = list(api.list_jobs(namespace=namespace, labels={"experiment": "harbor-hard500", "training_job": training_id, "role": "eval"}))
        active = [j for j in jobs if j.status.stage not in terminal]
        try:
            folders = [x.path for x in api.list_bucket_tree(bucket, prefix=prefix, recursive=False)
                       if Path(x.path).name.startswith("checkpoint-")]
        except EntryNotFoundError:
            folders = []
        pending = []
        for folder in folders:
            source = "hf://buckets/" + bucket + "/" + folder
            path = output / "manifests" / Path(folder).name / READY
            try:
                manifest = download_json(source, READY, path, api)
            except EntryNotFoundError:
                continue
            step = manifest['step']
            if step <= 500 or not (step % 100 == 0 or manifest.get('final')):
                continue
            if manifest['bundle_sha256'] != os.environ['BUNDLE_SHA256']:
                raise ValueError("Checkpoint runtime provenance mismatch")
            key = f"{step}-{digest(path)[:16]}"
            matches = [j for j in jobs if j.labels.get('checkpoint_key') == key]
            if len(matches) > 1:
                raise RuntimeError("Duplicate checkpoint evaluations")
            if not matches:
                pending.append((step, key, source, digest(path)))
        if pending and not active and training.status.stage not in {"CANCELED", "CANCELLED", "DELETED"}:
            step, key, source, sha = sorted(pending)[0]
            intent = output / f"intent-{key}.json"
            if intent.exists():
                raise RuntimeError("Ambiguous evaluation submission; reconcile intent")
            write_json(intent, {"state": "submitting", "step": step})
            api.sync_bucket(str(output), "hf://buckets/" + bucket + "/" + c['hf']['run_id'] + "/coordinator", quiet=True)
            env = {**training.environment, "RUN_OWNER": c['hf']['run_id'] + '-eval-' + key,
                   "CHECKPOINT_PREFIX": source, "CHECKPOINT_MANIFEST_SHA": sha, "TRAINING_JOB": training_id,
                   "CHECKPOINT_STEP": str(step)}
            job = api.run_job(namespace=namespace, image=c['hf']['image'], flavor='a100-large', timeout='4h',
                command=['python', '/bundle/bootstrap.py', '--role', 'eval', '--phase', 'checkpoint'], env=env,
                secrets={k: os.environ[k] for k in ['HF_TOKEN', 'E2B_API_KEY']},
                labels={'experiment':'harbor-hard500','role':'eval','training_job':training_id,'checkpoint_key':key},
                volumes=[Volume(type='dataset',source=os.environ['BUNDLE_REPO'],revision=os.environ['BUNDLE_REVISION'],mount_path='/bundle',read_only=True)])
            write_json(intent, {'state':'submitted','job_id':job.id,'step':step})
            active.append(job)
        failed = [j.id for j in jobs if j.status.stage in terminal and j.status.stage != 'COMPLETED']
        write_json(output / 'monitor.json', {'training_stage':training.status.stage,'active_eval_jobs':[j.id for j in active],
                   'pending_steps':[x[0] for x in pending], 'failed_evals':failed,'checked_at':time.time()})
        api.sync_bucket(str(output), 'hf://buckets/' + bucket + '/' + c['hf']['run_id'] + '/coordinator', quiet=True)
        if training.status.stage in {"CANCELED", "CANCELLED", "DELETED"} and not active:
            raise RuntimeError("Training canceled; remaining evaluations were not submitted")
        if terminal_seen is not None and time.monotonic() - terminal_seen >= 120 and not pending and not active:
            if failed:
                raise RuntimeError('Some checkpoint evaluations failed; inspect monitor.json')
            return
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['preflight','train','eval','coordinator'], required=True)
    parser.add_argument('--phase', default='smoke')
    args = parser.parse_args()
    os.environ['HF_PHASE'] = args.phase
    c = read(ROOT / 'config.json')
    output = ROOT / 'outputs' / os.environ['RUN_OWNER']
    output.mkdir(parents=True, exist_ok=True)
    verified_files = verify_bundle()
    if args.role == 'coordinator':
        coordinator(c, output)
        return
    from artifacts import Publisher
    publisher = Publisher(output)
    publisher.start()
    processes = []
    qualification = None
    try:
        cli_probe = start(eval_command(c, output, True) + ['--dry-run'],
                          output / 'eval-cli-preflight.log')
        processes.append(cli_probe)
        code = cli_probe.wait(timeout=120)
        processes.pop()
        write_json(output / 'eval_cli_preflight.json', {'passed': code == 0, 'returncode': code})
        if code:
            raise RuntimeError('Evaluation CLI preflight failed; see eval-cli-preflight.log')
        if args.role == 'train':
            kernel_env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '1', 'FLA_TILELANG': '1',
                          'PYTHONFAULTHANDLER': '1'}
            probe = start([ROOT / '.venv312/bin/python', ROOT / 'hf/runtime/hard_kernel_diagnostic.py',
                           '--worker'], output / 'kernel-check.log', kernel_env)
            processes.append(probe)
            code = probe.wait(timeout=600)
            processes.pop()
            write_json(output / 'kernel_preflight.json', {'passed': code == 0, 'returncode': code,
                       'backend': 'FLA TileLang', 'cuda_toolkit': '13.0.2'})
            if code:
                raise RuntimeError(f'GPU kernel preflight failed: exit {code}')
        if args.role in {'train','preflight'}:
            model = restore_parent(c, output)
        else:
            from checkpoint_store import restore_model
            model = output / 'inference-model'
            evidence = restore_model(os.environ['CHECKPOINT_PREFIX'], model, arm='blackbox',
                bundle_sha256=os.environ['BUNDLE_SHA256'], manifest_sha256=os.environ['CHECKPOINT_MANIFEST_SHA'])
            write_json(output / 'checkpoint_evaluation.json', evidence)
        if args.role == 'preflight':
            write_json(output / 'preflight.json', {'passed':True,'verified_files':verified_files,'full_parent_verified':True})
            return
        processes = services(c, output, model, args.role == 'train')
        if args.role == 'train':
            processes.append(start([ROOT / '.venv312/bin/python', ROOT / 'hf/runtime/hard_curriculum_logging.py',
                '--output', output, '--config', ROOT / 'config.json', '--watch'], output / 'trackio.log'))
        if args.role == 'train':
            final = train(c, output, model, publisher, args.phase == 'smoke')
            if args.phase == 'smoke':
                from training_capture_audit import audit_async
                audit_async(output / 'audit', 'blackbox')
                metrics = [json.loads(line) for line in (output/'audit/metrics.jsonl').read_text().splitlines()]
                if not any(row.get('grad_norm', 0) > 0 for row in metrics):
                    raise RuntimeError('Smoke has no fresh learning signal')
                if digest(model/'model.safetensors') == digest(final/'model.safetensors'):
                    raise RuntimeError('Smoke weights did not change')
                os.environ['CHECKPOINT_STEP'] = final.name.split('-')[-1]
                score = evaluate(c, output, True)
                qualification = {'passed':True,'bundle_sha256':os.environ['BUNDLE_SHA256'],
                    'finite_schedule_exhausted':True,'remote_optimizer_resume':True,'tito_pass':score['tito_pass'],
                    'training_capture_optimizer_audit':True,'weights_updated':True,'gpu_kernel_preflight':True,
                    'eval_smoke_passed':score['complete'],'final_checkpoint':str(final)}
        else:
            evaluate(c, output)
        if args.role == 'train':
            stop([processes.pop()])
        logging_started_at = time.time()
        logging = start([ROOT / '.venv312/bin/python', ROOT / 'hf/runtime/hard_curriculum_logging.py',
            '--output', output, '--config', ROOT / 'config.json'], output / 'trackio-final.log')
        try:
            logging.wait(timeout=120)
        except subprocess.TimeoutExpired:
            stop([logging])
        if qualification is not None:
            proof_path = output / 'trackio_verified.json'
            proof = read(proof_path) if proof_path.exists() else {}
            if (not proof.get('remote_readback_verified') or proof.get('events', 0) == 0
                    or proof.get('updated_at', 0) < logging_started_at):
                raise RuntimeError('Trackio remote readback is not verified')
            qualification['trackio_remote_verified'] = True
            write_json(output / 'qualification.json', qualification)
        write_json(output / 'status.json', {'passed':True,'finished_at':time.time()})
    except BaseException as exc:
        write_json(output / 'status.json', {'passed': False, 'exception_type': type(exc).__name__,
                                           'phase': args.phase, 'finished_at': time.time()})
        raise
    finally:
        failure_in_flight = sys.exc_info()[0] is not None
        try:
            stop(processes)
            if (output / 'trials').exists():
                with (output / 'cleanup.log').open('a') as log:
                    result = subprocess.run([str(ROOT / 'OpenEnv/.venv/bin/python'), '-c',
                        'from pathlib import Path; import sys; from baseline_checks import cleanup; cleanup(Path(sys.argv[1]))',
                        str(output)], stdout=log, stderr=subprocess.STDOUT, timeout=120)
                write_json(output / 'cleanup-exit.json', {'returncode': result.returncode})
                if result.returncode and not failure_in_flight:
                    raise RuntimeError('Owned sandbox cleanup failed; see cleanup.log')
        finally:
            publisher.finish()


if __name__ == '__main__':
    main()
