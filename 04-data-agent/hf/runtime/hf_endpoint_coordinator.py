"""Attempt-aware evaluator queue using the qualified HF endpoint transport."""
import os
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, "/workspace/repro/hf/runtime")
import hard_curriculum_job as runtime
from common import write_json
from checkpoint_store import digest, download_json

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
        planned_stop = False
        try:
            stop_receipt = download_json('hf://buckets/' + bucket + '/' + c['hf']['run_id'] + '/control-stop-1000',
                'status.json', output / 'stop-status.json', api)
            planned_stop = (stop_receipt.get('training_job') == training_id and
                stop_receipt.get('step') == 1000 and stop_receipt.get('all_files_hash_verified') is True)
        except EntryNotFoundError:
            pass
        pending = []
        for folder in folders:
            source = "hf://buckets/" + bucket + "/" + folder
            path = output / "manifests" / Path(folder).name / READY
            try:
                manifest = download_json(source, READY, path, api)
            except EntryNotFoundError:
                continue
            step = manifest['step']
            if step > 1000 or step <= 500 or not (step % 100 == 0 or manifest.get('final')):
                continue
            if manifest['bundle_sha256'] != os.environ['BUNDLE_SHA256']:
                raise ValueError("Checkpoint runtime provenance mismatch")
            key = f"{step}-{digest(path)[:16]}"
            matches = [j for j in jobs if j.labels.get('checkpoint_key') == key]
            matches.sort(key=lambda job: job.id)
            if sum(job.status.stage not in terminal for job in matches) > 1:
                raise RuntimeError("Concurrent duplicate checkpoint evaluations")
            accepted = False
            for previous in reversed(matches):
                if previous.status.stage not in terminal:
                    continue
                evidence = output / "scores" / (previous.id + ".json")
                eval_source = "hf://buckets/" + bucket + "/" + c['hf']['run_id'] + "/jobs/" + previous.environment['RUN_OWNER']
                try:
                    result = download_json(eval_source, "canonical_scores.json", evidence, api)
                    accepted = result.get('complete') and result.get('tito_pass') and result.get('graded_cells') == 1000
                except EntryNotFoundError:
                    pass
                if accepted:
                    break
            if accepted:
                continue
            if not matches or (matches[-1].status.stage in terminal and sum(j.labels.get("transport_generation") == "c100-v3" for j in matches) < 3):
                resume = json.dumps(["hf://buckets/" + bucket + "/" + c['hf']['run_id'] + "/jobs/" + j.environment['RUN_OWNER'] for j in matches])
                pending.append((step, key, source, digest(path), resume, len(matches) + 1))
        if pending and len(active) < 2 and (planned_stop or training.status.stage not in {"CANCELED", "CANCELLED", "DELETED"}):
            step, key, source, sha, resume, attempt = sorted(pending, key=lambda item: (item[0] != 1000, -item[0]))[0]
            intent = output / f"intent-{key}-endpoint-{attempt}.json"
            if intent.exists():
                raise RuntimeError("Ambiguous evaluation submission; reconcile intent")
            write_json(intent, {"state": "submitting", "step": step})
            api.sync_bucket(str(output), "hf://buckets/" + bucket + "/" + c['hf']['run_id'] + "/coordinator", quiet=True)
            env = {**training.environment, "RUN_OWNER": c['hf']['run_id'] + '-eval-' + key + '-endpoint-' + str(attempt),
                   "CHECKPOINT_PREFIX": source, "CHECKPOINT_MANIFEST_SHA": sha, "TRAINING_JOB": training_id,
                   "CHECKPOINT_STEP": str(step), "EVAL_RESUME_PREFIXES": resume}
            job = api.run_job(namespace=namespace, image=c['hf']['image'], flavor='a100x4', timeout='12h', expose=[8201],
                command=['python', '/qualification/' + os.environ['ENDPOINT_SUBDIR'] + '/bootstrap.py', '--role', 'eval', '--phase', 'checkpoint'], env=env,
                secrets={k: os.environ[k] for k in ['HF_TOKEN', 'E2B_API_KEY']},
                labels={'experiment':'harbor-hard500','role':'eval','training_job':training_id,'checkpoint_key':key,'transport':'hf-endpoint','attempt':str(attempt),'transport_generation':'c100-v3'},
                volumes=[Volume(type='dataset',source=os.environ['BUNDLE_REPO'],revision=os.environ['BUNDLE_REVISION'],mount_path='/bundle',read_only=True), Volume(type='dataset',source=os.environ['BUNDLE_REPO'],revision=os.environ['ENDPOINT_REVISION'],mount_path='/qualification',read_only=True)])
            write_json(intent, {'state':'submitted','job_id':job.id,'step':step})
            active.append(job)
        failed = [j.id for j in jobs if j.status.stage in terminal and j.status.stage != 'COMPLETED']
        write_json(output / 'monitor.json', {'training_stage':training.status.stage,'active_eval_jobs':[j.id for j in active],
                   'pending_steps':[x[0] for x in pending], 'failed_evals':failed,'transport':'hf-endpoint','eval_concurrency':100,'max_parallel_evals':2,'planned_training_stop':planned_stop,'checked_at':time.time()})
        api.sync_bucket(str(output), 'hf://buckets/' + bucket + '/' + c['hf']['run_id'] + '/coordinator', quiet=True)
        if training.status.stage in {"CANCELED", "CANCELLED", "DELETED"} and not planned_stop and not active:
            raise RuntimeError("Training canceled; remaining evaluations were not submitted")
        if terminal_seen is not None and time.monotonic() - terminal_seen >= 120 and not pending and not active:
            unresolved = [j for j in jobs if j.id in failed and not any(x.labels.get('checkpoint_key') == j.labels.get('checkpoint_key') and x.status.stage == 'COMPLETED' for x in jobs)]
            if unresolved:
                raise RuntimeError('Some checkpoint evaluations failed; inspect monitor.json')
            return
        time.sleep(60)


if __name__ == "__main__":
    runtime.verify_bundle()
    output = runtime.ROOT / "outputs" / os.environ["RUN_OWNER"]
    output.mkdir(parents=True, exist_ok=True)
    coordinator(runtime.read(runtime.ROOT / "config.json"), output)
