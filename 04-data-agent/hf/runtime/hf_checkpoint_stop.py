"""Stop the training job only after a full checkpoint is verified from the bucket."""
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, '/workspace/repro/hf/runtime')
import hard_curriculum_job as runtime
from checkpoint_store import READY, download_json, bucket_location, verify, digest
from common import write_json


def validate_target(manifest, step, bundle):
    if manifest.get('step') != step or manifest.get('bundle_sha256') != bundle:
        raise ValueError('Wrong checkpoint or runtime bundle')
    required = {'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rollout_state.json'}
    if not required.issubset(manifest.get('files', {})):
        raise ValueError('Checkpoint lacks full training state')
    if not any(name.endswith('.safetensors') for name in manifest['files']):
        raise ValueError('Checkpoint lacks model weights')
    if any(Path(name).name != name for name in manifest['files']):
        raise ValueError('Invalid checkpoint member')


def main():
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError
    runtime.verify_bundle()
    api = HfApi()
    namespace, job_id = os.environ['NAMESPACE'], os.environ['TRAINING_JOB']
    step = int(os.environ.get('STOP_CHECKPOINT', '1000'))
    job = api.inspect_job(namespace=namespace, job_id=job_id)
    source = 'hf://buckets/' + job.environment['ARTIFACT_BUCKET'] + '/' + job.environment['RUN_ID'] + '/jobs/' + job.environment['RUN_OWNER'] + '/run/checkpoint-' + str(step)
    bucket, prefix = bucket_location(source)
    output = runtime.ROOT / 'stop-control'
    output.mkdir(exist_ok=True)
    dest = 'hf://buckets/' + bucket + '/' + job.environment['RUN_ID'] + '/control-stop-' + str(step)
    target = runtime.ROOT / ('verified-stop-checkpoint-' + str(step))
    target.mkdir(exist_ok=True)
    deadline = time.monotonic() + 24 * 3600

    verified = {}

    def report(phase, **extra):
        write_json(output / 'status.json', {'phase': phase, 'training_job': job_id, 'step': step,
            'source': source, 'checked_at': time.time(), **verified, **extra})
        api.sync_bucket(str(output), dest, quiet=True)

    while time.monotonic() < deadline:
        try:
            current = api.inspect_job(namespace=namespace, job_id=job_id)
            try:
                manifest = download_json(source, READY, target / READY, api)
            except EntryNotFoundError:
                report('waiting_for_checkpoint', training_stage=current.status.stage)
                if current.status.stage in {'COMPLETED', 'ERROR', 'CANCELED', 'CANCELLED'}:
                    raise RuntimeError('Training stopped before target checkpoint was published')
                time.sleep(30)
                continue
            validate_target(manifest, step, job.environment['BUNDLE_SHA256'])
            report('verifying_checkpoint')
            api.download_bucket_files(bucket, files=[(prefix + '/' + name, str(target / name)) for name in manifest['files']], raise_on_missing_files=True)
            verify(target)
            write_json(output / 'verified_manifest.json', manifest)
            verified = {'checkpoint_manifest_sha256': digest(target / READY), 'all_files_hash_verified': True}
            report('checkpoint_verified')
            if current.status.stage not in {'COMPLETED', 'ERROR', 'CANCELED', 'CANCELLED'}:
                api.cancel_job(namespace=namespace, job_id=job_id)
            report('stop_requested', checkpoint_manifest_sha256=digest(target / READY), all_files_hash_verified=True)
            for _ in range(60):
                current = api.inspect_job(namespace=namespace, job_id=job_id)
                if current.status.stage in {'COMPLETED', 'ERROR', 'CANCELED', 'CANCELLED'}:
                    report('stopped', training_stage=current.status.stage, checkpoint_manifest_sha256=digest(target / READY), all_files_hash_verified=True)
                    return
                time.sleep(5)
            raise RuntimeError('Training cancellation did not reach a terminal state')
        except Exception as exc:
            report('retrying', error_type=type(exc).__name__)
            time.sleep(30)
    raise TimeoutError('Stop checkpoint was not verified within 24 hours')


if __name__ == '__main__':
    main()
