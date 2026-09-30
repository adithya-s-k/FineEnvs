import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))


@pytest.mark.parametrize('canceled', [False, True])
def test_checkpoint_eval_admission_and_restart(tmp_path, monkeypatch, canceled):
    import huggingface_hub
    import hard_curriculum_job as runtime

    for k, v in {'TRAINING_JOB': 'trainer', 'RUN_OWNER': 'coordinator', 'BUNDLE_SHA256': 'bundle',
                 'BUNDLE_REPO': 'org/repro', 'BUNDLE_REVISION': 'revision',
                 'HF_TOKEN': 'test-only', 'E2B_API_KEY': 'test-only'}.items():
        monkeypatch.setenv(k, v)
    training = NS(status=NS(stage='CANCELED' if canceled else 'COMPLETED'),
                  environment={'RUN_OWNER': 'trainer', 'BUNDLE_SHA256': 'bundle'})
    submitted = []
    jobs = []
    class API:
        def inspect_job(self, **kwargs): return training
        def sync_bucket(self, *args, **kwargs): pass
        def list_jobs(self, **kwargs):
            return jobs if kwargs['labels']['role'] == 'eval' else []
        def list_bucket_tree(self, *args, **kwargs):
            return [NS(path=f"{kwargs['prefix']}/checkpoint-{step}") for step in [500, 550, 600, 650]]
        def run_job(self, **kwargs):
            submitted.append(kwargs)
            job = NS(id=str(len(submitted)), status=NS(stage='COMPLETED'), labels=kwargs['labels'])
            jobs.append(job)
            return job
    def manifest(source, name, path, api):
        step = int(source.rsplit('-', 1)[1])
        value = {'step': step, 'final': step == 650, 'bundle_sha256': 'bundle'}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return value
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    monkeypatch.setattr(runtime, 'download_json', manifest)
    clock = itertools.count(1000, 200)
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(runtime.time, 'sleep', lambda _: None)
    config = {'hf': {'namespace': 'org', 'artifact_bucket': 'org/bucket', 'run_id': 'run', 'image': 'image'}}
    if canceled:
        with pytest.raises(RuntimeError, match='Training canceled'):
            runtime.coordinator(config, tmp_path)
        assert not submitted
    else:
        runtime.coordinator(config, tmp_path)
        runtime.coordinator(config, tmp_path)
        assert [j['env']['CHECKPOINT_STEP'] for j in submitted] == ['600', '650']
        assert all(j['flavor'] == 'a100-large' for j in submitted)
        assert all(j['env']['RUN_OWNER'] != 'trainer' for j in submitted)
