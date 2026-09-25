"""Launch an authorized comparison after its live qualification dependency passes.

Submission intent is durable before allocation. Repeated invocations adopt the
same HF owner; an ambiguous submission is never blindly repeated.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

HF = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def run(command):
    subprocess.run([str(x) for x in command], check=True)


class RecordedAPI:
    def __init__(self, api, intent):
        self.api, self.intent = api, intent

    def __getattr__(self, name):
        return getattr(self.api, name)

    def run_job(self, **kwargs):
        if self.intent.exists():
            raise RuntimeError('Submission intent already exists: reconcile its owner before allocating')
        write(self.intent, {'submitted_at': time.time(), 'state': 'submitting',
            'owner': kwargs['env']['RUN_OWNER'], 'namespace': kwargs['namespace'],
            'labels': kwargs['labels'], 'bundle_sha256': kwargs['env']['BUNDLE_SHA256']})
        job = self.api.run_job(**kwargs)
        write(self.intent, {**read(self.intent), 'state': 'submitted', 'training_job': job.id})
        return job


def whitebox(args, state):
    from huggingface_hub import HfApi
    from deploy import credentials, submit
    plan = read(args.ready / 'plan.json')
    if plan.get('status') != 'qualification_passed_main_not_submitted':
        raise ValueError('HF live optimizer/checkpoint/logging qualification is not complete')
    secrets = credentials(args.env_file)
    api = HfApi(token=secrets['HF_TOKEN'])
    intent = args.out / 'hf-submit-intent.json'
    if intent.exists():
        prior = read(intent)
        matches = [j for j in api.list_jobs(namespace=prior['namespace'], labels=prior['labels'])
                   if j.environment.get('RUN_OWNER') == prior['owner']]
        if len(matches) != 1 or matches[0].environment.get('BUNDLE_SHA256') != prior['bundle_sha256']:
            raise RuntimeError('Ambiguous HF submission; refusing to allocate a duplicate')
        training_job = matches[0].id
    else:
        job_args = argparse.Namespace(role='train', arm='whitebox', phase='long', flavor='h200x2',
            timeout='24h', dp=1, limit=0, resume_eval_owner=None, training_job=None,
            baseline_job=plan['baseline_job'], smoke_job=plan['smoke_job'],
            checkpoint_eval_job=plan['checkpoint_eval_job'], external_checkpoint_coordinator=True,
            dry_run=False)
        value = submit(RecordedAPI(api, intent), read(plan['config']), secrets, args.ready, job_args)
        training_job = value['id']
    state.update(training_job=training_job, provider='hf', training_submitted=True)
    write(args.out / 'state.json', state)
    controller = args.out / 'controller'
    if not (controller / 'submission.json').exists():
        if controller.exists():
            raise RuntimeError('Partial controller submission: reconcile Slurm before retrying')
        run([sys.executable, HF / 'hf_followup.py', 'prepare', '--training-job', training_job,
             '--bundle', args.bundle, '--env-file', args.env_file, '--coordination-dir', args.coordination_dir,
             '--out', controller, '--submit'])
    receipt = read(controller / 'submission.json')
    state.update(controller_job=receipt['controller_job'], controller_plan=str(controller / 'plan.json'))


def opencode(args, state):
    plan = read(args.ready / 'plan.json')
    if plan.get('checkpoint_eval_gpu_validation') != 'passed':
        raise ValueError('Native live checkpoint qualification is not complete')
    if not plan.get('native_baseline_score'):
        raise ValueError('Prepare a plan with both native diagnostic and four-harness comparison baselines')
    target = args.out / 'run'
    if not target.exists():
        run([sys.executable, HF / 'local_long.py', '--arm', 'opencode',
             '--smoke-run', plan['smoke_run'], '--baseline-score', plan['native_baseline_score'],
             '--comparison-baseline-score', plan['baseline_score'],
             '--checkpoint-eval-proof', plan['checkpoint_eval_proof'], '--env-file', args.env_file,
             '--coordination-dir', args.coordination_dir, '--out', target, '--submit'])
    live = read(target / 'plan.json')
    if not live.get('training_job') or not live.get('controller_job'):
        raise RuntimeError('Partial local submission: reconcile Slurm before retrying')
    state.update(training_job=live['training_job'], controller_job=live['controller_job'], provider='slurm',
        training_submitted=True, controller_plan=str(target / 'plan.json'),
        training_output=str(Path(live['root']) / 'outputs' / ('local-train-opencode-' + live['training_job'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['opencode', 'whitebox'], required=True)
    for name in ('ready', 'out', 'env-file', 'coordination-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(args.out / 'state.json') if (args.out / 'state.json').exists() else {
            'arm': args.arm, 'authorized': True, 'started_at': time.time(), 'training_submitted': False}
        if state.get('complete'):
            print(json.dumps(state))
            return
        write(args.out / 'state.json', state)
        try:
            (whitebox if args.arm == 'whitebox' else opencode)(args, state)
            state.update(complete=True, completed_at=time.time())
        except Exception as exc:
            state.update(error_type=type(exc).__name__, failed_at=time.time())
            write(args.out / 'state.json', state)
            raise
        write(args.out / 'state.json', state)
        print(json.dumps(state))


if __name__ == '__main__':
    main()
