"""Prepare or queue fixed 250-task, four-harness pass@1 checkpoint evaluations.

Default is one planning pass without job submission. --submit enables Slurm submission;
--watch continues polling. Each evaluation receives two separate inference GPUs.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'train'))
from checkpoint_artifacts import READY, SAVED, digest, finalize_saved, stage_model, write_json

REPO = Path('/fsx/adithyaskolavi/projects/trl_prod')
TOOLS = REPO / 'experiments/async_grpo_harbor_data_agent/tools'
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def freeze_eval_source(root, protocol):
    """One immutable evaluator for all checkpoints handled by this watcher."""
    if not root.exists():
        temporary = root.with_name(root.name + '.preparing')
        temporary.mkdir()
        for relative in ['OpenEnv/src/openenv', 'OpenEnv/envs/harbor_env', 'trl/trl']:
            shutil.copytree(REPO / relative, temporary / relative, symlinks=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ['launch_multi4_baseline.sh', 'finish_multi4_baseline.sh', 'baseline_checks.py',
                     'eval_concurrent.py', 'eval_pass_at_k.py', 'harnesses_supported.txt',
                     'smoke_multiharness_tito.py']:
            shutil.copy2(TOOLS / name, temporary / name)
        shutil.copy2(REPO / 'scripts/inference/vllm/serve_vllm_tunnel.sh', temporary / 'serve_vllm_tunnel.sh')
        shutil.copy2(PROJECT / 'train/checkpoint_artifacts.py', temporary / 'checkpoint_artifacts.py')
        write_json(temporary / 'protocol.json', protocol)
        hashes = {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob('*') if p.is_file()}
        write_json(temporary / 'source_hashes.json', hashes)
        temporary.rename(root)
    if json.loads((root / 'protocol.json').read_text()) != protocol:
        raise ValueError('Protocol changed within the checkpoint-evaluation series')
    for name, expected in json.loads((root / 'source_hashes.json').read_text()).items():
        if digest(root / name) != expected:
            raise ValueError(f'Frozen evaluator changed: {name}')
    return root


def eligible(checkpoint, interval, *, include_final=False):
    marker_file = checkpoint / READY if (checkpoint / READY).exists() else checkpoint / SAVED
    if not marker_file.exists():
        return None
    marker = json.loads(marker_file.read_text())
    step = int(checkpoint.name.removeprefix('checkpoint-'))
    if marker['step'] != step or marker['checkpoint'] != str(checkpoint.resolve()):
        raise ValueError(f'Invalid completion marker: {checkpoint}')
    return marker if step > 0 and (step % interval == 0 or include_final and marker.get('final', False)) else None


def prepare(checkpoint, output, protocol):
    baseline = Path(protocol['baseline_run'])
    if digest(baseline / 'manifest.json') != protocol['manifest_sha256']:
        raise ValueError('Baseline manifest changed')
    if digest(baseline / 'indices.txt') != protocol['indices_sha256']:
        raise ValueError('Baseline dispatch order changed')
    if not (checkpoint / READY).exists():
        finalize_saved(checkpoint)
    marker = json.loads((checkpoint / READY).read_text())
    if marker['base_model'] != protocol['model'] or marker['base_revision'] != protocol['model_revision']:
        raise ValueError('Checkpoint base model/revision differs from the baseline')
    output.mkdir(parents=True, exist_ok=False)
    from huggingface_hub import snapshot_download
    metadata = Path(snapshot_download(protocol['model'], revision=protocol['model_revision'],
        cache_dir='/fsx/adithyaskolavi/.cache/huggingface/hub',
        allow_patterns=['*tokenizer*', '*processor*', 'chat_template*', 'special_tokens_map.json',
                        'vocab.json', 'merges.txt', 'added_tokens.json']))
    staged = stage_model(checkpoint, output / 'model', metadata)
    manifest = json.loads((baseline / 'manifest.json').read_text())
    shutil.copytree(baseline / 'dataset', output / 'dataset')
    shutil.copy2(baseline / 'indices.txt', output / 'indices.txt')
    manifest['split'] = str((output / 'dataset').resolve())
    manifest['checkpoint'] = staged['checkpoint']
    write_json(output / 'manifest.json', manifest)
    snapshot = freeze_eval_source(output.parent / 'eval-source', protocol)
    (output / 'source-snapshot').symlink_to(snapshot.resolve(), target_is_directory=True)
    write_json(output / 'eval_plan.json', {'step': marker['step'], 'checkpoint': str(checkpoint),
        'model_source': str(output / 'model'), 'model_alias': protocol['model'],
        'protocol': protocol, 'gpu_count': 2, 'tp': 1, 'dp': 2, 'concurrency': 100,
        'evaluations': 1000, 'metric': 'pass@1', 'interval_unit': 'optimizer steps'})
    return output


def submission_env(output, protocol):
    env = {**os.environ, 'BASELINE_RUN': str(output),
           'BASELINE_SNAPSHOT': str(output / 'source-snapshot'),
           'EVAL_CODE_ROOT': str(output / 'source-snapshot'),
           'EVAL_MODEL_SOURCE': str(output / 'model'), 'EVAL_MODEL_ALIAS': protocol['model'],
           'OPENENV_HARBOR_AGENT_VERSIONS': json.dumps(protocol['harness_versions'])}
    # A training snapshot must never accidentally become an evaluation source on restart.
    env.pop('BASELINE_RESUME_FROM', None)
    env.pop('EVAL_SMOKE_ONLY', None)
    return env


def submit(output, protocol, partition, *, exclude_nodes=None):
    env = submission_env(output, protocol)
    key = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    command = ['sbatch', '--parsable', f'--partition={partition}', '--ntasks=1', '--gres=gpu:2',
               '--cpus-per-task=8', '--mem=128G', '--time=03:00:00', '--export=ALL',
               f'--job-name=multi4-ev-{key}', '--output=/fsx/%u/logs/%x-%j.out',
               '--error=/fsx/%u/logs/%x-%j.err',
               str(output / 'source-snapshot/launch_multi4_baseline.sh')]
    if exclude_nodes:
        command.insert(-1, f'--exclude={exclude_nodes}')
    job = subprocess.check_output(command, env=env, text=True).strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'Unrecognized Slurm submission result: {job!r}')
    # Persist the GPU ID before the dependent cleanup submission, so failure is resumable.
    write_json(output / 'submission.json', {'job_id': job, 'cleanup_job_id': None})
    return submit_cleanup(output, protocol, job)


def submit_cleanup(output, protocol, job):
    env = submission_env(output, protocol)
    cleanup = subprocess.check_output(['sbatch', '--parsable', '--partition=hopper-cpu',
        '--ntasks=1', '--cpus-per-task=2', '--mem=8G', '--time=00:20:00', '--export=ALL',
        f'--dependency=afterany:{job}', f'--job-name=multi4-ev-clean-{job}',
        '--output=/fsx/%u/logs/%x-%j.out', '--error=/fsx/%u/logs/%x-%j.err',
        str(output / 'source-snapshot/finish_multi4_baseline.sh'), job], env=env, text=True).strip().split(';')[0]
    if not cleanup.isdigit():
        raise RuntimeError('Cleanup submission failed; GPU ID is recorded in submission.json')
    record = {'job_id': job, 'cleanup_job_id': cleanup}
    write_json(output / 'submission.json', record)
    return record


def slurm_state(job):
    result = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job),
                                     '--format=JobID,State'], text=True)
    states = {line.split('|')[0]: line.split('|')[1].split()[0].rstrip('+')
              for line in result.splitlines() if '|' in line}
    return states.get(str(job), 'UNKNOWN')


def slurm_nodes(job):
    result = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job),
                                     '--format=JobID,NodeList'], text=True)
    for line in result.splitlines():
        fields = line.split('|')
        if len(fields) >= 2 and fields[0] == str(job) and fields[1] not in {'', 'None assigned', '(null)'}:
            return fields[1]
    raise RuntimeError('Cannot identify training node; leaving evaluation unsubmitted')


def trial_result_path(logs, trial_name):
    """Resolve measured agent metadata through this checkpoint's retry ancestry."""
    if not trial_name or Path(trial_name).name != trial_name:
        return None
    logs = Path(logs).resolve()
    checkpoint = logs.parent
    seen = set()
    while logs not in seen:
        seen.add(logs)
        native = logs / 'trials' / trial_name / 'result.json'
        if native.is_file():
            return native
        receipt = logs / 'resume_transport_migration.json'
        if not receipt.is_file():
            return None
        source = json.loads(receipt.read_text()).get('source')
        if not source:
            return None
        logs = Path(source).resolve()
        if logs.parent != checkpoint:
            return None
    return None


def summarize(output, job, protocol):
    logs = output / f'job-{job}'
    selected, ungraded_attempts = {}, 0
    for path in sorted((logs / 'traces').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            reward = row.get('reward')
            if reward is None:
                ungraded_attempts += 1
                continue
            if not isinstance(reward, (int, float)) or not math.isfinite(reward) or row.get('n_turns', 0) <= 0:
                continue
            selected.setdefault((row['harness'], row['index'], row['rep']), row)
    expected = {(h, i, 0) for h in protocol['harnesses'] for i in range(250)}
    if set(selected) != expected:
        return {'complete': False, 'graded_cells': len(selected), 'expected_cells': 1000}
    manifest = json.loads((output / 'manifest.json').read_text())
    scores = {}
    for h in protocol['harnesses']:
        rows = [selected[h, i, 0] for i in range(250)]
        score = sum(r['reward'] for r in rows) / 250
        categories = {}
        for difficulty in ['easy', 'medium', 'hard']:
            subset = [r for r in rows if manifest['tasks'][r['index']]['difficulty'] == difficulty]
            categories[difficulty] = {'graded': len(subset), 'correct': sum(r['reward'] for r in subset),
                                      'pass_at_1': sum(r['reward'] for r in subset) / len(subset)}
        scores[h] = {'graded': 250, 'correct': sum(r['reward'] for r in rows), 'pass_at_1': score,
                     'delta_from_base': score - protocol['scores'][h]['pass_at_1'], 'difficulty': categories}
    audit_path = logs / 'final_tito.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    versions = {h: {} for h in protocol['harnesses']}
    for (h, _, _), row in selected.items():
        native = trial_result_path(logs, row.get('trial_name'))
        data = json.loads(native.read_text()) if native is not None else {}
        version = (data.get('agent_info') or {}).get('version') or 'unverified'
        versions[h][version] = versions[h].get(version, 0) + 1
    pins_match = all(versions[h] == {protocol['harness_versions'][h]: 250} for h in protocol['harnesses'])
    result = {'complete': True, 'graded_cells': 1000, 'metric': 'pass@1', 'harnesses': scores,
              'average_pass_at_1': sum(v['pass_at_1'] for v in scores.values()) / 4,
              'tito_pass': audit.get('tito_pass', False), 'ungraded_attempts': ungraded_attempts,
              'harness_versions': versions, 'harness_versions_match_baseline': pins_match,
              'comparison_ready': audit.get('tito_pass', False) and pins_match,
              'checkpoint': str(output / 'model'), 'job_id': job}
    write_json(output / 'scores.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=PROJECT / 'eval/baseline_protocol.json')
    p.add_argument('--interval', type=int, default=100)
    p.add_argument('--include-final', action='store_true', help='Also evaluate a final checkpoint off the interval')
    p.add_argument('--max-active-evals', type=int, default=1)
    p.add_argument('--partition', default='hopper-extra')
    p.add_argument('--submit', action='store_true')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--train-job', type=int, help='Stop watching once training and all queued evals finish')
    p.add_argument('--poll-seconds', type=int, default=30)
    args = p.parse_args()
    args.checkpoints = args.checkpoints.resolve()
    args.output = args.output.resolve()
    args.protocol = args.protocol.resolve()
    if args.interval <= 0 or args.max_active_evals <= 0 or not 1 <= args.poll_seconds <= 60:
        p.error('Use a positive interval/cap and a poll interval between 1 and 60 seconds')
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.watcher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_file = args.output / 'state.json'
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        while True:
            active = 0
            for key, record in state.items():
                output = Path(record['output'])
                saved = output / 'submission.json'
                if saved.exists():
                    record.update(json.loads(saved.read_text()))
                if not record.get('job_id'):
                    raise RuntimeError(f'Submission interrupted for {key}; reconcile Slurm before retrying')
                if args.submit and not record.get('cleanup_job_id'):
                    record.update(submit_cleanup(output, protocol, record['job_id']))
                status = slurm_state(record['job_id'])
                record['slurm_state'] = status
                if status not in TERMINAL:
                    active += 1
                elif status == 'COMPLETED' and not record.get('scores', {}).get('complete'):
                    record['scores'] = summarize(output, record['job_id'], protocol)
            checkpoints = sorted((p for p in args.checkpoints.glob('checkpoint-*')
                                  if re.fullmatch(r'checkpoint-\d+', p.name)),
                                 key=lambda p: int(p.name.removeprefix('checkpoint-')))
            for checkpoint in checkpoints:
                marker = eligible(checkpoint, args.interval, include_final=args.include_final)
                if marker is None or str(checkpoint) in state:
                    continue
                print(json.dumps({'eligible_checkpoint': str(checkpoint), 'step': marker['step'],
                                  'mode': 'submit' if args.submit else 'plan'}), flush=True)
                if not args.submit or active >= args.max_active_evals:
                    continue
                excluded = slurm_nodes(args.train_job) if args.train_job else None
                output = args.output / f"step-{marker['step']:06d}"
                prepare(checkpoint, output, protocol)
                record = {'output': str(output), 'status': 'submitting'}
                state[str(checkpoint)] = record
                write_json(state_file, state)
                record.update(submit(output, protocol, args.partition,
                                     exclude_nodes=excluded))
                record['status'] = 'submitted'
                write_json(state_file, state)
                active += 1
            if args.submit:
                write_json(state_file, state)
            if not args.watch:
                break
            if args.train_job and slurm_state(args.train_job) in TERMINAL and active == 0:
                pending = [p for p in checkpoints if eligible(p, args.interval, include_final=args.include_final)
                           and str(p) not in state]
                if not pending:
                    print(json.dumps({'watch_complete': True, 'evaluations': len(state)}), flush=True)
                    break
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
