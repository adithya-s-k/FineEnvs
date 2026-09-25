"""Persistent CPU observer for the three data-agent training runs.

Samples artifacts every 600 seconds. No inference requests or model transfers.
Known transient controller failures can restart twice using their frozen script;
model, token, provenance and numerical failures require investigation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time

TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'NODE_FAIL', 'BOOT_FAIL', 'OUT_OF_MEMORY',
            'CANCELLED', 'CANCELED', 'PREEMPTED', 'ERROR', 'DELETED'}
WORKSPACE = Path(os.environ.get('TRAINING_WORKSPACE', Path(__file__).resolve().parents[3]))
LOGS = WORKSPACE / 'experiments/async_grpo_harbor_data_agent/logs'
COMPARISON = WORKSPACE / 'experiments/daytona_harness_comparison/logs/hf-20260915'


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else ({} if default is None else default)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def records(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines(keepends=True)
            if line.endswith('\n') and line.strip()]


def states(jobs):
    jobs = {str(j) for j in jobs if str(j).isdigit()}
    if not jobs:
        return {}
    text = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(sorted(jobs)),
                                   '--format=JobIDRaw,State'], text=True, timeout=20)
    return {p[0]: p[1].split()[0].rstrip('+') for line in text.splitlines()
            if len(p := line.split('|')) > 1 and p[0] in jobs}


def training_metrics(rows):
    # A completed run also emits a summary with the same step. Only optimizer
    # receipts belong in the reward curve; preserve its first receipt per step.
    updates = {}
    for row in rows:
        if 'grad_norm' in row:
            updates.setdefault(int(row['step']), row)
    rows = [updates[k] for k in sorted(updates)]
    recent, previous = rows[-20:], rows[-40:-20]
    rewards = lambda values: mean(float(r['reward']) for r in values) if values else None
    speeds = [float(r.get('perf/step_s', r.get('step_time', 0))) for r in recent]
    return {'step': rows[-1]['step'] if rows else 0, 'updates': len(rows),
        'reward_last20': rewards(recent), 'reward_previous20': rewards(previous),
        'reward_window_size': len(recent), 'previous_window_size': len(previous),
        'nonzero_gradients_last20': sum(float(r['grad_norm']) > 0 for r in recent),
        'nonfinite_updates': sum(any(not math.isfinite(float(r[k])) for k in
            ['loss', 'grad_norm', 'reward', 'kl', 'ratio', 'entropy'] if k in r) for r in rows),
        'max_staleness': max((float(r.get('sample/staleness_max', 0)) for r in rows), default=0),
        'mean_step_seconds_last20': mean(speeds) if speeds and all(speeds) else None}


def summarize_artifacts(output, arm):
    output = Path(output)
    metric_file = output / ('run/metrics.jsonl' if arm == 'whitebox' else 'audit/metrics.jsonl')
    summary = training_metrics(records(metric_file))
    summary['metric_fingerprint'] = str(summary['step'])
    status = read(output / 'status.json')
    summary.update(runtime_status=status, trackio=read(output / 'trackio_verified.json'),
        uploads=read(output / 'upload_status.json'),
        checkpoints=sorted(int(p.name.split('-')[-1]) for p in (output / 'run').glob('checkpoint-*')
            if p.is_dir() and p.name.split('-')[-1].isdigit() and
            any((p / f).exists() for f in ('checkpoint.saved.json', 'checkpoint.ready.json', 'checkpoint.hf.ready.json'))))
    if arm == 'whitebox':
        audits = [r for event in records(output / 'run/token_audit.jsonl') for r in event.get('rows', [])]
        summary['tito'] = {'rows': len(audits), 'passed': sum(bool(r.get('tito_pass')) for r in audits),
                           'supervised_tokens': sum(r.get('supervised', 0) for r in audits)}
    else:
        summary['tito'] = read(output / 'audit/tito_summary.json')
    return summary


def original():
    root = LOGS / 'multi4-long-prod-20260915'
    continuation = LOGS / 'multi4-long-prod-cont-20260915'
    if (continuation / 'submission.json').exists():
        root = continuation
    submission = read(root / 'submission.json')
    job = str(submission['training'])
    summary = summarize_artifacts(root / ('job-' + job), 'multi4')
    # A fresh allocation has no new metrics yet. Preserve the actual inherited
    # optimizer history instead of displaying step zero during restore/startup.
    history, inherited_saves = original_history(root)
    summary.update(training_metrics(history))
    resume = read(root / 'run_config.json').get('resume_state') or {}
    summary['step'] = max(summary['step'], resume.get('step', 0))
    summary['resumed_from_step'] = resume.get('step', 0)
    summary['resume_schedule_cursor'] = resume.get('group_offset')
    summary['checkpoints'] = inherited_saves
    monitor = read(root / 'monitor/status.json')
    summary.update(arm='multi4', provider='slurm', job=job, source=str(root),
        stage=states([job]).get(job, 'UNKNOWN'), controller_alerts=monitor.get('alerts', []),
        support_jobs={k: submission[k] for k in ('eval_watcher', 'monitor', 'logging') if k in submission},
        trackio=read(root / 'trackio/status.json'),
        allocation_continuation=read((LOGS / 'multi4-long-prod-20260915') / 'operations/allocation-continuation/status.json'))
    summary['support_states'] = states(summary['support_jobs'].values())
    summary['evaluations'] = original_evaluations(root)
    return summary


def original_evaluations(root):
    """Keep parent checkpoint evaluations visible after an allocation handoff."""
    selected, seen, upper = {}, set(), float('inf')
    while root not in seen:
        seen.add(root)
        for directory in (root / 'checkpoint-evals').glob('step-*'):
            step = int(directory.name.split('-')[-1])
            if step > upper:
                continue
            if step in selected and selected[step] != directory:
                raise ValueError('Conflicting checkpoint evaluation ancestry')
            selected[step] = directory
        resume = read(root / 'run_config.json').get('resume_state') or {}
        if not resume.get('checkpoint'):
            break
        upper, root = resume['step'], Path(resume['checkpoint']).parents[2]
    evaluations = []
    for step, directory in sorted(selected.items()):
        score = read(directory / 'scores.json')
        job = read(directory / 'submission.json').get('job_id')
        evaluations.append({'job': job, 'checkpoint': 'checkpoint-' + str(step),
            'source': str(directory), 'complete': score.get('comparison_ready', False),
            'graded_cells': score.get('graded_cells'), 'pass_at_1': score.get('average_pass_at_1')})
    observed = states(e['job'] for e in evaluations)
    for evaluation in evaluations:
        evaluation['stage'] = observed.get(str(evaluation['job']), 'UNKNOWN')
    return evaluations


def original_history(root):
    rows, saved, seen = {}, set(), set()
    upper = float('inf')
    while root not in seen:
        seen.add(root)
        job = str(read(root / 'submission.json')['training'])
        resume = read(root / 'run_config.json').get('resume_state') or {}
        lower = resume.get('step', 0)
        for row in records(root / ('job-' + job) / 'audit/metrics.jsonl'):
            if 'grad_norm' in row and lower < row['step'] <= upper:
                if row['step'] in rows:
                    raise ValueError('Conflicting optimizer ancestry')
                rows[row['step']] = row
        for p in (root / ('job-' + job) / 'run').glob('checkpoint-*'):
            if p.name.split('-')[-1].isdigit() and int(p.name.split('-')[-1]) <= upper and (p / 'checkpoint.saved.json').exists():
                saved.add(int(p.name.split('-')[-1]))
        if not resume.get('checkpoint'):
            break
        upper, root = lower, Path(resume['checkpoint']).parents[2]
    return [rows[k] for k in sorted(rows)], sorted(saved)


def fetch_whitebox(args):
    from huggingface_hub import HfApi
    from dotenv import dotenv_values
    api = HfApi(token=dotenv_values(args.env_file)['HF_API_KEY'])
    job = api.inspect_job(job_id=args.job, namespace='HuggingEnvs')
    prefix = job.environment['RUN_ID'] + '/jobs/' + job.environment['RUN_OWNER']
    destination = args.out
    destination.mkdir(parents=True, exist_ok=True)
    names = ['status.json', 'run/metrics.jsonl', 'run/token_audit.jsonl',
             'trackio_verified.json', 'upload_status.json']
    api.download_bucket_files(job.environment['ARTIFACT_BUCKET'],
                              [(prefix + '/' + n, destination / n) for n in names])
    summary = summarize_artifacts(destination, 'whitebox')
    summary.update(arm='whitebox', provider='hf', job=job.id, stage=job.status.stage,
                   owner=job.environment['RUN_OWNER'], artifact_prefix=prefix, fetched_at=time.time())
    write(destination / 'snapshot.json', summary)


def comparison(arm, out, env_file):
    launch = COMPARISON / 'long-launches' / arm
    state = read(launch / 'state.json')
    if not state.get('training_job'):
        receipt = read(launch / 'submission.json')
        job = receipt.get('launcher_job')
        return {'arm': arm, 'stage': 'AWAITING_QUALIFICATION', 'step': 0, 'job': None,
                'launcher_job': job, 'launcher_state': states([job]).get(str(job), 'UNKNOWN'),
                'qualification_dependency': receipt.get('afterok'), 'launch_state': state}
    job = str(state['training_job'])
    if arm == 'opencode':
        summary = summarize_artifacts(state['training_output'], arm)
        summary.update(arm=arm, provider='slurm', job=job, stage=states([job]).get(job, 'UNKNOWN'))
        plan = read(state['controller_plan'])
        audit_out = out / ('native-tito-' + job)
        if summary['step']:
            with (out / 'native-tito.log').open('a') as stream:
                result = subprocess.run([str(Path(plan['root']) / '.venv312/bin/python'),
                    str(Path(__file__).with_name('audit_native_progress.py')), '--root', plan['root'],
                    '--training', state['training_output'], '--out', str(audit_out)],
                    stdout=stream, stderr=subprocess.STDOUT, timeout=240)
            if result.returncode:
                summary['audit_error'] = 'native_tito_audit_failed'
            else:
                audit = read(audit_out / 'summary.json')
                summary['tito'] = {'opencode': audit['opencode']}
                summary['tito_checked_at'] = audit['checked_at']
    else:
        cache = out / 'whitebox-cache'
        result = subprocess.run([sys.executable, str(Path(__file__)), 'fetch-whitebox', '--job', job,
            '--env-file', str(env_file), '--out', str(cache)], stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError('HF artifact observation failed; training state is unknown')
        summary = read(cache / 'snapshot.json')
    controller_plan = Path(state['controller_plan'])
    controller = controller_plan.parent
    summary['controller_job'] = state['controller_job']
    summary['controller_state'] = states([state['controller_job']]).get(str(state['controller_job']), 'UNKNOWN')
    summary['controller_plan'] = str(controller_plan)
    if arm == 'whitebox':
        status = read(controller / 'output/decisions/monitor.json')
        summary['evaluations'] = read(controller / 'output/decisions/state.json')
        summary['controller_alerts'] = status.get('alerts', [])
        summary['pending_evaluations'] = status.get('pending_steps', [])
        summary['published_checkpoints'] = sorted(int(p.parent.name.split('-')[-1])
            for p in (controller / 'output/manifests').glob('checkpoint-*/checkpoint.hf.ready.json'))
    else:
        summary['evaluations'] = read(controller / 'checkpoint-evals/state.json')
        summary['controller_alerts'] = read(controller / 'checkpoint-evals/monitor.json').get('alerts', [])
    return summary


def diagnose(current, prior, now):
    alerts = []
    if current.get('audit_error'):
        alerts.append(current['audit_error'])
    same = current.get('job') == prior.get('job') and current.get('step') == prior.get('step')
    last = prior.get('last_optimizer_progress_at', now) if same else now
    current['last_optimizer_progress_at'] = last
    age = now - last
    if current.get('nonfinite_updates', 0):
        alerts.append('nonfinite_optimizer_metrics')
    if current.get('max_staleness', 0) > 4:
        alerts.append('staleness_exceeds_four')
    if current.get('stage') == 'RUNNING' and age > max(1800, 3 * (current.get('mean_step_seconds_last20') or 0)):
        alerts.append('no_optimizer_progress_over_30_minutes')
    if current.get('stage') in TERMINAL - {'COMPLETED'}:
        alerts.append('trainer_ended_' + current['stage'])
    if current.get('launcher_state') in TERMINAL - {'COMPLETED'}:
        alerts.append('launch_dependency_or_launcher_failed')
    if current.get('controller_state') in TERMINAL - {'COMPLETED'}:
        alerts.append('checkpoint_controller_ended_' + current['controller_state'])
    if current.get('step', 0) >= 60:
        saved = current.get('published_checkpoints', current.get('checkpoints', []))
        if max(saved, default=0) < ((current['step'] - 10) // 50) * 50:
            alerts.append('checkpoint_save_or_publication_lag')
    if current.get('trackio', {}).get('passed') is False:
        alerts.append('trackio_verification_failed')
    logging = current.get('trackio', {})
    if logging.get('local_ok') is False:
        alerts.append('trackio_local_logging_failed')
    if logging.get('online') and logging.get('sync', {}).get('ok') is False:
        alerts.append('trackio_online_sync_failed')
    for role, job in current.get('support_jobs', {}).items():
        state = current.get('support_states', {}).get(str(job), 'UNKNOWN')
        if state in TERMINAL - {'COMPLETED'}:
            alerts.append(role + '_support_ended_' + state)
    evaluations = current.get('evaluations', [])
    if isinstance(evaluations, list):
        for evaluation in evaluations:
            if evaluation.get('stage') in TERMINAL - {'COMPLETED'} and not evaluation.get('complete'):
                alerts.append('evaluation_failed_' + str(evaluation.get('checkpoint')))
    alerts.extend('controller:' + str(a) for a in current.get('controller_alerts', []))
    tito = current.get('tito', {})
    if 'rows' in tito:
        if tito['passed'] != tito['rows']:
            alerts.append('tito_failed')
    elif any(v.get('tito_pass', 0) != v.get('completed_results', 0) or
             v.get('eligible_tokens', 0) != v.get('retained_tokens', 0) for v in tito.values() if isinstance(v, dict)):
        alerts.append('tito_failed')
    if current.get('reward_window_size') == current.get('previous_window_size') == 20:
        if current['reward_last20'] < current['reward_previous20'] - .15:
            alerts.append('reward_window_decline_investigate_task_mix')
    return alerts


def repair_controller(summary, out):
    """Only retry a dead CPU observer after an identifiable transient failure."""
    if summary.get('arm') not in {'opencode', 'whitebox'} or summary.get('controller_state') not in TERMINAL - {'COMPLETED'}:
        return None
    if summary.get('controller_alerts'):
        return None  # failed eval/provenance needs a specific recovery, not a restart loop
    job = str(summary['controller_job'])
    directory = Path(summary['controller_plan']).parent
    error = directory / ('controller-' + job + '.err')
    message = error.read_text()[-12000:] if error.exists() else ''
    transient = ('ReadTimeout', 'ConnectTimeout', 'ConnectError', 'RemoteProtocolError', 'Temporary failure in name resolution')
    if summary['controller_state'] not in {'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED'} and not any(t in message for t in transient):
        return None
    path = out / ('controller-repairs-' + summary['arm'] + '.json')
    repairs = read(path, [])
    if len(repairs) >= 2 or any(r['previous_job'] == job for r in repairs):
        return None
    # Persist an intent before sbatch. Unknown responses require reconciliation.
    intent = {'previous_job': job, 'state': 'submitting', 'at': time.time()}
    repairs.append(intent)
    write(path, repairs)
    new = subprocess.check_output(['sbatch', '--parsable', '--partition=hopper-cpu', '--cpus-per-task=2',
        '--mem=8G', '--time=36:00:00', '--job-name=cmp-controller-recovery',
        '--output=' + str(directory / 'controller-%j.out'), '--error=' + str(directory / 'controller-%j.err'),
        str(directory / 'controller.slurm')], text=True, timeout=30).strip().split(';')[0]
    if not new.isdigit():
        raise RuntimeError('Ambiguous controller restart')
    intent.update(state='submitted', new_job=new)
    write(path, repairs)
    launch_state = COMPARISON / 'long-launches' / summary['arm'] / 'state.json'
    write(launch_state, {**read(launch_state), 'controller_job': new})
    return intent


def markdown(snapshot):
    lines = ['# Three-run training progress', '', 'Checked: ' + snapshot['checked_utc'], '',
        'Automatic checks run every 10 minutes on CPU. Reward windows describe optimizer updates; '
        'fixed-test pass@1 measures checkpoint quality. Startup and changing task difficulty can change training reward.', '',
        '| Run | Job | State | Step | Reward last 20 | Previous 20 | Alerts |',
        '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for arm, r in snapshot['runs'].items():
        def value(k):
            v = r.get(k)
            return f'{v:.3f}' if isinstance(v, (int, float)) else 'pending'
        lines.append(f"| {arm} | {r.get('job') or r.get('launcher_job', 'pending')} | {r.get('stage', 'UNKNOWN')} | "
            f"{r.get('step', 0)} | {value('reward_last20')} | {value('reward_previous20')} | "
            + ('; '.join(r.get('alerts', [])) or 'none') + ' |')
    evaluations = snapshot['runs'].get('multi4', {}).get('evaluations', [])
    if evaluations:
        lines += ['', '| Multi-harness checkpoint | Eval job | State | Audited pass@1 |',
                  '| --- | --- | --- | ---: |']
        for evaluation in evaluations:
            score = evaluation.get('pass_at_1')
            value = f'{score:.1%}' if evaluation.get('complete') and score is not None else 'pending full audit'
            lines.append(f"| {evaluation['checkpoint']} | {evaluation.get('job')} | {evaluation['stage']} | {value} |")
    lines += ['', 'Independent trainer/inference GPU pairs; independent eval GPUs. The Daytona '
        'checkpoint controllers reserve training capacity and admit one comparison eval at a time. '
        'The original multi-harness run uses E2B.', '',
        'JSON snapshots retain checkpoint/eval state, TiTO evidence, logging health, numerical checks and recovery actions. '
        'Transient dead CPU controllers can be restarted twice. Training, TiTO or provenance failures are recorded for '
        'diagnosis; the monitor never resets weights in response to a reward dip.', '']
    return '\n'.join(lines)


def watch(args):
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = read(args.out / 'status.json').get('runs', {})
    while not (args.out / 'STOP').exists() and not (args.out / 'completed.json').exists():
        began = time.monotonic()
        now = time.time()
        snapshot = {'checked_at': now, 'checked_utc': datetime.now(timezone.utc).isoformat(),
            'interval_seconds': 600, 'monitor_job': os.environ.get('SLURM_JOB_ID'), 'runs': {}, 'actions': []}
        for arm in ['multi4', 'opencode', 'whitebox']:
            try:
                row = original() if arm == 'multi4' else comparison(arm, args.out, args.env_file)
                row['alerts'] = diagnose(row, previous.get(arm, {}), now)
                if args.repair_controllers:
                    action = repair_controller(row, args.out)
                    if action:
                        snapshot['actions'].append({'arm': arm, **action})
            except Exception as exc:
                row = {**previous.get(arm, {}), 'observation_error': type(exc).__name__,
                       'alerts': ['observation_failed_' + type(exc).__name__]}
            snapshot['runs'][arm] = row
        write(args.out / 'status.json', snapshot)
        with (args.out / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(snapshot) + '\n')
        rendered = markdown(snapshot)
        args.markdown.write_text(rendered)
        with (args.out / 'HISTORY.md').open('a') as stream:
            stream.write(rendered + '\n---\n\n')
        changes = {arm: r['alerts'] for arm, r in snapshot['runs'].items()
                   if r['alerts'] != previous.get(arm, {}).get('alerts', [])}
        if changes or snapshot['actions']:
            with (args.out / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps({'checked_at': now, 'alert_changes': changes, 'actions': snapshot['actions']}) + '\n')
        print(json.dumps({'at': snapshot['checked_utc'], 'runs': {k: {f: r.get(f) for f in
            ['job', 'stage', 'step', 'alerts']} for k, r in snapshot['runs'].items()}}), flush=True)
        previous = snapshot['runs']
        if (all(r.get('stage') == 'COMPLETED' and not r.get('observation_error') for r in previous.values())
                and previous['multi4'].get('step', 0) >= 1000
                and all(previous[a].get('controller_state') == 'COMPLETED' for a in ['opencode', 'whitebox'])):
            write(args.out / 'completed.json', {'at': now, 'reason': 'all three runs and comparison eval controllers completed'})
            break
        if args.once:
            break
        time.sleep(max(1, 600 - (time.monotonic() - began)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    watcher = commands.add_parser('watch')
    watcher.add_argument('--out', type=Path, required=True)
    watcher.add_argument('--env-file', type=Path, required=True)
    watcher.add_argument('--markdown', type=Path, required=True)
    watcher.add_argument('--once', action='store_true')
    watcher.add_argument('--repair-controllers', action='store_true')
    fetch = commands.add_parser('fetch-whitebox')
    fetch.add_argument('--job', required=True)
    fetch.add_argument('--out', type=Path, required=True)
    fetch.add_argument('--env-file', type=Path, required=True)
    args = parser.parse_args()
    (fetch_whitebox if args.command == 'fetch-whitebox' else watch)(args)
