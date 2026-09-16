"""CPU-only Slurm watchdog. Writes local status/alerts; never changes or stops training."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_metrics(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    # A trainer can be in the middle of appending its latest line.
    return [json.loads(line) for line in lines if line.endswith('\n') and line.strip()]


def slurm_states(jobs):
    if not jobs:
        return {}
    result = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(map(str, jobs)),
        '--format=JobID,State'], text=True, timeout=30)
    found = {line.split('|')[0]: line.split('|')[1].split()[0].rstrip('+')
             for line in result.splitlines() if '|' in line}
    return {str(job): found.get(str(job), 'UNKNOWN') for job in jobs}


def resolved_empty_captures(root, logs, harness):
    """Resolve individually investigated incidents without waiving token validation.

    Historical failed attempts remain failed in the audit and dashboard. Only named
    incidents with verified recovery can stop paging; every new failure still alerts.
    An acknowledgement can never resolve a capture containing model tokens.
    """
    incidents = read_json(root / 'monitor/resolved_incidents.json', {}).get('episodes', {})
    failed, resolved = set(), set()
    for path in (logs / 'audit/tito_checks').glob('*.json'):
        check = read_json(path)
        if (check.get('harness') != harness or not check.get('completed_result')
                or check.get('tito_pass')):
            continue
        failed.add(path.stem)
        incident = incidents.get(path.stem, {})
        if not incident.get('verified_at') or not incident.get('recovery_evidence'):
            continue
        result = (read_json(logs / 'audit/rollouts' / path.name, {}) or {}).get('result') or {}
        fatal = [f for f in result.get('findings', []) if '[FATAL]' in f]
        if (result.get('n_turns') == 0 and result.get('n_trainable_tokens') == 0
                and result.get('turns') == [] and fatal
                and all(f.startswith('[FATAL] no_turns:') for f in fatal)):
            resolved.add(path.stem)
    return failed, resolved


def inspect_run(root, job, states, *, now=None):
    now = time.time() if now is None else now
    logs = root / f'job-{job}'
    config = read_json(root / 'run_config.json', {})
    metrics_path = logs / 'audit/metrics.jsonl'
    metrics = read_metrics(metrics_path)
    updates = [m for m in metrics if 'grad_norm' in m]
    latest = updates[-1] if updates else {}
    resume_step = config.get('resume_state', {}).get('step', 0)
    step = latest.get('step', resume_step)
    alerts = []
    state = states.get(str(job), 'UNKNOWN')
    if state in TERMINAL - {'COMPLETED'}:
        alerts.append(f'training_{state.lower()}')
    if state == 'UNKNOWN':
        alerts.append('training_state_unknown')
    for m in updates[-5:]:
        if any(isinstance(m.get(k), (float, int)) and not math.isfinite(m[k])
               for k in ('loss', 'grad_norm', 'ratio')):
            alerts.append('nonfinite_training_metric')
    if len(updates) >= 5 and all(m.get('grad_norm', 0) == 0 for m in updates[-5:]):
        alerts.append('five_updates_without_gradient')
    for metric, label in [('sample/dropped_stale_total', 'stale_rows_dropped'),
                          ('batch/dropped_oversize_total', 'oversize_rows_dropped')]:
        if any(m.get(metric, 0) > 0 for m in updates[-5:]):
            alerts.append(label)
    age = now - metrics_path.stat().st_mtime if metrics_path.exists() else None
    if state == 'RUNNING' and age is not None and age > 1800:
        alerts.append('no_optimizer_update_for_30_minutes')
    if state == 'RUNNING' and age is None and logs.exists() and now - logs.stat().st_mtime > 1800:
        alerts.append('no_optimizer_metrics_after_startup')
    tito_path = logs / 'audit/tito_summary.json'
    tito = read_json(tito_path, {})
    resolved_incidents = {}
    for harness, summary in tito.items():
        if summary['tito_pass'] != summary.get('completed_results', summary['rollouts']):
            failed, resolved = resolved_empty_captures(root, logs, harness)
            expected_failures = summary.get('completed_results', summary['rollouts']) - summary['tito_pass']
            if failed != resolved or len(failed) != expected_failures:
                alerts.append(f'tito_failure:{harness}')
            if resolved:
                resolved_incidents[harness] = sorted(resolved)
        if summary.get('incomplete_results', 0) and state == 'RUNNING':
            alerts.append(f'incomplete_rollout:{harness}')
        if summary['retained_tokens'] != summary['eligible_tokens'] or summary['rows_over_token_budget']:
            alerts.append(f'token_retention_failure:{harness}')
    evals = read_json(root / 'checkpoint-evals/state.json', {})
    evaluation = []
    for key, record in evals.items():
        eval_state = states.get(str(record.get('job_id')), record.get('slurm_state', 'UNKNOWN'))
        result = record.get('scores', {})
        if eval_state in TERMINAL - {'COMPLETED'}:
            alerts.append(f'eval_{eval_state.lower()}:{Path(key).name}')
        if eval_state == 'COMPLETED' and result and not result.get('comparison_ready', False):
            alerts.append(f'eval_not_comparable:{Path(key).name}')
        evaluation.append({'checkpoint': key, 'job_id': record.get('job_id'), 'state': eval_state,
                           'scores': result})
    submission = read_json(root / 'submission.json', {})
    watcher = submission.get('eval_watcher')
    if watcher and (states.get(str(watcher)) in TERMINAL - {'COMPLETED'}
                    or states.get(str(watcher)) == 'COMPLETED' and state not in TERMINAL):
        alerts.append('eval_watcher_stopped_early')
    saved = sorted(int(p.name.split('-')[1]) for p in (logs / 'run').glob('checkpoint-*')
                   if (p / 'checkpoint.saved.json').exists() or (p / 'checkpoint.ready.json').exists())
    interval = config.get('evaluation', {}).get('interval_optimizer_steps', 100)
    submitted = {int(Path(k).name.split('-')[1]) for k in evals}
    pending = [s for s in saved if s % interval == 0 and s not in submitted]
    return {'checked_at': datetime.fromtimestamp(now, timezone.utc).isoformat(),
            'job_id': job, 'training_state': state, 'optimizer_step': step,
            'resumed_from_step': resume_step,
            'new_optimizer_updates': len(updates),
            'awaiting_first_optimizer_update': not updates and state == 'RUNNING',
            'metrics_age_seconds': age, 'last_metrics': latest,
            'counters': {'stale_rows_dropped': sum(m.get('sample/dropped_stale_total', 0) for m in updates),
                         'oversize_rows_dropped': sum(m.get('batch/dropped_oversize_total', 0) for m in updates),
                         'nonzero_gradient_updates': sum(m.get('grad_norm', 0) > 0 for m in updates),
                         'resolved_empty_capture_incidents': sum(map(len, resolved_incidents.values()))},
            'resolved_empty_capture_incidents': resolved_incidents,
            'coverage': read_json(logs / 'audit/coverage.json', {}), 'tito': tito,
            'tito_audit_age_seconds': now - tito_path.stat().st_mtime if tito_path.exists() else None,
            'saved_checkpoints': saved, 'queued_eval_steps': pending, 'evaluations': evaluation,
            'job_states': states, 'alerts': sorted(set(alerts)),
            'monitoring_scope': 'local artifacts only; no chat wakeups or external notifications'}


def next_interval(previous, current, stable_checks, config):
    threshold = config.get('stable_after_optimizer_step', 10)
    required = config.get('required_progressing_checks', 2)
    if (current['alerts'] or not previous or current['optimizer_step'] < threshold
            or current['optimizer_step'] < previous['optimizer_step']):
        stable_checks = 0
    elif current['optimizer_step'] > previous['optimizer_step']:
        stable_checks += 1
    # An unchanged step during an ordinary rollout wait is not a failure. Keep the
    # evidence of healthy updates; inspect_run's stall/error alerts reset it above.
    interval = config['stable_interval_seconds'] if stable_checks >= required else config['startup_interval_seconds']
    return stable_checks, interval


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--train-job', required=True)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--audit', action='store_true', help='Replay newly captured rollouts on this CPU job')
    args = p.parse_args()
    config = read_json(args.run / 'run_config.json')['monitoring']
    destination = args.run / 'monitor'
    destination.mkdir(exist_ok=True)
    previous, stable_checks = None, 0
    while True:
        cycle_started = time.monotonic()
        try:
            submission = read_json(args.run / 'submission.json', {})
            evals = read_json(args.run / 'checkpoint-evals/state.json', {})
            jobs = {args.train_job} | {str(j) for j in submission.values() if str(j).isdigit()}
            jobs.update(str(r[k]) for r in evals.values() for k in ('job_id', 'cleanup_job_id') if r.get(k))
            audit_error = None
            if args.audit:
                audit_dir = args.run / f'job-{args.train_job}/audit'
                if audit_dir.exists():
                    with (destination / 'audit.log').open('a') as stream:
                        try:
                            subprocess.run(['nice', '-n', '10', sys.executable,
                                str(Path(__file__).with_name('audit_multiharness_training.py')), str(audit_dir)],
                                check=True, timeout=600, stdout=stream, stderr=subprocess.STDOUT)
                        except (subprocess.SubprocessError, OSError) as exc:
                            audit_error = type(exc).__name__
            status = inspect_run(args.run, args.train_job, slurm_states(jobs))
            if audit_error:
                status['alerts'].append('capture_audit_failed:' + audit_error)
            stable_checks, interval = next_interval(previous, status, stable_checks, config)
            status['next_check_seconds'] = interval
            temporary = destination / 'status.json.tmp'
            temporary.write_text(json.dumps(status, indent=2) + '\n')
            temporary.replace(destination / 'status.json')
            with (destination / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(status) + '\n')
            old_alerts = set(previous['alerts']) if previous else set()
            changes = {'new': sorted(set(status['alerts']) - old_alerts),
                       'resolved': sorted(old_alerts - set(status['alerts']))}
            if changes['new'] or changes['resolved']:
                with (destination / 'alerts.jsonl').open('a') as stream:
                    stream.write(json.dumps({'checked_at': status['checked_at'], **changes}) + '\n')
            print(json.dumps({'step': status['optimizer_step'], 'state': status['training_state'],
                              'alerts': status['alerts'], 'next_check_seconds': interval}), flush=True)
            previous = status
            # Continue through final audit/cleanup and outstanding evals; exclude our own monitor job.
            other_jobs = {j: s for j, s in status['job_states'].items() if j != str(submission.get('monitor'))}
            if status['training_state'] in TERMINAL and all(s in TERMINAL for s in other_jobs.values()):
                break
        except Exception as exc:
            with (destination / 'alerts.jsonl').open('a') as stream:
                stream.write(json.dumps({'checked_at': time.time(), 'monitor_error': str(exc)}) + '\n')
            if not args.watch:
                raise
            interval = config['startup_interval_seconds']
        if not args.watch:
            break
        time.sleep(max(1, interval - (time.monotonic() - cycle_started)))


if __name__ == '__main__':
    main()
