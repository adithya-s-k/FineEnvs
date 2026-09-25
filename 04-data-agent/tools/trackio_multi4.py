"""Replay durable scalar metrics into Trackio; sync on a CPU job, never on the trainer.

The small fragment adapter is pinned to Trackio 0.33.0. Its existing log_id deduplication
makes replay/restarts idempotent. SQLite lives on node-local disk; a consistent backup is
atomically published to FSx before any network call. Raw captures/completions stay local.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
import time

from monitor_multi4 import TERMINAL, read_json, read_metrics, slurm_states

TRACKIO_VERSION = '0.33.0'
REMOTE_ENV = ('TRACKIO_SPACE_ID', 'TRACKIO_SERVER_URL', 'TRACKIO_BUCKET_ID',
              'TRACKIO_DATASET_ID', 'TRACKIO_WEBHOOK_URL')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def scalars(data, prefix=''):
    result = {}
    for key, value in data.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            result.update(scalars(value, name + '/'))
        elif isinstance(value, (int, float)):
            if math.isfinite(value):
                result[name] = value
            else:
                result[name + '/nonfinite'] = 1
    return result


def event(project, run, step, metrics, config, *, identity=None):
    """Stable IDs survive source replay, worker restart, and out-of-order eval completion."""
    content = [project, run, step, metrics, identity]
    return {'v': 1, 'kind': 'metric', 'project': project, 'run': run,
            'run_id': digest([project, run])[:32], 'step': int(step), 'metrics': metrics,
            'config': config, 'log_id': digest(content),
            'timestamp': datetime.now(timezone.utc).isoformat()}


def score_metrics(scores, protocol):
    metrics = {'eval/pass_at_1': scores['average_pass_at_1'],
               'eval/delta_from_baseline': scores['average_pass_at_1'] - protocol['average_pass_at_1'],
               'eval/graded_cells': scores['graded_cells']}
    totals = {}
    for harness, values in scores['harnesses'].items():
        metrics[f'eval/{harness}/pass_at_1'] = values['pass_at_1']
        metrics[f'eval/{harness}/delta_from_baseline'] = values['pass_at_1'] - protocol['scores'][harness]['pass_at_1']
        for difficulty, counts in values.get('difficulty', {}).items():
            n = counts['graded']
            metrics[f'eval/{harness}/{difficulty}/pass_at_1'] = counts['correct'] / n
            total = totals.setdefault(difficulty, [0, 0])
            total[0] += counts['correct']; total[1] += n
    for difficulty, (correct, n) in totals.items():
        metrics[f'eval/{difficulty}/pass_at_1'] = correct / n
    return metrics


def evaluation_roots(root, config=None):
    """Include explicitly configured earlier allocations whose evals can finish late."""
    config = config if config is not None else read_json(root / 'run_config.json', {})
    roots = [root.resolve()]
    for value in config.get('logging', {}).get('evaluation_sources', []):
        source = Path(value)
        if not source.is_absolute() or not (source / 'run_config.json').is_file():
            raise ValueError('Evaluation source must be an existing absolute run directory')
        if source.resolve() not in roots:
            roots.append(source.resolve())
    return roots


def training_lineage(root, job):
    """Follow actual resume checkpoints, excluding abandoned post-checkpoint updates."""
    segments, seen = [], set()
    current, current_job, upper = root.resolve(), str(job), None
    while True:
        if current in seen:
            raise ValueError('Cycle in training checkpoint lineage')
        seen.add(current)
        config = read_json(current / 'run_config.json')
        resume = config.get('training', {}).get('resume_from_checkpoint')
        lower = 0
        parent = None
        if resume:
            checkpoint = Path(resume)
            checkpoint_match = re.fullmatch(r'checkpoint-(\d+)', checkpoint.name)
            job_match = re.fullmatch(r'job-(\d+)', checkpoint.parent.parent.name)
            if not checkpoint.is_absolute() or checkpoint.parent.name != 'run' or not checkpoint_match or not job_match:
                raise ValueError('Invalid training checkpoint lineage path')
            lower = int(checkpoint_match[1])
            parent = (checkpoint.parents[2], job_match[1], lower)
        selected = {}
        for row in read_metrics(current / f'job-{current_job}/audit/metrics.jsonl'):
            # Trainer's final aggregate summary can reuse its last optimizer step.
            if 'grad_norm' not in row:
                continue
            step = row['step']
            if step <= lower or (upper is not None and step > upper):
                continue
            if step in selected and selected[step] != row:
                raise ValueError(f'Conflicting optimizer records at step {step}')
            selected[step] = row
        if upper is None:
            upper = max(selected, default=lower)
        if sorted(selected) != list(range(lower + 1, upper + 1)):
            raise ValueError(f'Incomplete optimizer history in job {current_job}: expected {lower + 1}..{upper}')
        segments.append({'job': current_job, 'root': str(current), 'start_step': lower + 1,
                         'end_step': upper, 'resume_from_checkpoint': resume,
                         'training': config.get('training', {}), 'rows': list(selected.values())})
        if parent is None:
            break
        current, current_job, upper = parent
        current = current.resolve()
    return list(reversed(segments))


def collect(root, job):
    config = read_json(root / 'run_config.json')
    project = config['logging']['project']
    protocol = read_json(Path(config['evaluation']['protocol_file']))
    # Explicit allowlist: never serialize process environment, credentials, task text or captures.
    metadata = {k: config[k] for k in ('model', 'model_revision', 'harnesses', 'harness_versions',
                                      'training', 'sampling')}
    metadata.update(train_job=job, run_directory=str(root),
                    protocol_sha256=digest(protocol), trackio_version=TRACKIO_VERSION,
                    schedule_sha256=config['dataset']['schedule_sha256'],
                    metric_axis='optimizer step', tito_scope='capture and sequence builder; not optimizer consumption')
    records, withheld = [], []
    logs = root / f'job-{job}' if job else None
    if logs:
        for row in read_metrics(logs / 'audit/metrics.jsonl'):
            metrics = scalars({k: v for k, v in row.items() if k != 'step'}, 'train/')
            metrics['train/global_step'] = row['step']
            records.append(event(project, f'training-{job}', row['step'], metrics, metadata))
        for row in read_metrics(root / 'monitor/history.jsonl'):
            metrics = scalars({k: row.get(k, {}) for k in ('counters', 'coverage', 'tito')}, 'audit/')
            metrics.update({'audit/alert_count': len(row.get('alerts', [])),
                            'audit/eval_backlog': len(row.get('queued_eval_steps', []))})
            for harness, values in row.get('tito', {}).items():
                completed = values.get('completed_results')
                if completed:
                    metrics[f'audit/tito/{harness}/pass_fraction_completed'] = values['tito_pass'] / completed
                eligible = values.get('eligible_tokens', 0)
                if eligible:
                    metrics[f'audit/tito/{harness}/retention_fraction'] = values['retained_tokens'] / eligible
                graded = values.get('graded', 0)
                if graded:
                    metrics[f'audit/{harness}/reward_mean'] = values['reward_sum'] / graded
            records.append(event(project, f'audit-{job}', row['optimizer_step'], metrics, metadata,
                                 identity=row.get('checked_at')))
    if job and config['logging'].get('stitch_training_history'):
        segments = training_lineage(root, job)
        lineage_metadata = {k: v for k, v in metadata.items() if k != 'training'}
        lineage_metadata.update(
            description='Continuous optimizer history along the checkpoint resume chain; configuration changed between some allocations.',
            training_segments=[{k: v for k, v in segment.items() if k != 'rows'} for segment in segments])
        for segment in segments:
            for row in segment['rows']:
                metrics = scalars({k: v for k, v in row.items() if k != 'step'}, 'train/')
                metrics.update({'train/global_step': row['step'], 'train/source_job_id': int(segment['job'])})
                records.append(event(project, 'training-full-history', row['step'], metrics, lineage_metadata))
    baseline_path = Path(protocol['baseline_run']) / f"job-{protocol['baseline_job']}" / 'canonical_results.json'
    baseline = read_json(baseline_path)
    if not baseline['coverage_complete']:
        raise ValueError('Baseline coverage is incomplete')
    baseline_scores = {'average_pass_at_1': protocol['average_pass_at_1'], 'graded_cells': 1000,
        'harnesses': {h: {'pass_at_1': protocol['scores'][h]['pass_at_1'],
                         'difficulty': baseline['harnesses'][h]['difficulty']} for h in protocol['harnesses']}}
    records.append(event(project, 'evaluation-curve', 0, score_metrics(baseline_scores, protocol), metadata))
    evaluations = {}
    for source in evaluation_roots(root, config):
        for directory in sorted((source / 'checkpoint-evals').glob('step-*')):
            if not directory.is_dir():
                continue
            scores = read_json(directory / 'scores.json', {})
            step = int(directory.name.split('-')[1])
            eval_config = read_json(directory / 'eval_plan.json', {})
            comparable = (scores.get('complete') and scores.get('comparison_ready')
                          and scores.get('graded_cells') == 1000 and eval_config.get('protocol') == protocol
                          and set(scores.get('harnesses', {})) == set(protocol['harnesses'])
                          and all(v.get('graded') == 250 for v in scores.get('harnesses', {}).values()))
            if not comparable:
                withheld.append({'step': step, 'source': str(directory),
                                 'reason': 'incomplete, audit/version failure, or protocol mismatch'})
                continue
            metrics = score_metrics(scores, protocol)
            if step in evaluations and evaluations[step] != metrics:
                raise ValueError(f'Conflicting evaluation scores at optimizer step {step}')
            evaluations[step] = metrics
    for step, metrics in sorted(evaluations.items()):
        records.append(event(project, 'evaluation-curve', step, metrics, metadata))
    return records, withheld


def import_events(records):
    if version('trackio') != TRACKIO_VERSION:
        raise RuntimeError(f'Trackio adapter requires {TRACKIO_VERSION}; installed {version("trackio")}')
    from trackio.fragments import import_records
    import_records(records)


def backup_project(project, destination):
    from trackio.sqlite_storage import SQLiteStorage
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / SQLiteStorage.get_project_db_filename(project)
    temporary = target.with_suffix('.db.tmp')
    with sqlite3.connect(SQLiteStorage.get_project_db_path(project)) as source:
        with sqlite3.connect(temporary) as output:
            source.backup(output)
            output.execute('PRAGMA journal_mode=DELETE')
    temporary.replace(target)


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def configuration_records(project):
    """Replay one existing log per run with its allowlisted configuration.

    Trackio 0.33.0's get_all_logs_for_sync emits config=None. Its native bulk_log
    endpoint can still store config on an existing log_id without adding a metric.
    """
    from trackio.sqlite_storage import SQLiteStorage
    records, seen = [], set()
    for entry in SQLiteStorage.get_all_logs_for_sync(project):
        identity = entry.get('run_id') or entry['run']
        if identity in seen:
            continue
        seen.add(identity)
        config = SQLiteStorage.get_run_config(project, entry['run'], run_id=entry.get('run_id'))
        if config:
            records.append({**entry, 'config': config})
    return records


def sync_project(config):
    # Token is inherited or loaded only in the sync subprocess; never logged or put in config.
    from dotenv import dotenv_values
    values = dotenv_values(os.environ.get('DATA_AGENT_ENV_FILE', '.env'))
    token = os.environ.get('HF_TOKEN') or values.get('HF_TOKEN') or values.get('HF_API_KEY')
    if token:
        os.environ['HF_TOKEN'] = token
    from trackio.deploy import create_space_if_not_exists
    logging = config['logging']
    # Provision once, then send stable log IDs through Trackio's bulk API. Replacing
    # the live Space's mounted SQLite file can leave open readers on an old inode.
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        info = HfApi().space_info(logging['space_id'])
        if not info.private or info.sdk != 'gradio':
            raise ValueError('Online metrics require the configured private Gradio Space')
    except RepositoryNotFoundError:
        create_space_if_not_exists(logging['space_id'], bucket_id=logging['bucket_id'], private=True)
    from trackio.remote_client import RemoteClient
    from trackio.sqlite_storage import SQLiteStorage
    client = RemoteClient(logging['space_id'], hf_token=token, httpx_kwargs={'timeout': 30})
    records = SQLiteStorage.get_all_logs_for_sync(logging['project'])
    # Multiple allocation collectors share evaluation-curve. Trackio 0.33.0's
    # sync_incremental waits for exact remote/local row-count equality, which
    # cannot hold when a peer has already published additional checkpoint scores.
    # Keep native bulk_log deduplication, then verify our exact event contents.
    for start in range(0, len(records), 500):
        client.predict(api_name='/bulk_log', logs=records[start:start + 500], hf_token=token)
    metadata = configuration_records(logging['project'])
    if metadata:
        client.predict(api_name='/bulk_log', logs=metadata, hf_token=token)
    proof = verify_remote_records(client, logging['project'], records)
    for entry in metadata:
        summary = client.predict(api_name='/get_run_summary', project=logging['project'],
                                 run_id=entry['run_id'])
        if summary.get('config') != entry['config']:
            raise RuntimeError('Remote Trackio configuration differs for ' + entry['run'])
    return {**proof, 'configuration_runs_verified': len(metadata),
            'checked_at': datetime.now(timezone.utc).isoformat()}


def verify_remote_records(client, project, records, timeout=90):
    """Allow other publishers' events; require every local ID and scalar payload."""
    expected = {}
    for entry in records:
        identity = entry.get('log_id', '')
        if entry.get('project') != project or not re.fullmatch(r'[0-9a-f]{64}', identity):
            raise ValueError('Invalid scalar event identity')
        value = {k: entry[k] for k in ('run_id', 'step', 'metrics')}
        value['run_name'] = entry['run']
        if identity in expected and expected[identity] != value:
            raise ValueError('Conflicting local scalar event')
        expected[identity] = value
    pending = dict(expected)
    deadline = time.monotonic() + timeout
    while pending:
        keys = sorted(pending)
        for start in range(0, len(keys), 100):
            # IDs are validated SHA256 hex strings, never arbitrary SQL input.
            ids = ','.join("'" + k + "'" for k in keys[start:start + 100])
            result = client.predict(api_name='/query_project', project=project,
                query='SELECT log_id, run_id, run_name, step, CAST(metrics AS TEXT) AS metrics FROM metrics WHERE log_id IN (' + ids + ')')
            for row in result['rows']:
                identity = row['log_id']
                actual = {k: row[k] for k in ('run_id', 'run_name', 'step')}
                actual['metrics'] = json.loads(row['metrics']) if isinstance(row['metrics'], str) else row['metrics']
                if identity not in expected or actual != expected[identity]:
                    raise RuntimeError('Remote Trackio event content mismatch')
                pending.pop(identity, None)
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f'{len(pending)} Trackio events are not remotely visible')
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    return {'ok': True, 'events_verified': len(expected), 'verification': 'exact log_id/run_id/step/metrics'}


def bounded_sync(root, seconds, log):
    try:
        with log.open('a') as stream:
            result = subprocess.run([sys.executable, '-u', __file__, '--run', str(root), '--sync-only'],
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=seconds)
        return {'ok': result.returncode == 0, 'returncode': result.returncode}
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'sync_timeout', 'timeout_seconds': seconds}


def work_finished(root, job):
    submission = read_json(root / 'submission.json', {})
    # Do not wait for the monitor: it waits for us. Include watcher to prevent early exit
    # between training finishing and a just-saved checkpoint being submitted for evaluation.
    jobs = [str(job)] + [str(submission[k]) for k in ('cleanup', 'eval_watcher') if k in submission]
    for source in evaluation_roots(root):
        for record in read_json(source / 'checkpoint-evals/state.json', {}).values():
            jobs += [str(record[k]) for k in ('job_id', 'cleanup_job_id') if record.get(k)]
        # Recovery jobs can be submitted after an allocation's watcher exits.
        for path in (source / 'checkpoint-evals').glob('step-*/submission.json'):
            record = read_json(path, {})
            jobs += [str(record[k]) for k in ('job_id', 'cleanup_job_id') if record.get(k)]
    states = slurm_states(sorted(set(jobs)))
    return set(states) == set(jobs) and all(value in TERMINAL for value in states.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--train-job', default='')
    p.add_argument('--watch', action='store_true')
    p.add_argument('--online', action='store_true')
    p.add_argument('--sync-only', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    root = args.run.resolve()
    config = read_json(root / 'run_config.json')
    if args.sync_only:
        proof = sync_project(config)
        write_json(root / 'trackio/sync-receipt.json', proof)
        return
    if args.watch and not args.train_job:
        p.error('--watch requires --train-job')
    destination = root / 'trackio'
    destination.mkdir(exist_ok=True)
    for key in REMOTE_ENV:
        os.environ.pop(key, None)
    with (destination / '.collector.lock').open('w') as lock, tempfile.TemporaryDirectory(prefix='multi4-trackio-') as scratch:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ['TRACKIO_DIR'] = scratch
        os.environ['TRACKIO_STORAGE_MODE'] = 'sqlite'
        last_synced = None
        while True:
            start = time.monotonic()
            state = {'checked_at': datetime.now(timezone.utc).isoformat(), 'online': args.online}
            try:
                records, withheld = collect(root, args.train_job)
                import_events(records)
                backup_project(config['logging']['project'], destination / 'dashboard')
                state.update(events=len(records), withheld_evaluations=withheld, local_ok=True,
                             evaluation_sources=[str(p) for p in evaluation_roots(root, config)],
                             evaluation_steps=sorted({r['step'] for r in records if r['run'] == 'evaluation-curve'}),
                             full_history_steps=len([r for r in records if r['run'] == 'training-full-history']),
                             training_step=max((r['step'] for r in records if r['run'] == f'training-{args.train_job}'), default=None))
                current = digest([r['log_id'] for r in records])
                if args.online and current != last_synced:
                    state['sync'] = bounded_sync(root, config['logging']['sync_timeout_seconds'], destination / 'sync.log')
                    if state['sync']['ok']:
                        last_synced = current
                else:
                    state['sync'] = {'ok': bool(last_synced), 'skipped_unchanged': True}
                finished = args.watch and work_finished(root, args.train_job)
            except Exception as exc:
                state.update(error=type(exc).__name__ + ': ' + str(exc), local_ok=False)
                finished = False
            write_json(destination / 'status.json', state)
            with (destination / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(state) + '\n')
            print(json.dumps(state), flush=True)
            if not args.watch:
                if not state.get('local_ok') or (args.online and not state.get('sync', {}).get('ok')):
                    raise SystemExit(1)
                break
            if finished and (not args.online or state.get('sync', {}).get('ok')):
                break
            time.sleep(max(1, config['logging']['poll_seconds'] - (time.monotonic() - start)))


if __name__ == '__main__':
    main()
