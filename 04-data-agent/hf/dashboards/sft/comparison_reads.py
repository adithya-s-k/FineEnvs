"""Bound dense plotting data while preserving sparse metrics and raw SQL access."""
from functools import wraps
import math


def plot_rows(rows, limit=800):
    dense = [r for r in rows if any(k.startswith(('train/', 'train_sft/', 'train_verified/')) for k in r)]
    stride = max(1, math.ceil(len(dense) / limit))
    keep = {id(r) for r in dense[::stride]}
    if dense:
        keep.add(id(dense[-1]))
    return [r for r in rows if id(r) in keep or not any(k.startswith(('train/', 'train_sft/', 'train_verified/')) for k in r)
            or any(k.startswith(('eval', 'train_summary/')) for k in r)]


def install(server, storage, project, hidden_run_ids=()):
    hidden = set(hidden_run_ids)
    original_logs, original_batch = server.get_logs, server.get_logs_batch
    original_runs, original_configs = server.get_runs_for_project, server.get_run_configs

    @wraps(original_logs)
    def get_logs(project, run=None, run_id=None, scalar_only=False):
        if project != target:
            return original_logs(project, run, run_id, scalar_only)
        rows = storage.get_logs(project, run, max_points=None, run_id=run_id,
            scalar_only=server._normalize_bool_param(scalar_only, 'scalar_only'))
        return plot_rows(rows)

    @wraps(original_batch)
    def get_logs_batch(project, runs, max_points=3000, scalar_only=False):
        if project != target:
            return original_batch(project, runs, max_points, scalar_only)
        # Fetch runs separately so dense series from one run cannot displace another's evals.
        return [{'run': r.get('run'), 'run_id': r.get('run_id'),
                 'logs': get_logs(project, r.get('run'), r.get('run_id'), scalar_only)}
                for r in server._normalize_logs_batch_runs(runs)]

    @wraps(original_runs)
    def get_runs_for_project(project):
        rows = original_runs(project)
        return [r for r in rows if r['id'] not in hidden] if project == target else rows

    @wraps(original_configs)
    def get_run_configs(project):
        configs = original_configs(project)
        return {k: v for k, v in configs.items() if k not in hidden} if project == target else configs

    target = project
    server.get_logs, server.get_logs_batch = get_logs, get_logs_batch
    server.get_runs_for_project, server.get_run_configs = get_runs_for_project, get_run_configs
