import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('monitor_three_runs', Path(__file__).parents[1] / 'monitor_three_runs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_summary_not_a_second_optimizer_update():
    rows = [{'step': 1, 'reward': .25, 'grad_norm': 2}, {'step': 1, 'train_runtime': 4},
            {'step': 2, 'reward': .5, 'grad_norm': 0}]
    r = module.training_metrics(rows)
    assert r['updates'] == 2 and r['reward_last20'] == .375
    assert r['nonzero_gradients_last20'] == 1


def test_timeout_observation_does_not_imply_training_failure():
    prior = {'job': 'a', 'step': 12, 'last_optimizer_progress_at': 100}
    current = {'job': 'a', 'step': 12, 'stage': 'UNKNOWN'}
    assert not module.diagnose(current, prior, 2100)
    current['stage'] = 'RUNNING'
    assert 'no_optimizer_progress_over_30_minutes' in module.diagnose(current, prior, 2100)


def test_resume_progress_and_tito_failure():
    prior = {'job': 'a', 'step': 100, 'last_optimizer_progress_at': 100}
    current = {'job': 'b', 'step': 100, 'stage': 'RUNNING', 'checkpoints': [100],
               'tito': {'rows': 8, 'passed': 7}}
    alerts = module.diagnose(current, prior, 2200)
    assert alerts == ['tito_failed']
    assert current['last_optimizer_progress_at'] == 2200


def test_enabled_online_logging_is_not_successful_sync():
    row = {'trackio': {'online': True, 'local_ok': True, 'sync': {'ok': False}},
           'evaluations': [{'checkpoint': 'checkpoint-600', 'stage': 'FAILED', 'complete': False}]}
    assert module.diagnose(row, {}, 1) == ['trackio_online_sync_failed', 'evaluation_failed_checkpoint-600']
    row['trackio']['sync']['ok'] = True
    row['evaluations'][0]['stage'] = 'RUNNING'
    assert module.diagnose(row, {}, 2) == []


def test_parent_evaluations_remain_visible_after_resume(tmp_path, monkeypatch):
    import json
    parent, child = tmp_path / 'parent', tmp_path / 'child'
    child.mkdir(); parent.mkdir()
    (parent / 'run_config.json').write_text('{}')
    (child / 'run_config.json').write_text(json.dumps({'resume_state': {
        'step': 150, 'checkpoint': str(parent / 'job-1/run/checkpoint-150')}}))
    for root, steps in [(parent, [100, 200]), (child, [300])]:
        for step in steps:
            d = root / f'checkpoint-evals/step-{step:06d}'
            d.mkdir(parents=True)
            (d / 'submission.json').write_text(json.dumps({'job_id': str(step)}))
            (d / 'scores.json').write_text(json.dumps({'comparison_ready': step == 100,
                'average_pass_at_1': .25, 'graded_cells': 1000}))
    monkeypatch.setattr(module, 'states', lambda jobs: {str(j): 'COMPLETED' for j in jobs})
    evaluations = module.original_evaluations(child)
    assert [e['checkpoint'] for e in evaluations] == ['checkpoint-100', 'checkpoint-300']
    assert evaluations[0]['complete'] is True


def test_restore_startup_retains_parent_metrics_and_checkpoints(tmp_path):
    import json
    parent, child = tmp_path / 'parent', tmp_path / 'child'
    checkpoint = parent / 'job-1/run/checkpoint-2'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'checkpoint.saved.json').write_text('{}')
    (parent / 'submission.json').write_text('{"training": "1"}')
    (parent / 'run_config.json').write_text('{}')
    audit = parent / 'job-1/audit'
    audit.mkdir()
    (audit / 'metrics.jsonl').write_text(''.join(json.dumps({'step': s, 'reward': .5, 'grad_norm': 1}) + '\n' for s in [1,2,3]))
    child.mkdir()
    (child / 'submission.json').write_text('{"training": "2"}')
    (child / 'run_config.json').write_text(json.dumps({'resume_state': {'step': 2, 'checkpoint': str(checkpoint)}}))
    rows, saves = module.original_history(child)
    assert [r['step'] for r in rows] == [1,2]
    assert saves == [2]
