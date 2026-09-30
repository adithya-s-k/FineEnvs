"""Checkpoint eligibility, score denominators and retry selection without GPU jobs."""
import json
from pathlib import Path

import pytest

from checkpoint_evals import eligible, summarize, submission_env


@pytest.fixture
def series(tmp_path):
    harnesses = ['opencode', 'claude-code', 'codex', 'mini-swe-agent']
    protocol = {'harnesses': harnesses, 'model': 'base',
                'harness_versions': {h: '1.2.3' for h in harnesses},
                'scores': {h: {'pass_at_1': 0.0} for h in harnesses}}
    tasks = [{'difficulty': 'easy' if i < 33 else 'medium' if i < 151 else 'hard'}
             for i in range(250)]
    (tmp_path / 'manifest.json').write_text(json.dumps({'tasks': tasks}))
    logs = tmp_path / 'job-1'
    (logs / 'traces').mkdir(parents=True)
    rows = []
    for h in harnesses:
        for i in range(250):
            trial = f'{h}-{i}'
            rows.append({'harness': h, 'index': i, 'rep': 0, 'reward': 0,
                         'n_turns': 1, 'trial_name': trial})
            native = logs / 'trials' / trial / 'result.json'
            native.parent.mkdir(parents=True)
            native.write_text(json.dumps({'agent_info': {'version': '1.2.3'}}))
    (logs / 'final_tito.json').write_text('{"tito_pass":true}')
    return tmp_path, logs, rows, protocol


def test_first_graded_zero_is_retained_when_later_retry_succeeds(series):
    output, logs, rows, protocol = series
    rows.append({**rows[0], 'reward': 1})
    (logs / 'traces/all.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = summarize(output, 1, protocol)
    assert result['complete'] and result['comparison_ready']
    assert result['average_pass_at_1'] == 0
    scores = result['harnesses']['opencode']
    assert scores['graded'] == 250
    assert {d: v['graded'] for d, v in scores['difficulty'].items()} == {'easy': 33, 'medium': 118, 'hard': 99}


def test_missing_grade_does_not_publish_full_set_score(series):
    output, logs, rows, protocol = series
    rows[0]['reward'] = None
    (logs / 'traces/all.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = summarize(output, 1, protocol)
    assert result == {'complete': False, 'graded_cells': 999, 'expected_cells': 1000}
    assert not (output / 'scores.json').exists()


def test_changed_harness_version_prevents_comparable_result(series):
    output, logs, rows, protocol = series
    (logs / 'traces/all.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (logs / 'trials/opencode-0/result.json').write_text('{"agent_info":{"version":"9.9.9"}}')
    result = summarize(output, 1, protocol)
    assert result['complete'] and result['tito_pass']
    assert not result['comparison_ready']


def test_recovered_grades_use_original_measured_versions(series):
    output, logs, rows, protocol = series
    original = output / 'job-0'
    original.mkdir()
    (logs / 'trials').rename(original / 'trials')
    (logs / 'resume_transport_migration.json').write_text(json.dumps({'source': str(original)}))
    (logs / 'traces/all.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = summarize(output, 1, protocol)
    assert result['comparison_ready'] and result['average_pass_at_1'] == 0
    (original / 'trials/opencode-0/result.json').write_text('{"agent_info":{"version":"9.9.9"}}')
    assert not summarize(output, 1, protocol)['comparison_ready']


def test_version_ancestry_cycles_and_other_checkpoints_fail_closed(tmp_path):
    from checkpoint_evals import trial_result_path
    first, second = tmp_path / 'step-100/job-1', tmp_path / 'step-100/job-2'
    first.mkdir(parents=True)
    second.mkdir()
    for current, source in [(first, second), (second, first)]:
        (current / 'resume_transport_migration.json').write_text(json.dumps({'source': str(source)}))
    assert trial_result_path(first, 'trial') is None
    other = tmp_path / 'step-200/job-3'
    native = other / 'trials/trial/result.json'
    native.parent.mkdir(parents=True)
    native.write_text('{"agent_info":{"version":"1.2.3"}}')
    (second / 'resume_transport_migration.json').write_text(json.dumps({'source': str(other)}))
    assert trial_result_path(first, 'trial') is None
    assert trial_result_path(first, '../trial') is None


def test_only_completed_interval_checkpoints_are_eligible(tmp_path):
    root = tmp_path / 'checkpoint-50'
    root.mkdir()
    assert eligible(root, 50) is None
    marker = {'checkpoint': str(root.resolve()), 'step': 50}
    (root / 'checkpoint.ready.json').write_text(json.dumps(marker))
    assert eligible(root, 50) == marker
    assert eligible(root, 100) is None
    marker['final'] = True
    (root / 'checkpoint.ready.json').write_text(json.dumps(marker))
    assert eligible(root, 100) is None
    assert eligible(root, 100, include_final=True) == marker
    marker['step'] = 100
    (root / 'checkpoint.ready.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='completion marker'):
        eligible(root, 50)


def test_checkpoint_eval_cannot_inherit_baseline_resume(monkeypatch, tmp_path):
    monkeypatch.setenv('BASELINE_RESUME_FROM', '/wrong/base/results')
    monkeypatch.setenv('EVAL_SMOKE_ONLY', '1')
    env = submission_env(tmp_path, {'model': 'base', 'harness_versions': {}})
    assert 'BASELINE_RESUME_FROM' not in env
    assert 'EVAL_SMOKE_ONLY' not in env
    assert env['EVAL_MODEL_SOURCE'] == str(tmp_path / 'model')
    assert env['EVAL_CODE_ROOT'] == str(tmp_path / 'source-snapshot')


@pytest.mark.parametrize('step,expected', [(50, False), (100, True), (150, False), (200, True)])
def test_saved_handoff_obeys_100_step_eval_interval(tmp_path, step, expected):
    root = tmp_path / f'checkpoint-{step}'
    root.mkdir()
    (root / 'checkpoint.saved.json').write_text(json.dumps({'step': step, 'checkpoint': str(root)}))
    assert bool(eligible(root, 100)) == expected


def test_gpu_eval_excludes_training_node_and_has_independent_cleanup(monkeypatch, tmp_path):
    from checkpoint_evals import submit
    commands = []
    def capture(command, **kwargs):
        commands.append(command)
        return '123\n' if len(commands) == 1 else '124\n'
    monkeypatch.setattr('checkpoint_evals.subprocess.check_output', capture)
    result = submit(tmp_path, {'model': 'base', 'harness_versions': {}}, 'hopper-extra', exclude_nodes='training-node')
    assert result == {'job_id': '123', 'cleanup_job_id': '124'}
    assert '--exclude=training-node' in commands[0]
    assert '--gres=gpu:2' in commands[0]
    assert '--dependency=afterany:123' in commands[1]
