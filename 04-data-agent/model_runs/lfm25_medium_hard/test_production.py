import importlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest
from production import committed_state, check_qualification
from launch import commands

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3] / 'experiments/lfm25-medium-hard1000-20260921'


def test_existing_resume_recovers_cancelled_group():
    state = json.loads((ROOT / 'opencode/train-smoke-83796/run/checkpoint-2/curriculum_state.json').read_text())
    fixed = committed_state(state)
    assert 3 in state['settled_groups'] and 3 not in fixed['settled_groups']
    assert 0 in fixed['settled_groups']
    assert fixed['admitted_rollouts'] == state['admitted_rollouts']
    assert committed_state(fixed)['settled_groups'] == fixed['settled_groups']


def test_matching_smokes_and_launch_isolation():
    for arm in ['opencode', 'multi-harness']:
        assert check_qualification(ROOT / arm, HERE)
        train, watch = commands(arm, '123')
        assert '--gres=gpu:2' in train and '--partition=hopper-prod' in train
        assert '--gres=gpu:0' in watch and '--dependency=after:123' in watch
        assert train[-2:] == [arm, 'train']


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run.py', 'opencode'])
    runner = importlib.import_module('run')
    return runner, importlib.import_module('controller')


def test_production_command_and_early_exhaustion(runtime, monkeypatch, tmp_path):
    runner, controller = runtime
    import checkpoint_artifacts
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1')
    monkeypatch.delenv('LFM_REMOTE_SERVICE', raising=False)
    monkeypatch.delenv('LFM_RESUME_CHECKPOINT', raising=False)
    captured = []
    def start(cmd, log, env):
        captured.append((list(map(str, cmd)), env))
        (tmp_path / 'run/checkpoint-37').mkdir(parents=True)
        return Mock(wait=lambda: 0)
    monkeypatch.setattr(runner.base.rt, 'start', start)
    monkeypatch.setattr(checkpoint_artifacts, 'finalize_saved', lambda p: None)
    monkeypatch.setattr(checkpoint_artifacts, 'verify_ready', lambda p: None)
    checkpoint = runner.train(tmp_path, 'http://engine', 'http://env', False)
    cmd, env = captured[0]
    assert checkpoint.name == 'checkpoint-37'
    assert cmd[cmd.index('--save-steps') + 1] == '50'
    assert cmd[cmd.index('--max-steps') + 1] == '1000'
    assert cmd[cmd.index('--harnesses') + 1] == 'opencode'
    assert env['CURRICULUM_GROUP_LIMIT'] == '2000'
    assert env['LFM_TRIALS_DIR'] == str(tmp_path / 'trials')


def test_controller_finalizes_and_triggers_once(runtime, monkeypatch, tmp_path):
    runner, controller = runtime
    from checkpoint_artifacts import REQUIRED, mark_saved
    from safetensors.numpy import save_file
    import numpy as np
    monkeypatch.setattr(controller, 'OUT', tmp_path)
    monkeypatch.setattr(controller, 'sync', lambda *args: None)
    monkeypatch.setattr(controller.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(sys, 'argv', ['controller.py', '--training-job', '123'])
    for step in [50, 100, 200, 250]:
        cp = tmp_path / f'train-123/run/checkpoint-{step}'
        cp.mkdir(parents=True)
        for name in REQUIRED: (cp / name).write_text('{}')
        (cp / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
        save_file({'weight': np.ones(1, dtype=np.float32)}, cp / 'model.safetensors')
        mark_saved(cp, step, 'LiquidAI/LFM2.5-2.6B', runner.base.C['model_revision'])
    submitted = []
    queries = 0
    def command(args, **kwargs):
        nonlocal queries
        if args[0] == 'squeue':
            assert '-j' not in args
            queries += 1
            return '123|train-node\n' if queries == 1 else ''
        assert args[0] == 'sbatch'
        submitted.append(args)
        job = str(1000 + len(submitted))
        out = tmp_path / f'eval-{job}'
        out.mkdir()
        (out / 'canonical_scores.json').write_text(json.dumps({'complete':True,'comparison_ready':True,'graded_cells':1000}))
        return job
    monkeypatch.setattr(controller.subprocess, 'check_output', command)
    controller.main()
    assert [Path(c[-1]).name for c in submitted] == ['checkpoint-100', 'checkpoint-200', 'checkpoint-250']
    assert all('--exclude=train-node' in c and '--gres=gpu:2' in c for c in submitted)
    assert not (tmp_path / 'train-123/run/checkpoint-50/checkpoint.ready.json').exists()
    assert (tmp_path / 'train-123/run/checkpoint-100/checkpoint.ready.json').exists()
    controller.main()
    assert len(submitted) == 3


def test_spawned_worker_installs_resume_policy(runtime):
    import subprocess
    code = '''
import sys
sys.argv=['run.py','opencode']
import run
import efficiency
import hard_curriculum_train as finite
assert finite._lfm_resume_policy
state={'group_limit':4,'settled_groups':[0,3],'admitted_rollouts':{'committed':0}}
assert finite.pending_groups(4,state)==[1,2,3]
from production import install_resume_policy
install_resume_policy()
assert finite.pending_groups(4,state)==[1,2,3]
print('spawn import and pending-group checks passed')
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=HERE, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr[-3000:]


def test_controller_reports_failed_eval(runtime, monkeypatch, tmp_path):
    _, controller = runtime
    monkeypatch.setattr(controller, 'OUT', tmp_path)
    monkeypatch.setattr(controller, 'sync', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['controller.py', '--training-job', '123'])
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *args, **kwargs: '')
    (tmp_path / 'controller-123.json').write_text(json.dumps({'evaluations': {'100': {'job_id':'999'}}}))
    with pytest.raises(RuntimeError, match='without complete results'):
        controller.main()
    assert json.loads((tmp_path/'controller-123.json').read_text())['incomplete_evaluations']==['100']
