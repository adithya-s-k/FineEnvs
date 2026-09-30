"""A partial or changed checkpoint must never be evaluated as completed weights."""
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from checkpoint_artifacts import REQUIRED, mark_ready, mark_saved, finalize_saved, stage_model, verify_stage, resume_info


def test_trainer_handoff_does_not_hash_weights_and_cpu_finalizes(tmp_path, monkeypatch):
    root = checkpoint(tmp_path / 'checkpoint-50', sharded=True)
    with monkeypatch.context() as patch:
        patch.setattr('checkpoint_artifacts.digest', lambda _: pytest.fail('Weight hashing ran on trainer'))
        mark_saved(root, 50, 'base', 'revision')
    assert (root / 'checkpoint.saved.json').exists()
    assert not (root / 'checkpoint.ready.json').exists()
    assert finalize_saved(root)['step'] == 50


def test_cpu_rejects_checkpoint_modified_after_handoff(tmp_path):
    root = checkpoint(tmp_path / 'checkpoint-50')
    mark_saved(root, 50, 'base', 'revision')
    (root / 'optimizer.pt').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed after save'):
        finalize_saved(root)


def checkpoint(root, *, sharded=False):
    root.mkdir()
    for name in REQUIRED:
        (root / name).write_text('{}')
    (root / 'config.json').write_text('{"trained_config":true}')
    (root / 'trainer_state.json').write_text('{"global_step":50}')
    if sharded:
        save_file({'a': np.ones(2)}, root / 'model-1.safetensors')
        save_file({'b': np.ones(2)}, root / 'model-2.safetensors')
        (root / 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {'a': 'model-1.safetensors', 'b': 'model-2.safetensors'}}))
    else:
        save_file({'weight': np.ones(2)}, root / 'model.safetensors')
    return root


def test_missing_optimizer_state_cannot_publish(tmp_path):
    root = checkpoint(tmp_path / 'checkpoint-50')
    (root / 'optimizer.pt').unlink()
    with pytest.raises(ValueError, match='optimizer.pt'):
        mark_ready(root, 50, 'base', 'revision')
    assert not (root / 'checkpoint.ready.json').exists()


def test_missing_shard_cannot_publish(tmp_path):
    root = checkpoint(tmp_path / 'checkpoint-50', sharded=True)
    (root / 'model-2.safetensors').unlink()
    with pytest.raises(FileNotFoundError):
        mark_ready(root, 50, 'base', 'revision')
    assert not (root / 'checkpoint.ready.json').exists()


def test_stage_preserves_trained_config_and_detects_changed_weights(tmp_path):
    root = checkpoint(tmp_path / 'checkpoint-50', sharded=True)
    mark_ready(root, 50, 'base', 'revision')
    metadata = tmp_path / 'base-metadata'
    metadata.mkdir()
    (metadata / 'config.json').write_text('{"wrong_base_config":true}')
    (metadata / 'video_preprocessor_config.json').write_text('{}')
    target = tmp_path / 'staged'
    stage_model(root, target, metadata)
    assert json.loads((target / 'config.json').read_text()) == {'trained_config': True}
    assert (target / 'video_preprocessor_config.json').exists()
    assert (target / 'model-1.safetensors').is_symlink()
    assert verify_stage(target)['checkpoint']['step'] == 50
    save_file({'a': np.zeros(2)}, root / 'model-1.safetensors')
    with pytest.raises(ValueError, match='Staged checkpoint changed'):
        verify_stage(target)


def test_wrong_optimizer_step_cannot_publish(tmp_path):
    root = checkpoint(tmp_path / 'checkpoint-50')
    with pytest.raises(ValueError, match='trainer state'):
        mark_ready(root, 100, 'base', 'revision')


def test_resume_requires_complete_unchanged_optimizer_and_cursor(tmp_path):
    root = checkpoint(tmp_path / 'checkpoint-50')
    with pytest.raises(ValueError, match='completed checkpoint'):
        resume_info(root, 'base', 'revision')
    mark_saved(root, 50, 'base', 'revision')
    with pytest.raises(ValueError, match='rollout_state'):
        resume_info(root, 'base', 'revision')
    (root / 'rollout_state.json').write_text('{"prompt_index":37,"model_version":50}')
    info = resume_info(root, 'base', 'revision')
    assert info['step'] == 50 and info['group_offset'] == 37 and info['model_version'] == 50
    with pytest.raises(ValueError, match='model/revision'):
        resume_info(root, 'wrong-model', 'revision')
    (root / 'optimizer.pt').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed after save'):
        resume_info(root, 'base', 'revision')
