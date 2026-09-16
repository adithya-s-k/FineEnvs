"""Portable packaging and shared-service isolation at the actual rollout boundary."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import build
import harbor_service
from service_policy import Admission


def test_harbor_policy_is_per_session_and_releases_failed_rollouts(monkeypatch):
    from openenv.harbor import rollout, serving
    gate = Admission(4, 1)
    sessions = {}
    observed = []
    class Registry:
        def get(self, key): return sessions.get(key)
    async def native(**kwargs):
        key = kwargs['dataset']
        sessions[key] = SimpleNamespace(metadata={})
        kwargs['on_session_created'](key)
        await asyncio.sleep(.01)
        if key == 'test': raise RuntimeError('trial failure')
        return sessions[key].metadata['max_output_tokens']
    monkeypatch.setattr(rollout, 'run_rollout', native)
    monkeypatch.setattr(serving, 'space_public_url', lambda: 'https://example.hf.space')
    monkeypatch.setattr(harbor_service, 'admission', gate)
    harbor_service.install()
    async def exercise():
        return await asyncio.gather(*(rollout.run_rollout(dataset=d, registry=Registry(),
            on_session_created=observed.append) for d in ('train', 'test')), return_exceptions=True)
    values = asyncio.run(exercise())
    assert values[0] == 16384 and isinstance(values[1], RuntimeError)
    assert sessions['test'].metadata['max_output_tokens'] == 4096
    assert set(observed) == {'train', 'test'}
    assert gate.snapshot()['active'] == {'train': 0, 'eval': 0}


def test_replaced_outputs_are_archived_without_data_loss(tmp_path):
    old = tmp_path / 'stage'
    old.mkdir()
    (old / 'capture.json').write_text('original')
    archive = tmp_path / 'archive'
    build.preserve(old, archive)
    assert not old.exists()
    assert next(archive.glob('stage-*/capture.json')).read_text() == 'original'


def test_packaging_excludes_local_credentials_and_caches(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / '.env').write_text('secret')
    (source / 'app.py').write_text('print(1)')
    (source / 'temp').mkdir()
    (source / 'temp' / 'token').write_text('secret')
    target = tmp_path / 'packed'
    build._copy(source, target)
    assert sorted(p.name for p in target.iterdir()) == ['app.py']


def test_hub_eval_capacity_is_bounded_and_source_pins_are_immutable():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'configs/deployment.json').read_text())
    assert set(config['evaluation']['concurrency_per_arm'].values()) == {35}
    sources = json.loads((root / 'configs/sources.json').read_text())
    for source in sources['repositories']:
        assert len(source['revision']) == 40
        assert set(source['revision']) <= set('0123456789abcdef')
    assert len(sources['task_bundle']['sha256']) == 64


def test_training_preflight_rejects_legacy_server_missing_sampling():
    from service_contract import validate_tools
    properties = {key: {} for key in ("llm_url", "model", "require_tokens", "agent_timeout_s")}
    response = {"data": {"observation": {"tools": [{"name": "run_rollout", "input_schema": {"properties": properties}}]}}}
    with pytest.raises(ValueError, match="lacks training arguments: sampling"):
        validate_tools(response, "opencode")
    properties["sampling"] = {}
    assert validate_tools(response, "opencode")["passed"]
