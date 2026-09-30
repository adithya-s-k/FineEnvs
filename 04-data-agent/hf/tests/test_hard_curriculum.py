import asyncio
from collections import defaultdict
import json
from pathlib import Path
import queue
import sys
from types import SimpleNamespace

import pytest

HF = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HF), str(HF / 'runtime'), str(HF.parent / 'train')]
from hard_curriculum import request, write
from hard_curriculum_train import FiniteLoop, pending_groups
from atomic_rollouts import AtomicRolloutDataset, AtomicRolloutTrainer, RolloutBundle, RolloutsFinished
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample


def bundle(name='one', group=0):
    return RolloutBundle([RolloutSample([], [], [1, 2], [0, 1], [-.2, -.3], .5, 1, group, {})], name)


def dataset(bundles):
    q = queue.Queue()
    for item in bundles:
        q.put(item)
    worker = SimpleNamespace(rollout_buffer=q, model_version=1,
                             check_health=lambda _: pytest.fail('Finite stream hung'))
    return AtomicRolloutDataset(worker, defaultdict(list), 10, 100, 4, 60, max_rollouts_per_unit=2)


def test_partial_packing_unit_is_drained():
    sample = bundle()
    data = dataset([sample, RolloutsFinished()])
    assert list(data) == [{'rollouts': [sample]}]
    assert data.exhausted
    assert list(data) == []


def test_last_partial_optimizer_update_keeps_token_normalization():
    data = dataset([bundle(str(i)) for i in range(5)] + [RolloutsFinished()])
    batches, _ = AtomicRolloutTrainer.get_batch_samples(None, iter(data), 4, None)
    assert [len(b['rollouts']) for b in batches] == [2, 2, 1, 0]
    assert all(b['normalization_tokens'] == 5 for b in batches)


def test_all_empty_stream_ends_without_fake_training_data():
    batches, _ = AtomicRolloutTrainer.get_batch_samples(None, iter(dataset([RolloutsFinished()])), 4, None)
    assert batches == []


def test_resume_does_not_replay_committed_groups_after_a_hole():
    assert pending_groups(5, {'group_limit':5, 'settled_groups':[0, 2, 4]}) == [1, 3]
    with pytest.raises(ValueError):
        pending_groups(5, {'group_limit':6})
    with pytest.raises(ValueError):
        pending_groups(5, {'settled_groups':[5]})


def test_generation_dispatches_only_requested_groups_and_drains():
    async def check():
        loop = object.__new__(FiniteLoop)
        loop.curriculum_limit = 5
        loop.curriculum_state = {'settled_groups':[1, 3]}
        loop.dataset = [{'prompt':[{'role':'user','content':str(g)}]} for g in range(5)]
        loop.num_generations = 8
        loop.max_inflight_tasks = 16
        loop._model_version_value = SimpleNamespace(value=1)
        loop._heartbeat_value = SimpleNamespace(value=0)
        loop._groups_to_score = asyncio.Queue()
        calls = []
        async def generate(prompt, tools, tool_list, group_id):
            await asyncio.sleep(.001 if group_id == 0 else 0)
            calls.append(group_id)
            return [], [], [], 0, 0, None
        loop._generate_one = generate
        await loop._generate_loop(asyncio.Event())
        assert sorted(calls) == [g for g in [0,2,4] for _ in range(8)]
        groups = [await loop._groups_to_score.get() for _ in range(4)]
        assert groups[-1] is None
        assert {g.group_id for g in groups[:-1]} == {0,2,4}
        assert all(len(g.prompts) == 8 for g in groups[:-1])
    asyncio.run(check())


def test_preview_never_contains_secret_values_or_submits(tmp_path):
    write(tmp_path/'bundle.json', {'sha256':'bundle'})
    write(tmp_path/'config.json', {'hf':{'namespace':'org','image':'pinned','run_id':'hard500',
        'artifact_bucket':'org/artifacts','bundle_repo':'org/repro'},
        'resources':{'flavor':'h200x2'},'logging':{'space_id':'org/dashboard'}})
    req = request(tmp_path, 'long')
    assert req['timeout'] == '72h'
    assert req['env']['BUNDLE_REVISION'] == 'UPLOAD_REQUIRED'
    assert 'secrets' not in req
    assert req['flavor'] == 'h200x2'


def test_entrypoint_uses_frozen_trainer_main_signature(monkeypatch):
    import hard_curriculum_train as runtime
    calls = []
    monkeypatch.setattr(sys, 'argv', ['train', '--help'])
    monkeypatch.setitem(sys.modules, 'train_harbor_multi', SimpleNamespace(main=lambda: calls.append(True)))
    monkeypatch.setattr(runtime.FiniteTrainer, 'train', runtime.FiniteTrainer.train)
    monkeypatch.setattr(runtime.atomic, 'AtomicHarnessWorker', runtime.atomic.AtomicHarnessWorker)
    monkeypatch.setattr(runtime.atomic, 'AtomicRolloutTrainer', runtime.atomic.AtomicRolloutTrainer)
    runtime.main()
    assert calls == [True]


def test_finite_marker_is_sent_after_scoring_without_waiting_for_shutdown(monkeypatch):
    import hard_curriculum_train as runtime
    async def check():
        loop = object.__new__(FiniteLoop)
        event = asyncio.Event()
        markers = []
        async def scored(self, stop_event):
            assert not stop_event.is_set()
        def put(marker):
            markers.append(marker)
            event.set()
        monkeypatch.setattr(runtime.atomic.AtomicHarnessLoop, '_score_loop', scored)
        loop.rollout_buffer = SimpleNamespace(put_nowait=put)
        await asyncio.wait_for(loop._score_loop(event), timeout=1)
        assert len(markers) == 1 and isinstance(markers[0], RolloutsFinished)
    asyncio.run(check())
