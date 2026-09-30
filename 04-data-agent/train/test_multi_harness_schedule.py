"""Guard complete task/harness dispatch and stopping at a saved optimizer boundary."""
from types import SimpleNamespace
from collections import Counter
from copy import deepcopy

import pytest

from multi_harness import pair_rows, MultiHarborSessionFactory
from training_audit import PairCoverageCallback, PeriodicCheckpointCallback, WallTimeCallback
from harness_schedule import make_schedule, validate_schedule


def test_balanced_rotation_routes_tasks_and_harnesses_across_four_passes(tmp_path):
    harnesses = ['opencode', 'claude-code', 'codex', 'mini-swe-agent']
    tiers = ['easy'] * 150 + ['medium'] * 600 + ['hard'] * 250
    tasks = [{'name': f'task-{i}', 'task_index': 999 - i, 'difficulty': tier}
             for i, tier in enumerate(tiers)]
    schedule = make_schedule(tasks, harnesses)
    assert schedule == make_schedule(tasks, harnesses)
    factory = SimpleNamespace(harnesses=harnesses, schedule=schedule,
        prompt_rows=lambda: [{'task_name': t['name'], 'task_index': t['task_index']} for t in tasks])
    rows = pair_rows(factory)
    assert len(rows) == 4000
    from itertools import islice
    from trl.experimental.async_grpo.async_rollout_worker import _AsyncRolloutLoop
    worker = SimpleNamespace(dataset=rows, _dataset_iter=iter(rows), num_generations=4)
    dispatched = list(islice(_AsyncRolloutLoop._repeat_iterator(worker), 16004))
    for group_id, row in dispatched:
        expected = schedule['groups'][group_id % 4000]
        assert row['task_index'] == expected['task_index']
        assert MultiHarborSessionFactory.harness_for(factory, group_id) == expected['harness']
    for p in range(4):
        groups = schedule['groups'][p * 1000:(p + 1) * 1000]
        assert len({g['task_name'] for g in groups}) == 1000
        assert Counter(g['harness'] for g in groups) == {h: 250 for h in harnesses}
        assert [g['harness'] for g in groups] == harnesses * 250
        for g in groups:
            for _ in range(4):  # GRPO generations share their group ID and route.
                assert MultiHarborSessionFactory.harness_for(factory, g['group_in_cycle']) == g['harness']
                assert rows[g['group_in_cycle']]['task_index'] == g['task_index']
    assert [g['task_row'] for g in schedule['groups'][:32]] == list(range(32))
    assert len({(g['task_name'], g['harness']) for g in schedule['groups']}) == 4000
    trainer = SimpleNamespace(_trained_groups=set(range(1000)))
    callback = PairCoverageCallback(trainer, len(rows), harnesses, 0, tmp_path, schedule=schedule)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_step_end(SimpleNamespace(max_steps=1000), SimpleNamespace(global_step=200), control)
    callback.on_step_end(SimpleNamespace(max_steps=1000), SimpleNamespace(global_step=201), control)
    import json
    coverage = json.loads((tmp_path / 'coverage.json').read_text())
    assert coverage['unique_tasks_covered'] == 1000
    assert not coverage['pair_coverage_complete']
    assert coverage['harness_pair_counts'] == {h: 250 for h in harnesses}
    with pytest.raises(ValueError, match='Cartesian'):
        pair_rows(factory, all_pairs=True)
    factory.prompt_rows = lambda: list(reversed(rows[:1000]))
    with pytest.raises(ValueError, match='Server task identities'):
        pair_rows(factory)
    changed = deepcopy(schedule)
    changed['task_count'] = 4000
    with pytest.raises(ValueError, match='metadata'):
        validate_schedule(changed)


def test_cartesian_dispatch_reaches_each_pair_once_per_cycle(tmp_path):
    harnesses = ['opencode', 'claude-code', 'codex', 'mini-swe-agent']
    factory = SimpleNamespace(harnesses=harnesses, prompt_rows=lambda: [{'task': i} for i in range(5)])
    rows = pair_rows(factory, all_pairs=True)
    assert len(rows) == 20
    for cycle in range(2):
        pairs = [(rows[g % 20]['task'], MultiHarborSessionFactory.harness_for(factory, g))
                 for g in range(cycle * 20, (cycle + 1) * 20)]
        assert len(set(pairs)) == 20
        assert set(pairs) == {(i, h) for i in range(5) for h in harnesses}
    trainer = SimpleNamespace(_trained_groups=set(range(20)))
    callback = PairCoverageCallback(trainer, len(rows), harnesses, 20, tmp_path, all_pairs=True)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_step_end(SimpleNamespace(max_steps=40), SimpleNamespace(global_step=20), control)
    assert not control.should_training_stop  # a prefetched row is not proof it was trained
    callback.on_step_end(SimpleNamespace(max_steps=40), SimpleNamespace(global_step=21), control)
    assert control.should_training_stop


def test_wall_time_stop_requests_a_checkpoint(monkeypatch, tmp_path):
    clock = [0]
    monkeypatch.setattr('training_audit.time.monotonic', lambda: clock[0])
    callback = WallTimeCallback(60)
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    args = SimpleNamespace(output_dir=str(tmp_path / 'run'))
    callback.on_train_begin(None, None, control)
    clock[0] = 59
    callback.on_step_end(args, None, control)
    assert not control.should_training_stop
    clock[0] = 60
    callback.on_step_end(args, None, control)
    assert control.should_training_stop and control.should_save


def test_resume_keeps_task_and_harness_aligned_across_schedule_wrap(tmp_path):
    from itertools import islice
    import json
    from trl.experimental.async_grpo.async_rollout_worker import _AsyncRolloutLoop
    harnesses = ['opencode', 'claude-code', 'codex', 'mini-swe-agent']
    tasks = [{'name': f'task-{i}', 'task_index': i, 'difficulty': 'easy'} for i in range(1000)]
    schedule = make_schedule(tasks, harnesses)
    factory = SimpleNamespace(harnesses=harnesses, schedule=schedule, group_offset=3997,
        prompt_rows=lambda: [{'task_name': t['name'], 'task_index': t['task_index']} for t in tasks])
    rows = pair_rows(factory)
    worker = SimpleNamespace(dataset=rows, _dataset_iter=iter(rows[3997:]), num_generations=8)
    for local_group, row in islice(_AsyncRolloutLoop._repeat_iterator(worker), 64):
        expected = schedule['groups'][(3997 + local_group) % 4000]
        assert row['task_index'] == expected['task_index']
        assert MultiHarborSessionFactory.harness_for(factory, local_group) == expected['harness']
    callback = PairCoverageCallback(SimpleNamespace(_trained_groups={0, 1, 2, 3}), len(rows),
        harnesses, 0, tmp_path, schedule=schedule, group_offset=3997)
    for step in [51, 52]:
        callback.on_step_end(SimpleNamespace(max_steps=1000), SimpleNamespace(global_step=step),
                             SimpleNamespace(should_training_stop=False))
    coverage = json.loads((tmp_path / 'coverage.json').read_text())
    assert coverage['collated_group_ids'] == [3997, 3998, 3999, 4000]
    expected = {(schedule['groups'][g % 4000]['task_row'], schedule['groups'][g % 4000]['harness'])
                for g in coverage['collated_group_ids']}
    assert {(p['task_row'], p['harness']) for p in coverage['covered_pairs']} == expected


def test_operator_stop_saves_at_optimizer_boundary(tmp_path):
    callback = WallTimeCallback(82800)
    args = SimpleNamespace(output_dir=str(tmp_path / 'run'))
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    callback.on_step_end(args, None, control)
    assert not control.should_save
    (tmp_path / 'STOP_AFTER_STEP').touch()
    callback.on_step_end(args, None, control)
    assert control.should_training_stop and control.should_save


def test_hourly_checkpoint_saves_without_stopping_and_regular_save_resets_clock(monkeypatch):
    clock = [0]
    monkeypatch.setattr('training_audit.time.monotonic', lambda: clock[0])
    callback = PeriodicCheckpointCallback(3600)
    control = SimpleNamespace(should_training_stop=False, should_save=False)
    callback.on_train_begin(None, None, control)
    clock[0] = 3599
    callback.on_step_end(None, None, control)
    assert not control.should_save
    clock[0] = 3700  # saves only after a complete optimizer update
    callback.on_step_end(None, None, control)
    assert control.should_save and not control.should_training_stop
    callback.on_save(None, None, control)
    control.should_save = False
    clock[0] = 4000
    callback.on_save(None, None, control)  # regular checkpoint-50 resets the same timer
    clock[0] = 7300
    callback.on_step_end(None, None, control)
    assert not control.should_save
    clock[0] = 7600
    callback.on_step_end(None, None, control)
    assert control.should_save and not control.should_training_stop


@pytest.mark.parametrize('seconds', [0, -1])
def test_hourly_checkpoint_requires_positive_interval(seconds):
    with pytest.raises(ValueError, match='positive'):
        PeriodicCheckpointCallback(seconds)
