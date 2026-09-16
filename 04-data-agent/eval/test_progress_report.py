import json

from build_progress_report import evaluation_paths, matched_counts


def test_checkpoint_history_survives_allocation_move(tmp_path):
    parent, child = tmp_path / 'parent', tmp_path / 'child'
    parent.mkdir(); child.mkdir()
    (parent / 'run_config.json').write_text('{}')
    (child / 'run_config.json').write_text(json.dumps({'resume_state': {
        'step': 150, 'checkpoint': str(parent / 'job-1/run/checkpoint-150')}}))
    for root, steps in [(parent, [100, 200]), (child, [300])]:
        for step in steps:
            (root / f'checkpoint-evals/step-{step:06d}').mkdir(parents=True)
    assert sorted(evaluation_paths(child)) == [100, 300]


def test_partial_checkpoints_compare_the_intersection():
    selected = {'base': {1: {'reward': 0}, 2: {'reward': 1}, 3: {'reward': 0}},
                '100': {1: {'reward': 1}, 2: {'reward': 1}},
                '200': {1: {'reward': 0}, 3: {'reward': 1}}}
    result = matched_counts(selected)
    assert all(r['graded'] == 1 for r in result.values())
    assert [r['correct'] for r in result.values()] == [0, 1, 0]
