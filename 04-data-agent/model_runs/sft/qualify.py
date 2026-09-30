"""Reconcile preparation, optimizer, checkpoint, longest-row, and eval receipts."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from prepare import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--smoke', type=Path, required=True)
    args = p.parse_args()
    root, exp, smoke = args.workspace.resolve(), args.experiment.resolve(), args.smoke.resolve()
    code = Path(__file__).resolve().parent
    read = lambda p: json.loads(p.read_text())
    data = read(exp / 'data/preparation.json')
    first, resumed = [read(smoke / f'training_verified_step_{s}.json') for s in [2,4]]
    assert first['step'] == 2 and resumed['step'] == 4 and resumed['checkpoint_resume']
    events = [json.loads(line) for line in (smoke / 'metrics.jsonl').read_text().splitlines()]
    updates = [r for r in events if 'grad_norm' in r]
    assert [r['step'] for r in updates] == [1,2,3,4]
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) and r['grad_norm'] > 0 for r in updates)
    assert first['native_trl_label_equality'] and resumed['native_trl_label_equality']
    markers = [read(smoke / f'run/checkpoint-{s}/checkpoint.ready.json') for s in [2,4]]
    assert markers[0]['files']['model.safetensors'] != markers[1]['files']['model.safetensors']
    for s in [2,4]:
        checkpoint = smoke / f'run/checkpoint-{s}'
        state = read(checkpoint / 'trainer_state.json')
        assert state['global_step'] == s
        assert all((checkpoint / name).exists() for name in ['optimizer.pt','scheduler.pt','rng_state.pth'])
        assert read(smoke / f'eval_requests/checkpoint-{s}.json')['step'] == s
    long = read(exp / 'longest-probe/longest_verified.json')
    assert long['task']['tokens'] == int(data['lengths']['100'])
    assert long['optimizer_update']['grad_norm'] > 0
    evaluation = read(smoke / 'eval/evaluation_verified.json')
    assert evaluation['score']['complete'] and evaluation['heldout_tasks'] == 2
    services = read(smoke / 'eval/services.json')
    assert services['tito_probe'] and services['capture_instance_verified']
    raw_test = read(Path(data['test_manifest']))['tasks']
    frozen = root / 'experiments/lfm25-multi4-20260921/runtime/data/test/tasks'
    assert {x.name for x in frozen.iterdir() if x.is_dir()} == {t['name'] for t in raw_test}
    local_logging = read(exp / 'local_logging_verified.json')
    assert local_logging['passed'] and local_logging['remote_requests'] == 0
    assert local_logging['training_steps_read_back'] == [1, 2, 3, 4]
    assert local_logging['eval_graded'] == 8
    epoch_policy = read(exp / 'epoch_eval_validation.json')
    assert epoch_policy['passed'] and epoch_policy['native_trainer']
    assert epoch_policy['eval_requests'] == [[3, 1], [6, 2]]
    assert epoch_policy['environment_capacity'] == 128
    # Exercise submission construction only; do not start full benchmark jobs.
    preview = subprocess.check_output([sys.executable, str(code / 'watch_evals.py'), '--workspace', str(root),
                '--run-dir', str(smoke), '--training-job', '86335'], text=True)
    assert '--gres=gpu:1' in preview and '--concurrency 100' in preview
    (exp / 'eval_submission_preview.txt').write_text(preview)
    proof = {'passed': True, 'model': data['model'], 'model_revision': data['model_revision'],
             'dataset_revision': data['dataset_revision'], 'matched_tasks': data['matched_sft_tasks'],
             'train_sha256': digest(exp / 'data/train.jsonl'), 'test_manifest_sha256': data['test_manifest_sha256'],
             'nonzero_updates': 4, 'checkpoint_resume': True,
             'epoch_eval_validation': epoch_policy, 'local_logging_validation': local_logging,
             'compatibility_evidence': 'GPU smoke from the prior step-based schedule; epoch policy verified separately on CPU.', 'checkpoint_weights_changed_between_2_and_4': True,
             'assistant_masks_verified': True, 'longest_tokens': long['task']['tokens'],
             'longest_max_memory_allocated_gib': long['max_memory_allocated_gib'],
             'save_and_eval_request_callback': True, 'eval_submission_dry_run': True,
             'reloaded_checkpoint_eval': evaluation, 'production_launched': False,
             'smoke_output': str(smoke), 'scripts_sha256': {f.name:digest(f) for f in sorted(code.iterdir()) if f.suffix in ['.py','.sh']},
             'limits': ['8 rollout functional smoke, not a quality estimate or throughput qualification at concurrency100.',
                        'Full production submission and the long-running eval watcher have not been exercised.']}
    (exp / 'qualification.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
