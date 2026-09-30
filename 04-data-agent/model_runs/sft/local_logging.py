"""Local Trackio configuration and checkpoint-evaluation scalar logging."""
import json
import os
from pathlib import Path
from statistics import mean


def configure(run_root):
    os.environ['TRACKIO_DIR'] = str(Path(run_root).resolve() / 'trackio')
    os.environ['TRACKIO_STORAGE_MODE'] = 'jsonl'
    for key in ['TRACKIO_SPACE_ID', 'TRACKIO_SERVER_URL', 'TRACKIO_BUCKET_ID', 'TRACKIO_DATASET_ID']:
        os.environ.pop(key, None)


def materialize():
    from trackio import fragments
    inbox = Path(os.environ['TRACKIO_DIR']) / 'inbox'
    count = 0
    for source in sorted(inbox.rglob('*.jsonl')):
        records = fragments.parse_fragment_bytes(source.read_bytes())
        if records:
            count += fragments.import_records(records)
    return count


def eval_metrics(output):
    output = Path(output)
    record = json.loads((output / 'evaluation_verified.json').read_text())
    score = record['score']
    metrics = {'eval/pass_at_1': score['average_pass_at_1'],
               'eval/graded': score['graded_cells'], 'eval/expected': score['expected_cells'],
               'eval/complete': int(score['complete']), 'eval/tito_pass': int(score['tito_pass'])}
    if 'training_test_overlap' in record:
        metrics['eval/strictly_held_out'] = int(record['strictly_held_out'])
        for key, count in record['training_test_overlap'].items():
            metrics['eval/' + key] = count
    for harness, values in score['harnesses'].items():
        metrics[f'eval/{harness}/pass_at_1'] = values['pass_at_1']
        metrics[f'eval/{harness}/graded'] = values['evaluations']
        for tier, counts in values['difficulty'].items():
            if counts['pass_at_1'] is not None:
                metrics[f'eval/{harness}/{tier}/pass_at_1'] = counts['pass_at_1']
                metrics[f'eval/{harness}/{tier}/graded'] = counts['evaluations']
    cells = {}
    for trace in sorted((output / 'traces').glob('*.jsonl')):
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            key = row['harness'], row['index'], row['rep']
            if row.get('ok') and row.get('reward') is not None and key not in cells:
                cells[key] = row
    assert len(cells) == score['graded_cells']
    for harness in score['harnesses']:
        rows = [r for r in cells.values() if r['harness'] == harness]
        tokens_in, tokens_out, calls = [], [], []
        for row in rows:
            capture = json.loads(Path(row['capture_file']).read_text())
            turns = capture['turns']
            if all(t.get('prompt_token_ids') is not None and t.get('completion_token_ids') is not None for t in turns):
                tokens_in.append(sum(len(t['prompt_token_ids']) for t in turns))
                tokens_out.append(sum(len(t['completion_token_ids']) for t in turns))
            trajectory = output / 'trials' / row['trial_name'] / 'agent/trajectory.json'
            if trajectory.exists():
                native = json.loads(trajectory.read_text())
                if str(native.get('schema_version', '')).startswith('ATIF-'):
                    ids = [call.get('tool_call_id') for step in native['steps'] if step.get('source') == 'agent'
                           for call in step.get('tool_calls', []) or []]
                    if all(ids) and len(set(ids)) == len(ids):
                        calls.append(len(ids))
        prefix = f'eval/{harness}/'
        metrics[prefix + 'token_coverage'] = len(tokens_in)
        metrics[prefix + 'native_tool_count_coverage'] = len(calls)
        if rows:
            metrics[prefix + 'turns_mean'] = mean(r['n_turns'] for r in rows)
            metrics[prefix + 'wall_seconds_mean'] = mean(r['wall_s'] for r in rows)
        if tokens_in:
            metrics[prefix + 'input_tokens_mean'] = mean(tokens_in)
            metrics[prefix + 'generated_tokens_mean'] = mean(tokens_out)
        if calls:
            metrics[prefix + 'native_tool_calls_mean'] = mean(calls)
    return record, metrics


def log_evaluation(output, run_root):
    configure(run_root)
    import trackio
    materialize()
    record, metrics = eval_metrics(output)
    config = json.loads((Path(run_root) / 'config.json').read_text())['configuration']
    state = json.loads((Path(record['checkpoint']) / 'trainer_state.json').read_text())
    metrics['eval/epoch'] = state['epoch']
    receipt = Path(output) / 'trackio_local.json'
    payload = {'project': config['project'], 'run': config['run_name'] + ' · evaluations',
               'step': state['global_step'], 'metrics': metrics, 'storage': str(Path(run_root).resolve() / 'trackio'),
               'remote_upload': False, 'definition': 'Token means include all captured turns; native calls require unique ATIF action IDs. Missing counters are not zero-filled.'}
    if receipt.exists() and json.loads(receipt.read_text()) == payload:
        return payload
    trackio.init(project=payload['project'], name=payload['run'], space_id=None, resume='allow',
                 auto_log_gpu=False, auto_log_cpu=False)
    trackio.log(metrics, step=payload['step'])
    trackio.finish()
    materialize()
    receipt.write_text(json.dumps(payload, indent=2) + '\n')
    return payload


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(log_evaluation(args.output, args.run_root), indent=2))
