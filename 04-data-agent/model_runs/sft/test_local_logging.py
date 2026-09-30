"""Read back real smoke metrics while forbidding remote logging calls."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

from local_logging import configure, log_evaluation, materialize

p = argparse.ArgumentParser()
p.add_argument('--run-root', type=Path, required=True)
p.add_argument('--receipt', type=Path, required=True)
p.add_argument('--steps', type=int, default=4)
args = p.parse_args()
os.environ['TRACKIO_SPACE_ID'] = 'must-not-be-used/test'
os.environ['TRACKIO_SERVER_URL'] = 'https://must-not-be-used.invalid'
configure(args.run_root)
assert not os.environ.get('TRACKIO_SPACE_ID') and not os.environ.get('TRACKIO_SERVER_URL')
with patch('httpx.Client.request', side_effect=AssertionError('Unexpected remote HTTP request')), \
     patch('requests.sessions.Session.request', side_effect=AssertionError('Unexpected remote HTTP request')):
    payload = log_evaluation(args.run_root / 'eval', args.run_root)
    materialize()
    from trackio.sqlite_storage import SQLiteStorage
    records = SQLiteStorage.get_run_records(payload['project'])
    train_steps = set()
    eval_rows = []
    for record in records:
        rows = SQLiteStorage.get_logs(payload['project'], run_id=record['id'])
        train_steps.update(int(r['train/global_step']) for r in rows if 'train/loss' in r)
        if record['name'] == payload['run']:
            eval_rows += [r for r in rows if r.get('step') == payload['step']]
    assert train_steps == set(range(1, args.steps + 1)), train_steps
    assert eval_rows
    assert all(all(row[k] == v for k, v in payload['metrics'].items()) for row in eval_rows)
    before_count = sum(SQLiteStorage.get_log_count(payload['project'], run_id=r['id']) for r in records)
    log_evaluation(args.run_root / 'eval', args.run_root)
    materialize()
    after_count = sum(SQLiteStorage.get_log_count(payload['project'], run_id=r['id']) for r in records)
    assert before_count == after_count
    assert all(payload['metrics'][f'eval/{h}/graded'] == 2 for h in ['opencode','claude-code','codex','mini-swe-agent'])
proof = {'passed': True, 'remote_requests': 0, 'training_steps_read_back': sorted(train_steps),
         'eval_step': payload['step'], 'repeated_logging_is_idempotent': True,
         'historical_smoke_eval_records': len(eval_rows), 'eval_metrics_read_back': len(payload['metrics']),
         'eval_graded': payload['metrics']['eval/graded'], 'storage': payload['storage'],
         'raw_jsonl_preserved': bool(list((args.run_root / 'trackio/inbox').rglob('*.jsonl')))}
args.receipt.write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof, indent=2))
