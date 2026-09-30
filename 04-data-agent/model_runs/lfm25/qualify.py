"""Verify persisted smoke evidence before enabling a long-run submission."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from run import C, HERE, rt, verify_ready


def qualify(output):
    recorded=rt.read(output/'configuration.json')
    for key in ['model','model_revision','dataset','harnesses','harness_versions','training','sampling','rollouts','serving']:
        if recorded[key]!=C[key]:raise ValueError(f'Smoke configuration differs: {key}')
    for step in (2,4):
        marker=verify_ready(output/f'run/checkpoint-{step}')
        if marker['step']!=step or marker['base_model']!=C['model']:raise ValueError('Wrong checkpoint')
    score=rt.read(output/'checkpoint-eval/eval_smoke_verified.json')
    if not score['complete'] or score['graded_cells']!=8 or not score['tito_pass']:
        raise ValueError('Checkpoint eval smoke incomplete')
    if str(output/'run/checkpoint-4') not in (output/'checkpoint-eval/vllm.log').read_text():
        raise ValueError('No evidence of checkpoint reload')
    rows=[json.loads(x) for x in (output/'audit/metrics.jsonl').read_text().splitlines()]
    updates=[r for r in rows if 'grad_norm' in r]
    if {r['step'] for r in updates}!={1,2,3,4}:raise ValueError('Missing optimizer updates')
    if not any(r['grad_norm']>0 for r in updates):raise ValueError('No fresh gradient')
    if any(not math.isfinite(r['grad_norm']) or not math.isfinite(r.get('loss',0)) for r in updates):
        raise ValueError('Nonfinite optimization')
    settled=set(rt.read(output/'run/checkpoint-2/curriculum_state.json')['settled_groups'])
    receipts=[json.loads(x) for x in (output/'audit/optimizer_rollouts.jsonl').read_text().splitlines()]
    resumed={r['group_id'] for entry in receipts if entry['step']>2 for r in entry['rollouts']}
    if not resumed or settled & resumed:raise ValueError('Resume group accounting failed')
    proof={'optimizer_save_resume':True,'reloaded_checkpoint_eval':True,'committed_group_replay':False,
        'model':C['model'],'revision':C['model_revision'],'nonzero_updates':sum(r['grad_norm']>0 for r in updates),
        'config_sha256':hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()}
    rt.write_json(output/'qualification_passed.json',proof)
    return proof

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    print(json.dumps(qualify(p.parse_args().output.resolve()),indent=2))
