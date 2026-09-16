"""Require real optimizer, save/resume and token-provenance evidence before long runs."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]/'logs/20260915'
sys.path.insert(0,str(ROOT/'source/HuggingEnvs/04-data-agent/train'))
from checkpoint_artifacts import finalize_saved, verify_ready


def validate(arm,job):
    logs=ROOT/arm/f'training-smoke/job-{job}';run=logs/'run'
    assert (logs/'exit_code.txt').read_text().strip()=='0'
    markers={}
    for step in (2,4):
        checkpoint=run/f'checkpoint-{step}'
        finalize_saved(checkpoint);markers[step]=verify_ready(checkpoint)
        assert markers[step]['step']==step
    rows=[json.loads(l) for l in (logs/('audit/metrics.jsonl' if arm=='blackbox' else 'run/metrics.jsonl')).read_text().splitlines()]
    updates=[r for r in rows if 'grad_norm' in r]
    assert {r['step'] for r in updates}>={1,2,3,4}
    assert all(math.isfinite(v) for r in updates for v in r.values() if isinstance(v,float))
    assert any(r['grad_norm']>0 for r in updates), 'No optimizer learning signal observed'
    if arm=='blackbox':
        from checkpoint_artifacts import resume_info
        resume=resume_info(run/'checkpoint-2',markers[2]['base_model'],markers[2]['base_revision'])
        assert f"resume    checkpoint step=2, next schedule group={resume['group_offset']}" in (logs/'train-resumed.log').read_text()
        summary=json.loads((logs/'audit/tito_summary.json').read_text())
        assert summary and all(v['tito_pass']==v['completed_results'] and v['retained_tokens']==v['eligible_tokens'] and v['rows_over_token_budget']==0 for v in summary.values())
    else:
        initial=json.loads((run/'optimizer_evidence_from_0.json').read_text())
        resumed=json.loads((run/'optimizer_evidence_from_2.json').read_text())
        assert initial['final_step']==2 and resumed['final_step']==4
        assert initial['weights_changed'] or resumed['weights_changed']
        assert initial['final_weight_digest']==resumed['initial_weight_digest'], 'Resume did not load saved parameters'
        token_rows=[json.loads(l) for l in (run/'token_audit.jsonl').read_text().splitlines()]
        assert {r['step'] for r in token_rows}>={0,1,2,3}
        assert all(row['tito_pass'] and row['supervised']>0 for r in token_rows for row in r['rows'])
    # A native optimizer has persisted nonempty state at the resumed checkpoint.
    import torch
    optimizer=torch.load(run/'checkpoint-4/optimizer.pt',map_location='cpu',weights_only=False)
    assert optimizer['state'] and optimizer['param_groups']
    report={'arm':arm,'job_id':str(job),'passed':True,'optimizer_steps':[1,2,3,4],
            'nonzero_gradient_updates':sum(r['grad_norm']>0 for r in updates),
            'checkpoint_steps':[2,4],'native_optimizer_state_verified':True,
            'resume_verified':True,'tito_pass':True,'weights_updated':True}
    target=ROOT/arm/'training-smoke/validation.json';target.write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=['blackbox','whitebox'],required=True)
    p.add_argument('--job',required=True);a=p.parse_args();print(json.dumps(validate(a.arm,a.job)))
