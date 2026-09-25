"""Canonical pass@1 summaries: fixed identities, first graded attempt, pinned harnesses."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'logs/20260915'
HARNESS_VERSIONS={'opencode':'1.18.31','claude-code':'2.1.270','codex':'0.154.0','mini-swe-agent':'2.4.6'}


def summarize(arm,output):
    manifest=json.loads((ROOT/'test_manifest.json').read_text())
    selected={};ungraded=0
    files=sorted((output/'traces').glob('*.jsonl')) if arm=='blackbox' else [output/'attempts.jsonl']
    for p in files:
        for line in p.read_text().splitlines():
            r=json.loads(line)
            valid=r.get('reward') in (0,1) and (r.get('n_turns',0)>0 if arm=='blackbox' else r.get('tito_pass',False))
            if not valid:
                ungraded+=1;continue
            assert r.get('rep',0)==0 and 0<=r['index']<250
            selected.setdefault((r['harness'],r['index']),r)
    harnesses=list(HARNESS_VERSIONS) if arm=='blackbox' else ['whitebox_seta']
    expected={(h,i) for h in harnesses for i in range(250)}
    complete=set(selected)==expected
    if arm=='blackbox':
        audit_path=output/'final_tito.json'
        audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        tito=complete and audit.get('tito_pass',False) and audit.get('counts')=={h:250 for h in harnesses}
    else:
        tito=complete and all(r['tito_pass'] for r in selected.values())
    scores={};versions={h:{} for h in harnesses}
    for h in harnesses:
        rows=[r for (name,_),r in selected.items() if name==h]
        difficulty={}
        for level in ('easy','medium','hard'):
            subset=[r for r in rows if manifest['tasks'][r['index']]['difficulty']==level]
            correct=sum(r['reward'] for r in subset)
            difficulty[level]={'graded':len(subset),'correct':correct,'pass_at_1':correct/len(subset) if subset else None}
        correct=sum(r['reward'] for r in rows)
        scores[h]={'graded':len(rows),'correct':correct,'pass_at_1':correct/len(rows) if rows else None,'difficulty':difficulty}
        if arm=='blackbox':
            for row in rows:
                p=output/'trials'/row.get('trial_name','missing')/'result.json'
                data=json.loads(p.read_text()) if p.is_file() else {}
                version=(data.get('agent_info') or {}).get('version') or 'unverified'
                versions[h][version]=versions[h].get(version,0)+1
    pins=all(versions[h]=={HARNESS_VERSIONS[h]:250} for h in harnesses) if arm=='blackbox' else True
    result={'metric':'pass@1','arm':arm,'complete':complete,'graded_cells':len(selected),
            'expected_cells':len(expected),'harnesses':scores,'tito_pass':tito,
            'ungraded_attempts':ungraded,'harness_versions':versions,'harness_versions_match_baseline':pins,
            'comparison_ready':complete and tito and pins,
            'average_pass_at_1':sum(r['reward'] for r in selected.values())/len(selected) if selected else None,
            'selection':'first graded attempt per fixed task/harness; infrastructure failures excluded and retried'}
    (output/'canonical_scores.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['blackbox','whitebox'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();result=summarize(a.arm,a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('harnesses','harness_versions')}))
    raise SystemExit(2 if a.require_complete and not result['comparison_ready'] else 0)
