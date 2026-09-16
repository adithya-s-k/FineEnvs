"""Pass@1 for the native standalone OpenCode client, with a separate ledger per backend."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import time
from common import RUN, MODEL, configure, write_json
configure()
from data_agent_env import DataAgentEnv, opencode_agent_turns, to_trace_entries
from openenv.core.harness.capture.validate import validate_training_turn


def audit(result):
    entries = opencode_agent_turns(to_trace_entries(result))
    if result.rollout_type != "train" or not entries:
        raise ValueError("No TiTO training turns")
    for entry in entries:
        validate_training_turn(entry['prompt_token_ids'], entry['completion_token_ids'],
                               entry['per_token_logps'], entry['loss_mask'])
    logps = [p for e in entries for p in e['per_token_logps']]
    if not any(abs(p) > 1e-8 for p in logps):
        raise ValueError("All log probabilities are zero")
    return {"tito_pass": True, "agent_turns": len(entries),
            "supervised_tokens": sum(sum(e['loss_mask']) for e in entries),
            "forwarded_tokens": sum(len(e['loss_mask']) for e in entries)}


def summarize(ledger, expected, manifest, ungraded):
    selected = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row=json.loads(line)
            if row.get('correctness') is not None: selected.setdefault(row['index'],row)
    difficulty={}
    for i,row in selected.items():
        tier=manifest[i]['difficulty']; d=difficulty.setdefault(tier,{'graded':0,'correct':0})
        d['graded']+=1;d['correct']+=int(row['correctness'] >= 1.0)
    correct=sum(int(r['correctness']>=1.0) for r in selected.values())
    result={'metric':'pass@1','implementation':'standalone-opencode','graded_cells':len(selected),
        'expected_cells':expected,'complete':len(selected)==expected,'correct':correct,
        'pass_at_1':correct/len(selected) if selected else None,'difficulty':difficulty,
        'ungraded_attempts':ungraded,'tito_pass':bool(selected) and all(r.get('tito_pass') for r in selected.values())}
    result['comparison_ready']=result['complete'] and result['tito_pass']
    return selected,result


def evaluate(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    tasks=sorted(json.loads((RUN/'test_manifest.json').read_text())['tasks'],key=lambda t:t['name'])
    limit=args.limit or 250
    if limit > len(tasks): raise ValueError('Limit exceeds frozen test set')
    order=list(range(limit))
    # Spread the ramp over the frozen set instead of measuring only the leading easy tasks.
    import random
    random.Random(42).shuffle(order)
    all_scores={}
    for backend in args.backends.split(','):
        if backend not in {'daytona','hf','e2b'}:raise ValueError('Unsupported sandbox backend')
        dest=out/backend;dest.mkdir(exist_ok=True)
        ledger=dest/'results.jsonl';failed=dest/'ungraded.jsonl'
        ungraded=len(failed.read_text().splitlines()) if failed.exists() else 0
        selected,scores=summarize(ledger,limit,tasks,ungraded)
        backend_limit = getattr(args, backend + '_concurrency', None)
        concurrency_limit = args.concurrency if backend_limit is None else backend_limit
        if not 1 <= concurrency_limit <= 100:
            raise ValueError('Backend concurrency must be between 1 and 100')
        phases=[(min(8,concurrency_limit),min(8,limit)),
                (min(32 if backend == 'daytona' else 16,concurrency_limit),min(32,limit)),
                (concurrency_limit,limit)]
        if getattr(args, 'no_ramp', False):
            phases=[(concurrency_limit,limit)]
        scalability=[]
        write_json(dest/'configuration.json',{'backend':backend,'concurrency_limit':concurrency_limit,
                   'phases':phases,'expected_cells':limit})
        def one(index):
            start=time.monotonic()
            client=DataAgentEnv(args.server,message_timeout_s=1800)
            try:
                result=client.run_rollout(split='test',index=index,llm_url=args.vllm_url,
                    model=args.model,api_key=os.environ.get(args.api_key_env,''),sandbox=backend,
                    agent_step_limit=17,agent_timeout_s=600,require_tokens=True,timeout_s=1800)
                path=dest/'captures'/f'{index}-{result.metadata["rollout_id"]}.json'
                write_json(path,result.model_dump())
                if result.metadata.get('error') or result.correctness is None:
                    return {'index':index,'graded':False,'error':result.metadata.get('error','ungraded'),
                            'capture_file':str(path),'elapsed_s':time.monotonic()-start}
                if result.metadata.get('task_id') != tasks[index]['name']:
                    raise ValueError('Returned task identity differs from frozen manifest')
                if result.metadata.get('opencode_version') != '1.18.31':
                    raise ValueError('Unverified OpenCode version')
                # Once graded, a failed audit stops the cohort; never retry a scored zero.
                proof=audit(result)
                return {'index':index,'task_id':tasks[index]['name'],'backend':backend,
                    'correctness':result.correctness,'reward':result.reward,'capture_file':str(path),
                    'elapsed_s':time.monotonic()-start,'opencode_version':'1.18.31',**proof}
            finally:client.close()
        for stage,(concurrency,count) in enumerate(phases):
            remaining=[i for i in order if i not in selected][:count]
            if not remaining:continue
            start=time.monotonic();before=len(selected);attempts=0
            write_json(dest/'progress.json',{'stage':stage,'concurrency':concurrency,
                       'graded_before':before,'tasks_this_stage':len(remaining),'started_at':time.time()})
            for attempt in range(3):
                pending=[i for i in remaining if i not in selected]
                if not pending:break
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures={pool.submit(one,i):i for i in pending}
                    for future in concurrent.futures.as_completed(futures):
                        i=futures[future];attempts+=1
                        # Transport failures carry no returned grade and can be retried.
                        try:row=future.result()
                        except (ValueError,AssertionError):raise
                        except Exception as e:row={'index':i,'graded':False,'error_type':type(e).__name__}
                        target=ledger if row.get('correctness') is not None else failed
                        with target.open('a') as f:f.write(json.dumps(row)+'\n')
                        if target==failed:ungraded+=1
                        selected,scores=summarize(ledger,limit,tasks,ungraded)
                        write_json(dest/'scores.json',scores)
                        print(json.dumps({'backend':backend,'stage':stage,'graded':len(selected),'latest_index':i,
                            'latest_graded':target==ledger,'ungraded_attempts':ungraded}),flush=True)
                if len(selected)-before < len(remaining) and attempt<2:time.sleep(5)
            elapsed=time.monotonic()-start
            scalability.append({'stage':stage,'concurrency':concurrency,'new_graded':len(selected)-before,
                'attempts':attempts,'elapsed_s':elapsed,'graded_per_minute':(len(selected)-before)*60/elapsed})
            write_json(dest/'scalability.json',scalability)
            if len(selected)-before < .9*len(remaining):
                raise RuntimeError(f'{backend} failed the coverage gate at concurrency {concurrency}')
        if limit == 250 and scores["comparison_ready"]:
            from native_grading_audit import verify
            verify(ledger, dest)
            if backend == "daytona":
                write_json(out / "verification.json", json.loads((dest / "verification.json").read_text()))
        all_scores[backend]=scores
        write_json(out/'canonical_scores.json',all_scores)
        if not scores['comparison_ready']:raise RuntimeError(f'{backend} baseline incomplete or TiTO invalid')
    return all_scores

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',required=True);p.add_argument('--vllm-url',required=True)
    p.add_argument('--model',default=MODEL);p.add_argument('--api-key-env',default='HF_TOKEN')
    p.add_argument('--backends',default='daytona,hf');p.add_argument('--concurrency',type=int,default=32)
    p.add_argument('--daytona-concurrency',type=int);p.add_argument('--hf-concurrency',type=int)
    p.add_argument('--no-ramp',action='store_true')
    p.add_argument('--limit',type=int,default=250);p.add_argument('--out',required=True)
    print(json.dumps(evaluate(p.parse_args())),flush=True)
