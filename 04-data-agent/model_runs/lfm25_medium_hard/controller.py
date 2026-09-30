"""Submit independent checkpoint evals and sync training metrics to Trackio."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from run import base, arm, HERE
C, OUT, REPO, rt, verify_ready = base.C, base.OUT, base.REPO, base.rt, base.verify_ready


def due_steps(steps, final_step=None):
    return sorted(s for s in steps if s > 0 and (s % 100 == 0 or s == final_step))


def sync(training, completed):
    os.environ['TRACKIO_DIR']=str(OUT/'trackio-sync')
    os.environ['TRACKIO_STORAGE_MODE']='sqlite'
    import trackio_multi4 as native
    import httpx
    class RemoteClient:
        def __init__(self, *args, **kwargs): pass
        def predict(self, api_name, **kwargs):
            response = httpx.post('https://fineenvs-data-agent-training-comparison-trackio.hf.space/api/' + api_name.lstrip('/'), json=kwargs, headers={'Authorization':'Bearer '+os.environ['HF_TOKEN']}, timeout=60)
            response.raise_for_status()
            payload=response.json()
            if 'error' in payload: raise RuntimeError(payload['error'])
            return payload.get('data')
    from trackio.sqlite_storage import SQLiteStorage
    project=C['logging']['project']; name=C['run_name']
    metadata={'model':C['model'],'revision':C['model_revision'],'training_tasks':1000}
    records=[]
    path=training/'audit/metrics.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            metrics=native.scalars({k:v for k,v in row.items() if k!='step'},'train/')
            if 'reward' in row: metrics['train/shaped_reward']=row['reward']
            if 'rollout/correctness_mean' in row: metrics['train/raw_correctness']=row['rollout/correctness_mean']
            records.append(native.event(project,name,row['step'],metrics,metadata))
    receipts=training/'audit/optimizer_rollouts.jsonl'
    if receipts.exists():
        for line in receipts.read_text().splitlines():
            receipt=json.loads(line)
            ids={r['rollout_id'] for r in receipt['rollouts']}
            evidence=[]
            for rollout_id in ids:
                path=training/'audit/efficiency'/f'{rollout_id}.json'
                if path.exists(): evidence.append(rt.read(path))
            scored=[r for r in evidence if r['correctness'] is not None]
            counted=[r for r in scored if r['count_verified']]
            metrics={'train/action_count_coverage':len(counted)/len(scored) if scored else 0}
            if scored:
                metrics['train/admitted_correctness']=sum(r['correctness'] for r in scored)/len(scored)
                metrics['train/admitted_efficiency_bonus']=sum(r['bonus'] for r in scored)/len(scored)
            if counted: metrics['train/native_actions']=sum(r['native_actions'] for r in counted)/len(counted)
            records.append(native.event(project,name,receipt['step'],metrics,metadata,identity='admitted-reward-components'))
    for step,path in completed.items():
        score=rt.read(path)
        if not score.get('comparison_ready'):continue
        metrics={'eval/pass_at_1':score['average_pass_at_1']}
        for h,item in score['harnesses'].items():
            metrics[f'eval/harness/{h}/pass_at_1']=item['pass_at_1']
            for difficulty,tier in item['difficulty'].items():
                metrics[f'eval/harness_difficulty/{h}/{difficulty}/pass_at_1']=tier['pass_at_1']
        records.append(native.event(project,name,int(step),metrics,metadata))
    if records:
        native.import_events(records)
        native.backup_project(project,OUT/'trackio-backup')
        client=RemoteClient(C['logging']['space_id'],hf_token=os.environ['HF_TOKEN'],httpx_kwargs={'timeout':30})
        # Use the same tested transport as the existing comparison logger.
        logs=SQLiteStorage.get_all_logs_for_sync(project)
        for i in range(0,len(logs),500):
            client.predict(api_name='/bulk_log',logs=logs[i:i+500],hf_token=os.environ['HF_TOKEN'])
        native.verify_remote_records(client,project,logs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--training-job',required=True);a=p.parse_args()
    training=OUT/('train-'+a.training_job)
    state_path=OUT/('controller-'+a.training_job+'.json')
    state=rt.read(state_path) if state_path.exists() else {'evaluations':{}}
    while True:
        queue=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%A|%N'],text=True)
        active=dict(line.strip().split('|', 1) for line in queue.splitlines() if line.strip())
        nodes=active.get(a.training_job, '')
        if nodes and nodes not in ('(null)','None'):
            state['training_node']=nodes
        running=sum(str(v['job_id']) in active for v in state['evaluations'].values())
        steps=[int(p.name.split('-')[-1]) for p in (training/'run').glob('checkpoint-*')]
        final_step=max(steps) if steps and a.training_job not in active else None
        for step in due_steps(steps,final_step):
            if str(step) in state['evaluations'] or running>=2:continue
            checkpoint=training/f'run/checkpoint-{step}'
            if not (checkpoint/'checkpoint.saved.json').exists():continue
            try:
                if not (checkpoint/'checkpoint.ready.json').exists():
                    base.finalize_saved(checkpoint)
                verify_ready(checkpoint)
            except (ValueError, OSError):
                # A final save can rewrite the periodic checkpoint at the same step.
                continue
            policy_path=OUT.parent/'eval_resources.json'
            policy=rt.read(policy_path) if policy_path.exists() else {}
            gpu_count=int(policy.get('gpus',2))
            if gpu_count not in (1,2):raise ValueError('Eval GPU count must be 1 or 2')
            cmd=['sbatch','--parsable','--partition='+policy.get('partition','hopper-prod'),f'--gres=gpu:{gpu_count}','--cpus-per-task=8',
                '--mem=96G','--time=12:00:00',f'--job-name=lfm25-eval-{step}',
                f'--output={OUT}/eval-{step}-%j.log',str(HERE/'job.sh'),arm,'eval','--model-path',str(checkpoint)]
            if state.get('training_node') and policy.get('exclude_training_node',True):
                cmd.insert(1,'--exclude='+state['training_node'])
            job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
            state['evaluations'][str(step)]={'job_id':job,'checkpoint':str(checkpoint)}
            rt.write_json(state_path,state);running+=1
        completed={}
        for step,item in state['evaluations'].items():
            path=OUT/('eval-'+item['job_id'])/'canonical_scores.json'
            if path.exists():
                score=rt.read(path)
                if score.get('complete') and score.get('comparison_ready') and score.get('graded_cells')==C['evaluation']['evaluations_per_checkpoint']:
                    completed[step]=str(path)
        try:sync(training,completed);state.pop('logging_error',None)
        except Exception as e:state['logging_error']=str(e)
        state.update(last_checked=time.time(),completed=completed)
        rt.write_json(state_path,state)
        if a.training_job not in active and running==0:
            missing=sorted(set(state['evaluations'])-set(completed),key=int)
            if missing:
                state['incomplete_evaluations']=missing
                rt.write_json(state_path,state)
                raise RuntimeError('Evaluation jobs ended without complete results: '+','.join(missing))
            return
        time.sleep(60)

if __name__=='__main__':main()
