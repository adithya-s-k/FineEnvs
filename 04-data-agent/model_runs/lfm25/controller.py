"""Submit independent checkpoint evals and sync training metrics to Trackio."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from run import C, OUT, HERE, REPO, rt, verify_ready


def due_steps(steps):
    return sorted(s for s in steps if 0 < s <= 1000 and s % 100 == 0)


def sync(training, completed):
    os.environ['TRACKIO_DIR']=str(OUT/'trackio-sync')
    os.environ['TRACKIO_STORAGE_MODE']='sqlite'
    import trackio_multi4 as native
    from trackio.remote_client import RemoteClient
    from trackio.sqlite_storage import SQLiteStorage
    project=C['logging']['project']; name=C['run_name']
    metadata={'model':C['model'],'revision':C['model_revision'],'training_tasks':1000}
    records=[]
    path=training/'audit/metrics.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            records.append(native.event(project,name,row['step'],native.scalars(
                {k:v for k,v in row.items() if k!='step'},'train/'),metadata))
    for step,path in completed.items():
        score=rt.read(path)
        if not score.get('comparison_ready'):continue
        metrics={'eval/pass_at_1':score['average_pass_at_1']}
        for h,item in score['harnesses'].items():
            metrics[f'eval/{h}/pass_at_1']=item['pass_at_1']
            for difficulty,tier in item['difficulty'].items():
                metrics[f'eval/{h}/{difficulty}/pass_at_1']=tier['pass_at_1']
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
        active=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%A'],text=True).split()
        nodes=subprocess.check_output(['squeue','-j',a.training_job,'-h','-o','%N'],text=True).strip()
        if nodes and nodes not in ('(null)','None'):
            state['training_node']=nodes
        running=sum(str(v['job_id']) in active for v in state['evaluations'].values())
        steps=[int(p.name.split('-')[-1]) for p in (training/'run').glob('checkpoint-*')]
        for step in due_steps(steps):
            if str(step) in state['evaluations'] or running>=2:continue
            checkpoint=training/f'run/checkpoint-{step}'
            if not (checkpoint/'checkpoint.ready.json').exists():continue
            try:
                verify_ready(checkpoint)
            except (ValueError, OSError):
                # A final save can rewrite the periodic checkpoint at the same step.
                continue
            cmd=['sbatch','--parsable','--partition=hopper-prod','--gres=gpu:2','--cpus-per-task=8',
                '--mem=96G','--time=12:00:00',f'--job-name=lfm25-eval-{step}',
                f'--output={OUT}/eval-{step}-%j.log',str(HERE/'job.sh'),'eval','--model-path',str(checkpoint)]
            if state.get('training_node'):
                cmd.insert(1,'--exclude='+state['training_node'])
            job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
            state['evaluations'][str(step)]={'job_id':job,'checkpoint':str(checkpoint)}
            rt.write_json(state_path,state);running+=1
        completed={}
        for step,item in state['evaluations'].items():
            path=OUT/('eval-'+item['job_id'])/'canonical_scores.json'
            if path.exists():completed[step]=str(path)
        try:sync(training,completed);state.pop('logging_error',None)
        except Exception as e:state['logging_error']=str(e)
        state.update(last_checked=time.time(),completed=completed)
        rt.write_json(state_path,state)
        if a.training_job not in active and running==0:
            return
        time.sleep(60)

if __name__=='__main__':main()
