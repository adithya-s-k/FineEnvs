"""Qualify each student, then train and submit separate epoch evaluations."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

p = argparse.ArgumentParser()
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--experiment', type=Path, required=True)
a = p.parse_args()
root, arm = a.workspace.resolve(), a.experiment.resolve()
code = Path(__file__).resolve().parent
python = str(root / '.venv312/bin/python')
env = {**os.environ, 'PYTHONPATH':str(root/'trl'), 'OMP_NUM_THREADS':'4', 'TOKENIZERS_PARALLELISM':'false'}

def state(phase, **extra):
    record = {'phase':phase, 'updated_utc':datetime.now(timezone.utc).isoformat(), 'job':os.environ['SLURM_JOB_ID'], **extra}
    temp = arm/'pipeline_status.tmp'
    temp.write_text(json.dumps(record,indent=2)+'\n'); temp.replace(arm/'pipeline_status.json')
    print(json.dumps(record),flush=True)

def run(name, args):
    state(name)
    with (arm/(name+'.log')).open('a') as log:
        subprocess.run([python,'-u',str(code/args[0]),*args[1:]],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

try:
    proof = json.loads((arm/'data/preparation.json').read_text())
    assert proof['rollouts']==3189 and proof['training_examples']==17929 and proof['unique_tasks']==888
    assert set(proof['harness_training_examples'])=={'opencode','claude-code','codex','mini-swe-agent'}
    run('native_preparation', ['test_opencode_messages.py','--data',str(arm/'data'),'--receipt',str(arm/'native_preparation_verified.json')])
    run('epoch_eval_validation',['test_eval_policy.py',str(arm/'epoch_eval_validation.json')])
    from prepare import digest
    test_source=Path(proof['test_manifest']).parent/'dataset/tasks'
    frozen=root/'experiments/lfm25-multi4-20260921/runtime/data/test/tasks'
    names={r['name'] for r in json.loads(Path(proof['test_manifest']).read_text())['tasks']}
    assert names=={p.name for p in frozen.iterdir() if p.is_dir()}
    count=0
    for name in sorted(names):
        for path in (test_source/name).rglob('*'):
            if path.is_file():
                assert digest(path)==digest(frozen/name/path.relative_to(test_source/name))
                count+=1
    (arm/'test_split_equivalence.json').write_text(json.dumps({'passed':True,'tasks':len(names),'compared_files':count})+'\n')
    run('smoke_driver',['smoke_opencode.py','--workspace',str(root),'--data',str(arm/'data'),'--output',str(arm/'smoke')])
    steps=len(json.loads((arm/'data/smoke_task_ids.json').read_text()))
    run('logging_validation',['test_local_logging.py','--run-root',str(arm/'smoke'),'--receipt',str(arm/'local_logging_verified.json'),'--steps',str(steps)])
    run('qualification',['qualify_opencode.py','--experiment',str(arm)])
    qualification=json.loads((arm/'qualification.json').read_text())
    assert qualification['passed']
    assert all(digest(code/name)==value for name,value in qualification['scripts_sha256'].items())
    out=arm/'production';out.mkdir(exist_ok=False)
    watcher=[python,'-u',str(code/'watch_evals.py'),'--workspace',str(root),'--run-dir',str(out),'--training-job',os.environ['SLURM_JOB_ID'],'--partition','hopper-prod,hopper-dev','--concurrency','100','--max-active','1','--submit']
    watcher_job=subprocess.check_output(['sbatch','--parsable','--partition=hopper-cpu','--cpus-per-task=1','--mem=4G','--time=48:00:00',f'--job-name={arm.name}-multi-sft-watch',f'--output={out}/watcher-%j.log','--wrap',shlex.join(watcher)],text=True).strip().split(';')[0]
    (out/'submission.json').write_text(json.dumps({'training_job':os.environ['SLURM_JOB_ID'],'watcher_job':watcher_job,'watcher_command':watcher},indent=2)+'\n')
    state('training',watcher_job=watcher_job,updates_per_epoch=(proof['training_examples']+7)//8)
    command=[python,'-u',str(code/'train_entry.py'),'--workspace',str(root),'--data',str(arm/'data'),'--output',str(out),'--epochs','2','--learning-rate','3e-6','--gradient-accumulation','8','--save-steps','50','--eval-concurrency','100']
    (out/'train_command.json').write_text(json.dumps(command,indent=2)+'\n')
    with (out/'train.log').open('a') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    state('training_complete_evaluations_continue',watcher_job=watcher_job)
except Exception as exc:
    state('failed',error=str(exc))
    raise
