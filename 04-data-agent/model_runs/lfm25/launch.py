"""Print the production commands; submit only with --submit after qualification."""
import argparse
import json
import shlex
import subprocess
from run import C, OUT, HERE, PY, rt


def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');a=p.parse_args()
    proofs=list(OUT.glob('train-smoke-*/qualification_passed.json'))
    config_hash=__import__('hashlib').sha256((HERE/'config.json').read_bytes()).hexdigest()
    qualified=any(rt.read(x).get('revision')==C['model_revision'] and
        rt.read(x).get('config_sha256')==config_hash for x in proofs)
    train=['sbatch','--parsable','--partition=hopper-prod','--gres=gpu:2','--cpus-per-task=16',
        '--mem=192G','--time=24:00:00','--job-name=lfm25-multi4-train',f'--output={OUT}/train-%j.log',
        str(HERE/'job.sh'),'train']
    baseline=['sbatch','--partition=hopper-prod','--gres=gpu:2','--cpus-per-task=8','--mem=96G',
        '--time=12:00:00','--job-name=lfm25-baseline',f'--output={OUT}/baseline-%j.log',str(HERE/'job.sh'),'eval']
    print('Qualified:',qualified)
    print('Baseline:',shlex.join(baseline))
    print('Training:',shlex.join(train))
    if not a.submit:return
    if not qualified:raise SystemExit('Optimizer/save/resume/reloaded-checkpoint smoke must pass first')
    receipt=OUT/'production_submission.json'
    if receipt.exists():raise SystemExit('A production submission already exists; inspect it before relaunching')
    job=subprocess.check_output(train,text=True).strip().split(';')[0]
    rt.write_json(receipt,{'training_job':job,'controller_job':None})
    command=[str(PY),str(HERE/'controller.py'),'--training-job',job]
    controller=subprocess.check_output(['sbatch','--parsable','--partition=hopper-cpu','--cpus-per-task=2',
        '--mem=16G','--time=48:00:00','--job-name=lfm25-eval-controller',f'--output={OUT}/controller-%j.log',
        '--wrap',shlex.join(command)],text=True).strip().split(';')[0]
    rt.write_json(receipt,{'training_job':job,'controller_job':controller})
    print(json.dumps(rt.read(receipt)))

if __name__=='__main__':main()
