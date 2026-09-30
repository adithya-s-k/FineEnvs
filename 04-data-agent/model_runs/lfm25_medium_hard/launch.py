"""Submit isolated training and a CPU checkpoint/eval controller."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
from production import check_qualification

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'experiments/lfm25-medium-hard1000-20260921'


def commands(arm, training_job='TRAIN_JOB'):
    output = OUT / arm
    common = ['sbatch', '--parsable', '--export=ALL,NCCL_P2P_DISABLE=1,NCCL_SHM_DISABLE=1,NCCL_CUMEM_ENABLE=0']
    train = common + ['--partition=hopper-prod', '--gres=gpu:2', '--cpus-per-task=16', '--mem=192G',
                     '--time=24:00:00', f'--job-name=lfm-{arm}', f'--output={output}/train-%j.log',
                     str(HERE / 'job.sh'), arm, 'train']
    controller = common + ['--partition=hopper-prod', '--gres=gpu:0', '--cpus-per-task=2', '--mem=4G', '--time=48:00:00',
                           f'--dependency=after:{training_job}', f'--job-name=lfm-watch-{arm}',
                           f'--output={output}/controller-%j.log', '--wrap',
                           shlex.join([str(ROOT / '.venv312/bin/python'), str(HERE / 'controller.py'),
                                       arm, '--training-job', str(training_job)])]
    return train, controller


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=['opencode', 'multi-harness'])
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    proof = check_qualification(OUT / args.arm, HERE)
    train, controller = commands(args.arm)
    record = {'qualification': proof, 'train_command': train, 'controller_command': controller}
    if args.submit:
        job = subprocess.check_output(train, text=True).strip().split(';')[0]
        record['training_job'] = job
        _, controller = commands(args.arm, job)
        try:
            record['controller_job'] = subprocess.check_output(controller, text=True).strip().split(';')[0]
        except Exception:
            subprocess.run(['scancel', job], check=True)
            raise
        record['controller_command'] = controller
        (OUT / args.arm / f'launch-{job}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
