"""Submit saved SFT checkpoint evaluations on separate GPU allocations."""
import argparse
import json
import getpass
from pathlib import Path
import shlex
import subprocess
import time


def active(job_id):
    jobs = subprocess.check_output(['squeue', '-h', '-u', getpass.getuser(), '-o', '%A'], text=True).split()
    return str(job_id) in jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--training-job', required=True)
    p.add_argument('--partition', default='hopper-prod')
    p.add_argument('--max-active', type=int, default=1)
    p.add_argument('--concurrency', type=int, default=100)
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    root, run = args.workspace.resolve(), args.run_dir.resolve()
    code = Path(__file__).resolve().parent
    while True:
        requests = sorted((f for f in (run / 'eval_requests').glob('checkpoint-*.json') if not f.name.endswith('.submitted.json') and '.attempt-' not in f.name), key=lambda f: int(f.stem.split('-')[-1]))
        pending = []
        ongoing = 0
        for request in requests:
            info = json.loads(request.read_text())
            receipt = request.with_suffix('.submitted.json')
            if receipt.exists():
                job = json.loads(receipt.read_text())['job_id']
                if active(job):
                    ongoing += 1
                elif not (run / 'evaluations' / f"checkpoint-{info['step']}" / 'evaluation_verified.json').exists():
                    submitted = json.loads(receipt.read_text())
                    attempt = submitted.get('attempt', 1)
                    if attempt < 3:
                        info['recovery_attempt'] = attempt + 1
                        pending.append((request, info))
            else:
                pending.append((request, info))
        for request, info in pending:
            if ongoing >= args.max_active:
                break
            checkpoint = Path(info['checkpoint'])
            if not (checkpoint / 'checkpoint.saved.json').exists():
                raise RuntimeError(f'Unfinished save: {checkpoint}')
            output = run / 'evaluations' / f"checkpoint-{info['step']}"
            output.parent.mkdir(exist_ok=True)
            command = [str(root / '.venv312/bin/python'), str(code / 'evaluate.py'), '--workspace', str(root),
                       '--checkpoint', str(checkpoint), '--output', str(output), '--concurrency', str(args.concurrency)]
            if 'recovery_attempt' in info:
                command += ['--resume']
                command[command.index('--concurrency')+1] = '10'
            model_name = 'qwen35'  if info.get('model') == 'Qwen/Qwen3.5-2B' else 'lfm25'
            sbatch = ['sbatch', '--parsable', f'--partition={args.partition}', '--gres=gpu:1',
                      '--cpus-per-task=8', '--mem=128G', '--time=12:00:00', f'--job-name={model_name}-sft-eval',
                      f'--output={output.parent}/checkpoint-{info["step"]}-%j.log',
                      '--export=ALL,PYTHONPATH=' + str(root / 'trl'), '--wrap', shlex.join(command)]
            print(shlex.join(sbatch), flush=True)
            if args.submit:
                job = subprocess.check_output(sbatch, text=True).strip().split(';')[0]
                receipt = request.with_suffix('.submitted.json')
                if receipt.exists():
                    old = json.loads(receipt.read_text())
                    receipt.rename(request.with_suffix(f".attempt-{old.get('attempt', 1)}.json"))
                receipt.write_text(json.dumps({'job_id': job, 'command': sbatch, 'checkpoint': str(checkpoint), 'attempt': info.get('recovery_attempt',1)}, indent=2) + '\n')
                ongoing += 1
        if not args.submit:
            break
        subprocess.run([str(root / '.venv312/bin/python'), str(code / 'plot_comparison.py')], check=True)
        if not active(args.training_job) and not pending and ongoing == 0:
            failures = []
            for request in requests:
                step = json.loads(request.read_text())['step']
                if not (run / 'evaluations' / f'checkpoint-{step}' / 'evaluation_verified.json').exists():
                    failures.append(step)
            if len(requests) != 2:
                failures.append('expected_two_epoch_evaluations')
            (run / 'eval_controller_summary.json').write_text(json.dumps({'requested_checkpoints': len(requests), 'incomplete_checkpoints': failures}, indent=2) + '\n')
            if failures:
                raise RuntimeError(f'Evaluations need recovery: {failures}; original results retained')
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
