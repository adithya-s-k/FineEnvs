"""Print or submit the matched-task SFT trainer and its separate eval watcher."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, help='Defaults to the experiment data directory')
    p.add_argument('--partition', default='hopper-prod')
    p.add_argument('--eval-partition', default='hopper-prod')
    p.add_argument('--train-time', default='08:00:00')
    p.add_argument('--submit', action='store_true')
    args = p.parse_args()
    root, experiment, out = args.workspace.resolve(), args.experiment.resolve(), args.output.resolve()
    data = args.data.resolve() if args.data else experiment / 'data'
    code = Path(__file__).resolve().parent
    proof_file = experiment / 'qualification.json'
    if not proof_file.exists():
        raise SystemExit('Run and verify the training, longest-sequence, and four-harness smokes first.')
    proof = json.loads(proof_file.read_text())
    if not proof['passed']:
        raise SystemExit('SFT qualification did not pass.')
    from prepare import digest
    if any(digest(code / name) != value for name, value in proof['scripts_sha256'].items()):
        raise SystemExit('Implementation changed since qualification; rerun the affected checks.')
    if digest(data / 'train.jsonl') != proof['train_sha256']:
        raise SystemExit('Selected data changed since qualification.')
    preparation = json.loads((data / 'preparation.json').read_text())
    task_count = preparation['matched_sft_tasks']
    model_name = 'qwen35' if preparation.get('model') == 'Qwen/Qwen3.5-2B' else 'lfm25'
    command = [str(root / '.venv312/bin/python'), '-u', str(code / 'train_entry.py'),
               '--workspace', str(root), '--data', str(data), '--output', str(out),
               '--epochs', '2', '--learning-rate', '3e-6', '--gradient-accumulation', '8',
               '--save-steps', '50', '--eval-concurrency', '100']
    train = ['sbatch', '--parsable', f'--partition={args.partition}', '--gres=gpu:1',
             '--cpus-per-task=8', '--mem=96G', f'--time={args.train_time}', f'--job-name={model_name}-sft-{task_count}',
             f'--output={out}/train-%j.log', f'--export=ALL,PYTHONPATH={root}/trl,OMP_NUM_THREADS=4',
             '--wrap', shlex.join(command)]
    watcher = [str(root / '.venv312/bin/python'), '-u', str(code / 'watch_evals.py'),
               '--workspace', str(root), '--run-dir', str(out), '--training-job', 'TRAIN_JOB_ID',
               '--partition', args.eval_partition, '--concurrency', '100', '--max-active', '1', '--submit']
    print('Trainer:', shlex.join(train))
    print('Separate eval watcher:', shlex.join(watcher))
    if not args.submit:
        return
    out.mkdir(parents=True, exist_ok=False)
    job = subprocess.check_output(train, text=True).strip().split(';')[0]
    (out / 'submission.json').write_text(json.dumps({'training_job': job, 'training_command': train}, indent=2) + '\n')
    watcher[watcher.index('TRAIN_JOB_ID')] = job
    watcher_job = subprocess.check_output(['sbatch','--parsable','--partition=hopper-cpu','--cpus-per-task=1',
        '--mem=4G','--time=48:00:00',f'--job-name={model_name}-sft-eval-watch',f'--output={out}/watcher-%j.log',
        '--wrap',shlex.join(watcher)], text=True).strip().split(';')[0]
    (out / 'submission.json').write_text(json.dumps({'training_job': job, 'watcher_job': watcher_job,
        'training_command': train, 'watcher_command': watcher}, indent=2) + '\n')


if __name__ == '__main__':
    main()
