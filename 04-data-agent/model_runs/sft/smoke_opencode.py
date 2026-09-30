"""Verify a student update, resume, longest example and four-harness evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.workspace.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parent
    python = str(root / '.venv312/bin/python')
    count = len(json.loads((args.data / 'smoke_task_ids.json').read_text()))
    env = {**os.environ, 'PYTHONPATH': str(root / 'trl'), 'PYTHONUNBUFFERED': '1',
           'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'}
    train = [python, '-u', str(code / 'train_entry.py'), '--workspace', str(root),
             '--data', str(args.data.resolve()), '--output', str(output), '--smoke',
             '--max-steps', str(count), '--save-steps', str(count), '--gradient-accumulation', '1']
    phases = [
        ('train-first', train + ['--stop-after', '2']),
        ('train-resumed', train + ['--resume', str(output / 'run/checkpoint-2')]),
        ('eval-driver', [python, '-u', str(code / 'evaluate.py'), '--workspace', str(root),
                         '--checkpoint', str(output / f'run/checkpoint-{count}'),
                         '--output', str(output / 'eval'), '--smoke']),
    ]
    (output / 'commands.json').write_text(json.dumps(phases, indent=2) + '\n')
    for name, command in phases:
        print(f'Starting {name}', flush=True)
        with (output / f'{name}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Passed {name}', flush=True)


if __name__ == '__main__':
    main()
