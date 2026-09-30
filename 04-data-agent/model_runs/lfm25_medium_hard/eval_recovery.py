"""Retain graded cells when only an evaluation transport/allocation changes."""
import json
from pathlib import Path
import shutil


def stage(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    previous = json.loads((source / 'services.json').read_text())
    current = json.loads((output / 'services.json').read_text())
    if previous['model'] != current['model'] or previous['revision'] != current['revision']:
        raise ValueError('Cannot reuse results from another checkpoint/revision')
    traces = output / 'traces'
    traces.mkdir(exist_ok=True)
    if any(traces.iterdir()):
        raise ValueError('Recovery destination already contains traces')
    for path in (source / 'traces').iterdir():
        if path.suffix in ('.json', '.jsonl'):
            shutil.copy2(path, traces / path.name)
    trials = output / 'trials'
    trials.mkdir(exist_ok=True)
    for path in (source / 'trials').iterdir():
        target = trials / path.name
        if not target.exists():
            target.symlink_to(path.resolve(), target_is_directory=True)
    (output / 'resume_source.json').write_text(json.dumps({'source': str(source), 'checkpoint': current['model'],
        'policy': 'Keep first graded cell; retry only ungraded cells. Original artifacts retained.'}, indent=2) + '\n')


def install(base):
    import os
    source = os.environ.get('LFM_EVAL_RESUME_FROM')
    if not source:
        return
    source = Path(source).resolve()
    previous_services = base.services
    previous_evaluate = base.evaluate
    old_job = source.name.removeprefix('eval-')
    if not old_job.isdigit():
        raise ValueError('Recovery source must be an eval-JOB directory')

    def services(output, model, training):
        if training:
            raise ValueError('Evaluation recovery cannot be used for training')
        current = os.environ.get('SLURM_JOB_ID')
        os.environ['SLURM_JOB_ID'] = old_job
        try:
            return previous_services(output, model, training)
        finally:
            if current is None: os.environ.pop('SLURM_JOB_ID', None)
            else: os.environ['SLURM_JOB_ID'] = current

    def evaluate(output, engine, server, smoke):
        if smoke:
            raise ValueError('Full evaluation recovery only')
        stage(source, output)
        return previous_evaluate(output, engine, server, smoke)

    base.services = services
    base.evaluate = evaluate
