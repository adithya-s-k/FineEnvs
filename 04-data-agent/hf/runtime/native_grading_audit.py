"""Verify frozen native grading parameters and rescore the unchanged first answers."""
import hashlib
import json
from pathlib import Path

from common import RUN, write_json


def verify(ledger, output):
    from data_agent_env.task import DataAgentTask
    from data_agent_env.tasks import _frozen_rows
    from data_agent_env.verifier import grade_rollout
    import data_agent_env

    selected = {}
    encoded = Path(ledger).read_bytes()
    for line in encoded.decode().splitlines():
        row = json.loads(line)
        if row.get('correctness') is not None:
            selected.setdefault(row['index'], row)
    if set(selected) != set(range(250)):
        raise ValueError('Native grading qualification requires all 250 fixed first answers')
    tasks = {}
    checked = 0
    for split in ('train', 'test'):
        rows = _frozen_rows(str(RUN / 'datasets'), split)
        for index, row in enumerate(rows):
            task = DataAgentTask.from_row(row)
            if (task.atol, task.rtol) != (row['atol'], row['rtol']):
                raise ValueError('Native task parsing changed a frozen grading tolerance')
            checked += 1
            if split == 'test': tasks[index] = task
    if checked != 1250:
        raise ValueError('Unexpected fixed training/test task count')
    reports = []
    for index, row in sorted(selected.items()):
        captured = Path(row['capture_file']).read_bytes()
        result = json.loads(captured)
        task = tasks[index]
        if result['metadata']['task_id'] != task.task_id or row['task_id'] != task.task_id:
            raise ValueError('Native capture/task identity mismatch')
        answer, source = result.get('answer'), result.get('answer_source')
        grade = grade_rollout(task, lambda _: answer if source == 'file' else None,
                              ('/answer',), final_message=answer if source == 'chat' else None)
        if grade.correctness != result['correctness'] or grade.correctness != row['correctness']:
            raise ValueError('Rescoring changed a first graded native answer')
        reports.append({'index': index, 'task_id': task.task_id, 'correctness': grade.correctness,
                        'capture_sha256': hashlib.sha256(captured).hexdigest()})
    if Path(ledger).read_bytes() != encoded:
        raise ValueError('First-graded ledger changed during verification')
    package = Path(data_agent_env.__file__).parent
    proof = {'passed': True, 'parameters_verified': checked, 'original_graded_records_preserved': True,
             'ledger_sha256': hashlib.sha256(encoded).hexdigest(), 'reports': reports,
             'runtime_files': {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                               for name in ('task.py', 'tasks.py', 'verifier.py', 'grader.py')}}
    write_json(Path(output) / 'verification.json', proof)
    return proof
