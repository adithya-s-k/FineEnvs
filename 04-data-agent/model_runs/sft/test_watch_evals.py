"""Verify fresh, resumed and already-complete evaluation submission behavior."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

code=Path(__file__).resolve().parent
with TemporaryDirectory() as temp:
    root=Path(temp);run=root/'production';req=run/'eval_requests';req.mkdir(parents=True)
    for step in [2242,4484]:
        checkpoint=run/'run'/f'checkpoint-{step}';checkpoint.mkdir(parents=True)
        (checkpoint/'checkpoint.saved.json').write_text('{}')
        (req/f'checkpoint-{step}.json').write_text(json.dumps({'step':step,'checkpoint':str(checkpoint),'model':'Qwen/Qwen3.5-2B'}))
    def preview():
        text=io.StringIO()
        with patch.object(sys,'argv',[str(code/'watch_evals.py'),'--workspace',str(root),'--run-dir',str(run),'--training-job','123','--max-active','2']),patch('subprocess.check_output',return_value=''),contextlib.redirect_stdout(text):
            runpy.run_path(str(code/'watch_evals.py'),run_name='__main__')
        return text.getvalue()
    initial=preview();assert initial.count('sbatch')==2 and initial.count('--concurrency 100')==2 and '--resume' not in initial
    receipt=req/'checkpoint-2242.submitted.json';receipt.write_text(json.dumps({'job_id':'456','attempt':1}))
    resumed=preview();assert resumed.count('sbatch')==2 and resumed.count('--resume')==1 and '--concurrency 10' in resumed
    receipt.write_text(json.dumps({'job_id':'456','attempt':3}))
    exhausted=preview();assert exhausted.count('sbatch')==1 and '--resume' not in exhausted
    receipt.write_text(json.dumps({'job_id':'456','attempt':1}))
    done=run/'evaluations/checkpoint-2242';done.mkdir(parents=True);(done/'evaluation_verified.json').write_text('{}')
    complete=preview();assert complete.count('sbatch')==1 and 'checkpoint-2242' not in complete
print('PASS: concurrency 100, missing-cell recovery at 10, bounded retries, completed evals preserved; no jobs submitted.')
