"""Prepare a fresh OpenCode-only ablation of the last stable Harbor recipe."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
REFERENCE = REPO / 'experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-cont-20260915'
TOOLS = REPO / 'experiments/async_grpo_harbor_data_agent/tools'
DEFAULT_OUT = REPO / 'experiments/async_grpo_harbor_data_agent/logs/harbor-opencode-only-20260916'
sys.path.insert(0, str(Path(__file__).parent))
from harness_schedule import validate_schedule


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n')


def single_harness_schedule(reference):
    validate_schedule(reference)
    schedule = copy.deepcopy(reference)
    n = schedule['task_count']
    schedule.update(harnesses=['opencode'], passes_per_cycle=1, groups_per_cycle=n)
    schedule['groups'] = schedule['groups'][:n]
    for group in schedule['groups']:
        group['harness'] = 'opencode'
    validate_schedule(schedule)
    assert all({k:v for k,v in a.items() if k != 'harness'} == {k:v for k,v in b.items() if k != 'harness'}
               for a,b in zip(reference['groups'][:n], schedule['groups'], strict=True))
    return schedule


def prepare(root):
    if root.exists():
        raise ValueError('Run directory exists; inspect its submission record before retrying')
    config = read(REFERENCE/'run_config.json')
    prior = copy.deepcopy(config)
    manifest = read(REFERENCE/'manifest.json')
    schedule = single_harness_schedule(read(REFERENCE/'harness_schedule.json'))
    root.mkdir(parents=True)
    for name in ('indices.txt','runtime_versions.json'):
        shutil.copyfile(REFERENCE/name,root/name)
    manifest.update(harnesses=['opencode'],pairs_per_cycle=1000,passes_per_cycle=1,
                    groups_per_pass=1000,rollouts_per_scheduled_cycle=8000)
    write(root/'manifest.json',manifest);write(root/'harness_schedule.json',schedule)
    (root/'pairs.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in schedule['groups']))
    snap=root/'source-snapshot'
    shutil.copytree(REFERENCE/'source-snapshot',snap,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Keep model/trainer/optimizer code identical to the completed stable run.
    # Only transport receives the already-qualified keepalive fix used by eval800.
    qualified=REFERENCE/'checkpoint-evals/step-000800/recovery-20260916-keepalive/source-snapshot'
    assert read(REFERENCE/'checkpoint-evals/step-000800/scores.json')['comparison_ready']
    transport_changes=[]
    for name in ('server.py','sse.py'):
        rel=Path('OpenEnv/src/openenv/core/harness/capture')/name
        before=digest(snap/rel);shutil.copyfile(qualified/rel,snap/rel)
        transport_changes.append({'file':str(rel),'before':before,'after':digest(snap/rel)})
    for name in ('monitor_multi4.py','trackio_multi4.py','supervise_multi4.py'):
        shutil.copyfile(TOOLS/name,snap/'tools'/name)
    shutil.copyfile(REPO/'HuggingEnvs/04-data-agent/eval/checkpoint_evals.py',snap/'HuggingEnvs/04-data-agent/eval/checkpoint_evals.py')
    launch=snap/'tools/launch_multi4_long.sh';text=launch.read_text()
    text=text.replace("assert m['task_count']==1000 and m['pairs_per_cycle']==4000", "assert m['task_count']==1000 and m['pairs_per_cycle']==1000 * len(m['harnesses'])")
    launch.write_text(text)
    log=snap/'tools/launch_multi4_trackio.sh';log.write_text(log.read_text().replace('--watch --online','--watch'))
    supervisor=snap/'tools/supervise_multi4.py';text=supervisor.read_text()
    text=text.replace("'--train-job', train_job, '--watch', '--online']", "'--train-job', train_job, '--watch']\n        if read_json(root / 'run_config.json', {}).get('logging', {}).get('online', True):\n            command.append('--online')")
    supervisor.write_text(text)
    submit=snap/'tools/submit_multi4_long.py';text=submit.read_text()
    text=text.replace("'--partition=hopper-extra'", "'--partition=' + config['resources']['partition']")
    text=text.replace("'--job-name=multi4-long-2b'", "'--job-name=harbor-opencode-only'")
    submit.write_text(text)
    evaluator=root/'checkpoint-evals/eval-source';evaluator.parent.mkdir()
    shutil.copytree(qualified,evaluator,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    write(evaluator/'source_hashes.json',{str(p.relative_to(evaluator)):digest(p) for p in evaluator.rglob('*') if p.is_file() and p!=evaluator/'source_hashes.json'})
    watcher=snap/'tools/launch_multi4_eval_watcher.sh';text=watcher.read_text()
    text=text.replace('--interval "$INTERVAL"', '--protocol "$TRAIN_RUN_ROOT/checkpoint-evals/eval-source/protocol.json" --interval "$INTERVAL"')
    watcher.write_text(text)
    config.update(status='prepared',harnesses=['opencode'],initialization='Fresh pinned Qwen3.5-2B base; independent optimizer and scheduler',
                  source_snapshot=str(snap),frozen_eval_source=str(evaluator),reference_run=str(REFERENCE),reference_job=80608)
    for key in ('restart_of','restart_reason','resume_state','job_id','replaced_by'):
        config.pop(key,None)
    config['harness_versions']={'opencode':prior['harness_versions']['opencode']}
    config['training'].pop('resume_from_checkpoint',None)
    config['training'].pop('budget_reference',None)
    config['training']['soft_max_train_seconds']=82200
    config['resources'].update(partition='hopper-prod',slurm_walltime='24:00:00')
    config['dataset'].update(pairs_per_cycle=1000,rollouts_per_scheduled_cycle=8000,passes_per_cycle=1,
        harness_groups_per_pass={'opencode':1000},schedule_file=str(root/'harness_schedule.json'),
        schedule_sha256=digest(root/'harness_schedule.json'),manifest_sha256=digest(root/'manifest.json'))
    config['evaluation'].update(protocol_file=str(evaluator/'protocol.json'),partition='hopper-prod',
                                 max_active_eval_jobs=1,interval_optimizer_steps=100)
    config['logging'].update(project=root.name,space_id='HuggingEnvs/data-agent-training-comparison-trackio',
        bucket_id='HuggingEnvs/data-agent-training-comparison-trackio',private=False,online=False,
        online_via='Uniform comparison publisher',evaluation_sources=[],local_directory=str(root/'trackio'),
        collector_file=str(snap/'tools/trackio_multi4.py'))
    config['logging'].pop('parent_project',None)
    config['monitoring'].update(stable_after_optimizer_step=10,startup_interval_seconds=120,stable_interval_seconds=600)
    config['monitoring']['support_job_supervisor'].update(status_file=str(root/'supervisor/status.json'),source_file=str(snap/'tools/supervise_multi4.py'))
    config['limitations']=[
        'Matches the last stable Harbor recipe, not every historical recipe used earlier in the resumed reference.',
        'The first pass preserves the reference task order exactly; the same 1000-task single-harness pass repeats if exhausted.',
        '1000 is the optimizer-step target and task-pool size; it does not guarantee full task coverage.',
        'Worker ceiling32; total outstanding rollouts16; eight generations per task; staleness at most4.',
        'Separate eval GPUs, endpoints and node; FSx and E2B quota remain shared.',
        'Checkpoint capture transport uses the already-qualified streaming keepalive repair; sampled tokens and loss are unchanged.',
        'CPU monitoring writes local alerts every2minutes initially and every10minutes after stability; no automatic chat wakeups.',
        'Failed or incomplete evaluations are recorded as incomplete, never converted to final pass@1 scores.']
    write(root/'run_config.json',config)
    hashes={str(p.relative_to(snap)):digest(p) for p in snap.rglob('*') if p.is_file()}
    write(root/'source_hashes.json',hashes)
    excluded={'resume_from_checkpoint','budget_reference'}
    assert {k:v for k,v in prior['training'].items() if k not in excluded} == config['training']
    proof={'prepared':True,'fresh_base':True,'same_training_hyperparameters':True,'same_sampling':config['sampling']==prior['sampling'],
        'first_pass_task_order_identical':True,'training_harnesses':['opencode'],'evaluation_harnesses':read(evaluator/'protocol.json')['harnesses'],
        'reference_job':80608,'source_files':len(hashes),'transport_changes':transport_changes,'live_startup_verified':False}
    write(root/'validation.json',proof)
    env={**os.environ,'TRAIN_RUN_ROOT':str(root),'SLURM_JOB_ID':'preflight','MULTI4_PREFLIGHT_ONLY':'1'}
    with (root/'preflight.log').open('w') as stream:
        subprocess.run(['bash',str(launch)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    env.update(TRAIN_JOB_ID='PREFLIGHT',MULTI4_EVAL_PREFLIGHT_ONLY='1')
    with (root/'eval-preflight.log').open('w') as stream:
        subprocess.run(['bash',str(watcher)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'run':str(root),'validation':proof},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    prepare(parser.parse_args().out.resolve())
