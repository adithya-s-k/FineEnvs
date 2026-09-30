"""Use the qualified LFM services with isolated medium/hard arm inputs."""
import os,sys,json,importlib.util
from dns_fallback import install as install_relay_dns
install_relay_dns()
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
arm=sys.argv.pop(1)
if arm not in ['opencode','multi-harness']:raise ValueError('Unknown arm')
basefile=HERE.parent/'lfm25/run.py'
spec=importlib.util.spec_from_file_location('lfm_base',basefile);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.OUT=ROOT/'experiments/lfm25-medium-hard1000-20260921'/arm
base.C=json.loads((base.OUT/'config.json').read_text());base.TRAIN_SPLIT=base.C['dataset']['split']
original_start=base.rt.start

def start(command, log, env=None):
    command=list(command)
    policy_path=base.OUT.parent/'transport_policy.json'
    policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
    expose=os.environ.get('LFM_EXPOSE') or policy.get('expose')
    if '--expose' in command and expose:
        if expose not in ('gradio','cloudflare','direct'):raise ValueError('Unknown exposure transport')
        command[command.index('--expose')+1]=expose
        env={**(os.environ if env is None else env), 'TUNNEL_TRANSPORT_PROTOCOL':'http2',
             'TUNNEL_LOGFILE':str(Path(log).with_suffix('.tunnel.log'))}
    return original_start(command,log,env)
base.rt.start=start

original_command=base.train_command

def command(output,engine,server,steps,resume=None,smoke=False):
    result=original_command(output,engine,server,steps,resume,smoke);result[1]=HERE/'train_entry.py'
    result[result.index('--harnesses')+1]='+'.join(base.C['training_harnesses'])
    return result
base.train_command=command

def train(output,engine,server,smoke):
    from checkpoint_artifacts import digest,finalize_saved,verify_ready
    gpus=os.environ['CUDA_VISIBLE_DEVICES'].split(',');remote=os.environ.get('LFM_REMOTE_SERVICE')
    if not remote and len(gpus)<2:raise ValueError('Separate inference/training GPUs required')
    if not smoke:
        from production import check_qualification
        check_qualification(base.OUT, HERE)
    trials=Path(remote)/'trials' if remote else output/'trials'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus[0] if remote else gpus[1], 'TRACKIO_DIR':str(output/'trackio'),'TRACKIO_STORAGE_MODE':'jsonl','CURRICULUM_GROUP_LIMIT':'2000','CURRICULUM_AUDIT':str(output/'audit'),'CURRICULUM_SCHEDULE_SHA256':digest(base.OUT/'harness_schedule.json'),'LFM_TRIALS_DIR':str(trials)}
    env['PYTHONPATH']=str(HERE)+os.pathsep+env['PYTHONPATH']
    for steps in ([2,4] if smoke else [base.C['training']['max_steps']]):
        resume=output/'run/checkpoint-2' if smoke and steps==4 else None
        if not smoke and os.environ.get('LFM_RESUME_CHECKPOINT'):
            resume=Path(os.environ['LFM_RESUME_CHECKPOINT']).resolve()
            verify_ready(resume)
        if resume:env['CURRICULUM_RESUME_STATE']=str(resume/'curriculum_state.json')
        cmd=command(output,engine,server,steps,resume,smoke)
        base.rt.write_json(output/f'train-command-{steps}.json',{'command':list(map(str,cmd))})
        p=base.rt.start(cmd,output/f'train-{steps}.log',env)
        if p.wait():raise RuntimeError(f'Training phase {steps} failed')
        checkpoints=list((output/'run').glob('checkpoint-*'))
        ckpt=max(checkpoints,key=lambda p:int(p.name.split('-')[-1]))
        finalize_saved(ckpt);verify_ready(ckpt)
    if not smoke:return ckpt
    metrics=[json.loads(x) for x in (output/'audit/metrics.jsonl').read_text().splitlines()]
    import math
    if not any(m.get('grad_norm',0)>0 for m in metrics):raise RuntimeError('No nonzero gradient')
    if any(not math.isfinite(m.get('grad_norm',0)) for m in metrics):raise RuntimeError('Nonfinite gradient')
    rewards=[json.loads(p.read_text()) for p in (output/'audit/efficiency').glob('*.json')]
    if not any(r['count_verified'] and (r['bonus'] or 0)>0 for r in rewards):raise RuntimeError('No verified positive efficiency bonus')
    settled=set(base.rt.read(output/'run/checkpoint-2/curriculum_state.json')['settled_groups'])
    receipts=[json.loads(x) for x in (output/'audit/optimizer_rollouts.jsonl').read_text().splitlines()]
    if settled & {r['group_id'] for e in receipts if e['step']>2 for r in e['rollouts']}:raise RuntimeError('Committed group replay')
    base.rt.write_json(output/'training_smoke_verified.json',{'steps':4,'nonzero_updates':sum(m.get('grad_norm',0)>0 for m in metrics),'reward_records':len(rewards),'verified_counts':sum(r['count_verified'] for r in rewards),'checkpoint_resume':True,'committed_group_replay':False})
    return ckpt
base.train=train
if __name__=='__main__':
    from eval_recovery import install as install_eval_recovery
    install_eval_recovery(base)
    base.main()
    if sys.argv[1]=='train-smoke':
        from hashlib import sha256
        out=base.OUT/('train-smoke-'+os.environ.get('SLURM_JOB_ID','manual'))
        proof=json.loads((out/'qualification_passed.json').read_text())
        proof.update(config_sha256=sha256((base.OUT/'config.json').read_bytes()).hexdigest(),schedule_sha256=sha256((base.OUT/'harness_schedule.json').read_bytes()).hexdigest(),reward_sha256=sha256((HERE/'reward.py').read_bytes()).hexdigest())
        (out/'qualification_passed.json').write_text(json.dumps(proof,indent=2)+'\n')
