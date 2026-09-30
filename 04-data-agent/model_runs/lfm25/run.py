"""LFM multi-harness serving, evaluation and optimizer qualification."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = Path(os.environ.get('LFM_RUN_ROOT', REPO / 'experiments/lfm25-multi4-20260921'))
FROZEN = OUT / 'runtime'
os.environ['REPRO_ROOT'] = str(FROZEN)
sys.path.insert(0, str(REPO/'HuggingEnvs/04-data-agent/hf'))
from deploy import credentials
os.environ.update(credentials(REPO/'experiments/.env'))
sys.path.insert(0, str(FROZEN/'hf/runtime'))
import hard_curriculum_job as rt
from checkpoint_artifacts import verify_ready, finalize_saved
C = json.loads((HERE/'config.json').read_text())
PY = REPO/'.venv312/bin/python'
ENV_PY = REPO/'OpenEnv/.venv/bin/python'
TRAIN_SPLIT = C['dataset']['split']
TEST_SPLIT = FROZEN/'data/test'


def services(output, model, training):
    if str(model) != C['model']:
        marker=verify_ready(model)
        if marker['base_model'] != C['model'] or marker['base_revision'] != C['model_revision']:
            raise ValueError('Checkpoint belongs to another model or revision')
    port = 20000 + int(os.environ.get('SLURM_JOB_ID','0')) % 1000 * 32
    engine, server = f'http://127.0.0.1:{port}', f'http://127.0.0.1:{port+10}'
    gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':gpus[0], 'VLLM_SERVER_DEV_MODE':'1',
           'VLLM_USE_DEEP_GEMM':'0', 'VLLM_USE_FLASHINFER_SAMPLER':'0'}
    command = [PY,'-m','vllm.entrypoints.openai.api_server','--model',model,
        '--served-model-name',C['model'],'--port',str(port),'--dtype','bfloat16',
        '--generation-config','vllm','--max-model-len','131072','--gpu-memory-utilization','0.85',
        '--no-enable-prefix-caching','--enable-auto-tool-choice','--tool-call-parser','lfm2',
        '--reasoning-parser','qwen3','--default-chat-template-kwargs','{"preserve_thinking":true}',
        '--return-tokens-as-token-ids','--logprobs-mode','processed_logprobs','--enforce-eager',
        '--override-generation-config','{"temperature":0.8,"top_p":1.0,"top_k":-1,"repetition_penalty":1.0,"max_tokens":4096}']
    if model == C['model']:command += ['--revision',C['model_revision']]
    if training:command += ['--weight-transfer-config','{"backend":"nccl"}']
    elif len(gpus)>1:
        command += ['--tensor-parallel-size','1','--data-parallel-size',str(len(gpus)),
                    '--data-parallel-rpc-port',str(port+3)]
        env['CUDA_VISIBLE_DEVICES']=os.environ['CUDA_VISIBLE_DEVICES']
    processes=[]
    try:
        p=rt.start(command,output/'vllm.log',env);processes.append(p)
        rt.ready(engine+'/health',p,seconds=1200)
        from openenv.core.harness.capture.validate_llm import validate_llm
        report=validate_llm(engine+'/v1',C['model'])
        if not report.trainable:raise RuntimeError(f'Token/logprob probe failed: {report}')
        env={**os.environ,'MAX_CONCURRENT_ENVS':'64','OPENENV_HARBOR_TRIALS_DIR':str(output/'trials'),
             'OPENENV_HARBOR_AGENT_VERSIONS':json.dumps(C['harness_versions'])}
        p=rt.start([ENV_PY,'-m','openenv.cli','harbor','serve','--dataset',TRAIN_SPLIT,
            '--dataset',TEST_SPLIT,'--llm-url',engine+'/v1','--model',C['model'],
            '--port',str(port+10),'--capture-port',str(port+11),'--expose','gradio',
            '--max-output-tokens','4096'],output/'openenv.log',env);processes.append(p)
        rt.ready(server+'/health',p,seconds=1200)
        import re,httpx
        match=re.search(r'capture\s+:\d+\s+->\s+(https://\S+)',(output/'openenv.log').read_text())
        if not match:raise RuntimeError('Capture forwarding URL missing')
        rt.ready(match[1]+'/health',p,seconds=120)
        local=httpx.get(f'http://127.0.0.1:{port+11}/health').raise_for_status().json()
        remote=httpx.get(match[1]+'/health',timeout=30).raise_for_status().json()
        if not local.get('instance') or local['instance']!=remote.get('instance'):raise RuntimeError('Capture identity mismatch')
        rt.write_json(output/'services.json',{'engine':engine,'server':server,'model':C['model'],
            'revision':C['model_revision'],'capture_instance_verified':True,'tito_probe':True})
        return processes,engine,server
    except BaseException:
        rt.stop(processes);raise


def eval_command(output, engine, server, smoke):
    return [PY,FROZEN/'eval/eval_concurrent.py','--server',server,'--vllm-url',engine+'/v1',
        '--model',C['model'],'--harnesses',','.join(C['harnesses']),'--split',TEST_SPLIT,
        '--indices','0,1' if smoke else '@'+str(FROZEN/'inputs/test_indices.txt'),
        '--repeat','1','--temperature','0.8','--reward-key','correctness,reward','--sandbox','e2b',
        '--agent-timeout','600','--agent-step-limit','17','--max-retries','3',
        '--trace-dir',output/'traces','--capture-dir',output/'captures','--out',output/'results.json',
        '--concurrency','4' if smoke else '50','--server-concurrency','4' if smoke else '50',
        '--sandbox-concurrency','4' if smoke else '50']


def evaluate(output, engine, server, smoke):
    command=eval_command(output,engine,server,smoke)
    for attempt in range(3):
        process=rt.start(command+(['--resume'] if (output/'traces/eval_config.json').exists() else []),output/'eval.log')
        if process.wait() not in (0,2):raise RuntimeError('Eval process failed')
        score=rt.score_eval(C,output,smoke)
        if score['complete']:
            rt.write_json(output/('eval_smoke_verified.json' if smoke else 'canonical_scores.json'),score)
            return score
    raise RuntimeError('Incomplete eval; missing grades are not zeros')


def train_command(output, engine, server, steps, resume=None, smoke=False):
    command=[PY,HERE/'train_entry.py','--server',server,'--vllm-url',engine,'--model',C['model'],
        '--model-revision',C['model_revision'],'--split',TRAIN_SPLIT,'--harnesses','+'.join(C['harnesses']),
        '--harness-schedule',OUT/'harness_schedule.json','--task-indices','@'+str(OUT/'indices.txt'),
        '--sandbox','e2b','--learning-rate','3e-6','--num-generations','8','--max-inflight','32',
        '--max-staleness','4','--grad-accum','4','--atomic-rollouts','--max-outstanding-rollouts','16',
        '--max-row-tokens','131072','--per-device-batch-size','4','--reward-key','correctness,reward',
        '--agent-step-limit','17','--agent-timeout','600','--token-budget','40960',
        '--max-completion-length','16384','--temperature','0.8','--top-p','1','--top-k','0',
        '--seed','0','--dtype','bfloat16','--save-steps','2' if smoke else '50',
        '--max-steps',str(steps),'--checkpoint-max-seconds','3600','--audit-dir',output/'audit',
        '--output-dir',output/'run','--project',C['logging']['project'],'--run-name',C['run_name']+('-smoke' if smoke else '')]
    if resume:command += ['--resume-from-checkpoint',resume]
    return command


def train(output, engine, server, smoke):
    from checkpoint_artifacts import digest
    gpus=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    remote = bool(os.environ.get('LFM_REMOTE_SERVICE'))
    if len(gpus)<2 and not remote:raise RuntimeError('Training requires separate inference and trainer GPUs')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus[0] if remote else gpus[1],'TRACKIO_DIR':str(output/'trackio'),
         'TRACKIO_STORAGE_MODE':'jsonl','CURRICULUM_GROUP_LIMIT':'4000','CURRICULUM_AUDIT':str(output/'audit'),
         'CURRICULUM_SCHEDULE_SHA256':digest(OUT/'harness_schedule.json')}
    for steps in ([2,4] if smoke else [1000]):
        resume=output/'run/checkpoint-2' if smoke and steps==4 else None
        if resume:
            verify_ready(resume)
            env['CURRICULUM_RESUME_STATE']=str(resume/'curriculum_state.json')
        command=train_command(output,engine,server,steps,resume,smoke)
        rt.write_json(output/f'train-command-{steps}.json',{'command':list(map(str,command))})
        p=rt.start(command,output/f'train-{steps}.log',env)
        if p.wait():raise RuntimeError(f'Training phase {steps} failed')
        checkpoint=output/f'run/checkpoint-{steps}'
        finalize_saved(checkpoint);verify_ready(checkpoint)
    if smoke:
        rows=[json.loads(x) for x in (output/'audit/metrics.jsonl').read_text().splitlines()]
        updates=[r for r in rows if r.get('grad_norm',0)>0]
        if not updates:raise RuntimeError('Smoke had no nonzero optimizer gradient; not qualified')
        import math
        if any(not math.isfinite(r['grad_norm']) or not math.isfinite(r.get('loss',0)) for r in updates):
            raise RuntimeError('Nonfinite optimizer metrics')
        settled=set(rt.read(output/'run/checkpoint-2/curriculum_state.json')['settled_groups'])
        receipts=[json.loads(x) for x in (output/'audit/optimizer_rollouts.jsonl').read_text().splitlines()]
        resumed={r['group_id'] for entry in receipts if entry['step']>2 for r in entry['rollouts']}
        if settled & resumed:raise RuntimeError('Resume replayed an already committed task group')
        rt.write_json(output/'training_smoke_verified.json',{'steps':4,'nonzero_updates':len(updates),
            'checkpoint_resume':True,'committed_group_replay':False,'final_checkpoint':str(checkpoint)})
    return checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['eval-smoke','train-smoke','eval','train','server-smoke'])
    p.add_argument('--model-path');args=p.parse_args()
    rt.verify_bundle()
    output=OUT/(args.phase+'-'+os.environ.get('SLURM_JOB_ID','manual'));output.mkdir(parents=True,exist_ok=False)
    rt.write_json(output/'configuration.json',C)
    processes=[]
    try:
        training=args.phase.startswith('train')
        if args.phase == 'server-smoke':
            import time,socket
            processes,engine,server=services(output,C['model'],True)
            rt.write_json(output/'remote_ready.json',{'engine':engine,'server':server,'host':socket.gethostname()})
            deadline=time.monotonic()+3*3600
            while not (output/'STOP').exists() and time.monotonic()<deadline:time.sleep(5)
            return
        remote=os.environ.get('LFM_REMOTE_SERVICE')
        if remote:
            import time,socket
            receipt=Path(remote)/'remote_ready.json'
            deadline=time.monotonic()+1500
            while not receipt.exists() and time.monotonic()<deadline:time.sleep(5)
            info=rt.read(receipt)
            if info['host']!=socket.gethostname():raise RuntimeError('Remote smoke services must be on the same allocated node')
            engine,server=info['engine'],info['server']
        else:
            processes,engine,server=services(output,args.model_path or C['model'],training)
        if training:
            checkpoint=train(output,engine,server,args.phase.endswith('smoke'))
            if args.phase == 'train-smoke':
                rt.stop(processes); processes=[]
                if remote:(Path(remote)/'STOP').touch()
                proof=output/'checkpoint-eval'; proof.mkdir()
                processes,engine,server=services(proof,str(checkpoint),False)
                evaluate(proof,engine,server,True)
                rt.write_json(output/'qualification_passed.json',{'optimizer_save_resume':True,
                    'reloaded_checkpoint_eval':True,'model':C['model'],'revision':C['model_revision'],
                    'config_sha256':__import__('hashlib').sha256((HERE/'config.json').read_bytes()).hexdigest()})
        else:evaluate(output,engine,server,args.phase.endswith('smoke'))
    finally:
        rt.stop(processes)
        if os.environ.get('LFM_REMOTE_SERVICE'):
            (Path(os.environ['LFM_REMOTE_SERVICE'])/'STOP').touch()

if __name__=='__main__':main()
