"""Resumable pass@1 using frozen TRL's actual token-preserving bash/SETA loop."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import inspect
import json
import math
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
from transformers import AutoProcessor
from transformers.utils import get_json_schema
from trl.chat_template_utils import (
    add_response_schema, get_training_chat_template,
    is_chat_template_prefix_preserving, parse_response,
)
from trl.trainer.grpo_trainer import GRPOTrainer
from whitebox_bash import white_box_bash_env

MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
SYSTEM = (
    'You are a terminal agent working in a sandbox. Use the available tools to inspect the '
    'filesystem and solve the task. Work step by step: look before you act. When you are confident, '
    'call submit_solution with the final answer and nothing else -- not the command that would '
    'produce it.'
)


class NativeLoop:
    _get_tool_suffix_ids = GRPOTrainer._get_tool_suffix_ids
    _tool_call_loop = GRPOTrainer._tool_call_loop

    def __init__(self, env, processor, url, model, deadline):
        self.processing_class = processor
        self._tokenizer = processor.tokenizer
        self._is_vlm = True
        self.chat_template = (
            None if is_chat_template_prefix_preserving(processor)
            else get_training_chat_template(processor)
        )
        self.chat_template_kwargs = {'enable_thinking': False}
        methods = {
            n:m for n,m in inspect.getmembers(env,predicate=inspect.ismethod)
            if not n.startswith('_') and n not in {'reset','get_reward'}
        }
        self.tools = [get_json_schema(m) for m in methods.values()]
        self._sync_tool_dicts = [methods]
        self._async_tool_dicts = [{}]
        self.max_tool_calling_iterations = 16  # first generation + 16 continuations = 17 calls
        self.max_completion_length = 16384  # includes masked tool-result tokens, as in sync training
        self.use_vllm = True
        self.vllm_mode = 'server'
        self.model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=131072)))
        self.url, self.model_name, self.deadline = url.rstrip('/'), model, deadline
        self.client = httpx.Client(timeout=120)
        self.calls = []
        self.stop_reason = None

    def _generate_single_turn(self, prompts, images, multimodal_fields, has_tool_images=False):
        assert len(prompts)==1 and not images and not has_tool_images
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            self.stop_reason = 'episode_deadline'
            return [[]], [[]]
        prompt=prompts[0]
        if self.calls:
            parent=self.calls[-1]['prompt_ids']+self.calls[-1]['completion_ids']
            assert prompt[:len(parent)]==parent, 'native loop lost the sampled prefix'
        try:
            response=self.client.post(self.url+'/completions',json={
            'model':self.model_name,'prompt':prompt,'max_tokens':min(4096,131072-len(prompt)),
            'temperature':0.8,'top_p':1.0,'top_k':-1,'n':1,'logprobs':0,
            'return_token_ids':True,'return_tokens_as_token_ids':True,
            },timeout=remaining).raise_for_status().json()
        except httpx.ReadTimeout:
            if time.monotonic() < self.deadline:
                raise  # An earlier transport failure remains an ungraded attempt.
            self.stop_reason = 'episode_deadline'
            # The native loop already accepts an empty continuation on truncation.
            # Stop without inventing tokens; grade work done within the episode budget.
            return [[]], [[]]
        choice=response['choices'][0]
        ids=choice['token_ids']
        lp=choice['logprobs']
        assert choice['prompt_token_ids']==prompt, 'engine prompt differs from supplied token IDs'
        assert ids and len(ids)==len(lp['tokens'])==len(lp['token_logprobs'])
        assert lp['tokens']==[f'token_id:{i}' for i in ids], 'sampled token/logprob pairing mismatch'
        assert all(math.isfinite(p) for p in lp['token_logprobs'])
        self.calls.append({'prompt_ids':prompt.copy(),'completion_ids':ids.copy(),
                           'logprobs':lp['token_logprobs'].copy(),'finish_reason':choice['finish_reason']})
        return [ids], [lp['token_logprobs']]

    def run(self, prompt):
        prompts=[[{'role':'system','content':SYSTEM},{'role':'user','content':'Solve the task.'+prompt}]]
        tokenized=self.processing_class.apply_chat_template(
            prompts[0], tools=self.tools, add_generation_prompt=True,tokenize=True,
            chat_template=self.chat_template,return_dict=False,**self.chat_template_kwargs,
        )
        ids=tokenized[0] if isinstance(tokenized[0],list) else tokenized
        generated,lps=self._generate_single_turn([ids],None,{})
        completions=[[parse_response(self._tokenizer,generated[0],prefix=ids)]]
        masks,completions,generated,lps,tools,failures,_=self._tool_call_loop(
            copy.deepcopy(prompts),[ids],generated,completions,lps,None,{},
        )
        mask,completion,logprobs=masks[0],generated[0],lps[0]
        all_sampled=[t for call in self.calls for t in call['completion_ids']]
        all_lp=[p for call in self.calls for p in call['logprobs']]
        supervised=[t for t,m in zip(completion,mask,strict=True) if m]
        supervised_lp=[p for p,m in zip(logprobs,mask,strict=True) if m]
        checks={
            'sampled_ids_preserved':supervised==all_sampled[:len(supervised)],
            'sampled_logprobs_preserved':supervised_lp==all_lp[:len(supervised_lp)],
            'tool_context_masked':all(p==0.0 for p,m in zip(logprobs,mask,strict=True) if not m),
            'loss_mask_binary':set(mask)<={0,1},
            'real_logprobs':bool(supervised_lp) and any(p<0 for p in supervised_lp),
            'finite_logprobs':all(math.isfinite(p) for p in logprobs),
        }
        assert all(checks.values()), checks
        return {'prompt_ids':ids,'completion_ids':completion,'logprobs':logprobs,'loss_mask':mask,
                'calls':self.calls,'messages':completions,'tito_checks':checks,'tito_pass':True,
                'turns':len(self.calls),'tool_calls':tools,'tool_failures':failures,
                'stop_reason':self.stop_reason,'budget_policy':'600s episode; empty native continuation at deadline; no fabricated token IDs'}


def episode(args,index,processor):
    start=time.monotonic()
    env=white_box_bash_env(args.server,toolsets='bash,seta',step_limit=17,timeout_s=600)()
    loop=None
    rec={'index':index,'reward':None,'tito_pass':False,'harness':'whitebox_seta','pass_k':1}
    try:
        prompt=env.reset(split='test',index=index)
        loop=NativeLoop(env,processor,args.vllm_url,args.model,time.monotonic()+600)
        capture=loop.run(prompt)
        reward=env.get_reward()
        if not math.isfinite(reward):
            raise RuntimeError('episode ungraded')
        assert reward in (0,1)
        path=args.out/'captures'/f'{index:03d}-{uuid.uuid4().hex}.json'
        path.write_text(json.dumps(capture)+'\n')
        rec.update({k:capture[k] for k in ('tito_pass','turns','tool_calls','tool_failures','stop_reason')})
        rec.update(reward=reward,capture_file=str(path),supervised_tokens=sum(capture['loss_mask']),
                   forwarded_tokens=len(capture['prompt_ids'])+len(capture['completion_ids']))
    except Exception as exc:
        rec['error_type']=type(exc).__name__
        # Local exception diagnostic; do not put prompts, provider bodies or signed URLs in reports.
        import traceback
        (args.out/f'error-{index:03d}-{uuid.uuid4().hex}.txt').write_text(traceback.format_exc())
    finally:
        if env._session is not None:
            try: env.get_reward()
            except Exception: pass
        env._mcp.close()
        if loop is not None: loop.client.close()
    rec['elapsed_s']=time.monotonic()-start
    return rec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--server',required=True)
    p.add_argument('--vllm-url',required=True)
    p.add_argument('--model',default=MODEL)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--max-new-rollouts',type=int,default=0)
    p.add_argument('--require-complete',action='store_true')
    p.add_argument('--ramp',action='store_true',help='Measure 8 slots, then 32, then 100; stop if a ramp grades under 90%.')
    args=p.parse_args()
    args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'captures').mkdir(exist_ok=True)
    config={'model':args.model,'revision':REVISION,'dataset':str(args.run/'datasets/test'),
            'vllm_url':args.vllm_url,'server':args.server,'pass_k':1,'temperature':0.8,'top_p':1.0,
            'max_output_tokens_per_call':4096,'max_episode_completion_tokens':16384,'max_model_calls':17,
            'toolsets':['bash','seta'],'native_loop_source':inspect.getfile(GRPOTrainer),
            'source_manifest':str(args.run/'source_input_hashes.json')}
    manifest=args.out/'eval_config.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==config, 'resume config mismatch'
    else: manifest.write_text(json.dumps(config,indent=2)+'\n')
    records=args.out/'attempts.jsonl'
    selected={}
    if records.exists():
        for line in records.read_text().splitlines():
            r=json.loads(line)
            if r.get('reward') in (0,1) and r.get('tito_pass'):
                selected.setdefault(r['index'],r)
    indices=list(map(int,(args.run/'test_indices.txt').read_text().replace(',',' ').split()))
    processor=AutoProcessor.from_pretrained(MODEL,revision=REVISION)
    tokenizer=processor.tokenizer
    if not (getattr(tokenizer,'response_template',None) or getattr(tokenizer,'response_schema',None)):
        processor=add_response_schema(processor)
    start=time.monotonic()
    phases=[(8,8),(32,32),(100,0)] if args.ramp else [(args.concurrency,args.max_new_rollouts)]
    ramp=[]
    for concurrency,limit in phases:
        pending=[i for i in indices if i not in selected]
        if limit: pending=pending[:limit]
        before=len(selected);phase_start=time.monotonic()
        with records.open('a') as out, concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures=[pool.submit(episode,args,i,processor) for i in pending]
            for f in concurrent.futures.as_completed(futures):
                r=f.result()
                out.write(json.dumps(r)+'\n');out.flush()
                if r['reward'] in (0,1) and r['tito_pass']: selected.setdefault(r['index'],r)
                print(json.dumps({'graded':len(selected),'latest':r}),flush=True)
        ramp.append({'concurrency':concurrency,'attempted':len(pending),'graded':len(selected)-before,
                     'elapsed_s':time.monotonic()-phase_start})
        (args.out/'ramp.json').write_text(json.dumps(ramp,indent=2)+'\n')
        if args.ramp and len(selected)-before < 0.9*len(pending): break
    manifest=json.loads((args.run/'test_manifest.json').read_text())
    difficulty={}
    for level in ('easy','medium','hard'):
        subset=[r for i,r in selected.items() if manifest['tasks'][i]['difficulty']==level]
        difficulty[level]={'graded':len(subset),'correct':sum(r['reward'] for r in subset)}
    report={'metric':'pass@1','complete':len(selected)==250,'graded':len(selected),'expected':250,
            'correct':sum(r['reward'] for r in selected.values()),
            'pass_at_1':sum(r['reward'] for r in selected.values())/len(selected) if selected else None,
            'graded_indices':sorted(selected),'difficulty':difficulty,'concurrency':args.concurrency,
            'phase_elapsed_s':time.monotonic()-start,'ramp':ramp,
            'tito_pass':bool(selected) and all(r['tito_pass'] for r in selected.values())}
    (args.out/'scores.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return 2 if args.require_complete and not report['complete'] else 0


if __name__=='__main__':
    raise SystemExit(main())
