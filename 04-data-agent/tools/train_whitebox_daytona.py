"""Isolated synchronous GRPO comparison with actual token and loss-mask audits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from whitebox_bash import white_box_bash_env
from eval_whitebox_native import MODEL, REVISION, SYSTEM
from generation_routing import generation_group_size


class AuditedTrainer(GRPOTrainer):
    """Observe the native loop without replacing its generation, rewards or loss."""

    def _generate_single_turn(self, prompt_ids, *args, **kwargs):
        completion_ids, logprobs = super()._generate_single_turn(prompt_ids,*args,**kwargs)
        records=getattr(self,'_captured_calls',[])
        for prompt,completion,lps in zip(prompt_ids,completion_ids,logprobs,strict=True):
            assert len(completion)==len(lps) and all(math.isfinite(p) for p in lps)
            records.append({'prompt_ids':prompt.copy(),'completion_ids':completion.copy(),'logprobs':lps.copy()})
        self._captured_calls=records
        return completion_ids,logprobs

    def _generate(self,*args,**kwargs):
        self._captured_calls=[]
        return super()._generate(*args,**kwargs)

    def _tool_call_loop(self,prompts,prompt_ids,*args,**kwargs):
        result=super()._tool_call_loop(prompts,prompt_ids,*args,**kwargs)
        masks,_,completions,lps,_,_,_=result
        evidence=Path(self.args.output_dir)/'capture_audit'
        evidence.mkdir(exist_ok=True)
        (evidence/f'step-{self.state.global_step}.json').write_text(json.dumps({
            'prompt_ids':prompt_ids,'completions':completions,'masks':masks,'logprobs':lps,
            'calls':self._captured_calls})+'\n')
        from whitebox_tito import audit_rows
        checks=audit_rows(prompt_ids,completions,masks,lps,self._captured_calls)
        path=Path(self.args.output_dir)/'token_audit.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps({'step':self.state.global_step,'rows':checks,'calls':len(self._captured_calls)})+'\n')
        return result

    def compute_loss(self,model,inputs,*args,**kwargs):
        if 'tool_mask' in inputs:
            mask=inputs['completion_mask']*inputs['tool_mask']
            assert torch.isfinite(inputs['sampling_per_token_logps'][mask.bool()]).all()
            assert set(inputs['tool_mask'].unique().tolist())<={0,1}
        return super().compute_loss(model,inputs,*args,**kwargs)


def digest(model):
    h=hashlib.sha256()
    for name,param in model.named_parameters():
        if param.requires_grad:
            h.update(name.encode());h.update(param.detach().flatten()[:256].float().cpu().numpy().tobytes())
    return h.hexdigest()


class SmokeEvidence(TrainerCallback):
    def __init__(self,path): self.path=path
    def on_train_begin(self,args,state,control,model=None,**kwargs):
        self.initial=digest(model)
        self.initial_step=state.global_step
    def on_log(self,args,state,control,logs=None,**kwargs):
        for key,value in (logs or {}).items():
            if isinstance(value,float): assert math.isfinite(value), f'nonfinite metric: {key}'
        with (self.path/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'step':state.global_step,**(logs or {})})+'\n')
    def on_train_end(self,args,state,control,model=None,**kwargs):
        result={'initial_step':self.initial_step,'final_step':state.global_step,
                'initial_weight_digest':self.initial,'final_weight_digest':digest(model)}
        result['weights_changed']=result['initial_weight_digest']!=result['final_weight_digest']
        (self.path/f'optimizer_evidence_from_{self.initial_step}.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--server',required=True)
    p.add_argument('--vllm-url',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-steps',type=int,default=1000)
    p.add_argument('--save-steps',type=int,default=50)
    p.add_argument('--resume-from-checkpoint')
    p.add_argument('--max-train-seconds',type=float,default=0)
    p.add_argument('--checkpoint-max-seconds',type=float,default=0)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    schedule=json.loads((args.run/'reference_schedule.json').read_text())
    rows=[{'prompt':[{'role':'system','content':SYSTEM},{'role':'user','content':'Solve the task.'}],
           'split':'train','index':g['task_index']} for g in schedule['groups'][:schedule['task_count']]]
    config=GRPOConfig(
        output_dir=str(args.output_dir),model_init_kwargs={'revision':REVISION,'dtype':'bfloat16'},
        learning_rate=3e-6,lr_scheduler_type='constant',warmup_steps=0,beta=0.0,loss_type='dapo',
        num_generations=8,per_device_train_batch_size=1,gradient_accumulation_steps=8,
        max_steps=args.max_steps,max_completion_length=16384,max_tool_calling_iterations=16,
        temperature=0.8,top_p=1.0,top_k=0,chat_template_kwargs={'enable_thinking':False},
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        bf16=True,optim='paged_adamw_8bit',max_grad_norm=1.0,seed=0,shuffle_dataset=False,
        use_vllm=True,vllm_mode='server',vllm_server_base_url=args.vllm_url,
        vllm_max_model_length=131072,vllm_server_timeout=900,
        vllm_group_port=49000+int(os.environ.get('SLURM_JOB_ID','0'))%1000,
        generation_kwargs={'max_tokens':16384},
        save_strategy='steps',save_steps=args.save_steps,save_total_limit=None,
        logging_steps=1,log_completions=True,num_completions_to_print=1,
        report_to='trackio',project='daytona-whitebox-qwen35-2b',
        run_name=f'whitebox-{os.environ.get("RUN_OWNER", os.environ.get("SLURM_JOB_ID", "local"))}',
        trackio_space_id=None,trackio_static_space_id=False,
    )
    (args.output_dir/'comparison_config.json').write_text(json.dumps(config.to_dict(),indent=2,default=str)+'\n')
    trainer=AuditedTrainer(model=MODEL,args=config,train_dataset=Dataset.from_list(rows),reward_funcs=[],
        environment_factory=white_box_bash_env(args.server,split='train',toolsets='bash,seta',step_limit=17),
        callbacks=[SmokeEvidence(args.output_dir)])
    from training_audit import CheckpointReadyCallback
    trainer.add_callback(CheckpointReadyCallback(MODEL,REVISION))
    if args.max_train_seconds:
        from training_audit import WallTimeCallback
        trainer.add_callback(WallTimeCallback(args.max_train_seconds))
    if args.checkpoint_max_seconds:
        from training_audit import PeriodicCheckpointCallback
        trainer.add_callback(PeriodicCheckpointCallback(args.checkpoint_max_seconds))
    # Check the response's engine IDs and sampled logprob IDs before GRPO consumes them.
    generate=trainer.vllm_generation.generate
    def checked_generate(*a,**kw):
        # Native server generation groups duplicated initial prompts by G. After tools,
        # histories differ and need one continuation each; grouping those would replace
        # seven trajectories' contexts with the first one's context.
        prompts=kw.get('prompts',a[0] if a else None)
        kw={**kw,'num_generations':generation_group_size(prompts,kw.get('num_generations',1))}
        output=generate(*a,**kw)
        returned_prompts,completions,logprobs,token_ids=output
        assert returned_prompts==prompts
        for ids,lps,lp_ids in zip(completions,logprobs,token_ids,strict=True):
            assert len(ids)==len(lps)==len(lp_ids)
            assert all(len(ids_at_pos)==1 and ids_at_pos[0]==tok for tok,ids_at_pos in zip(ids,lp_ids,strict=True))
        return output
    trainer.vllm_generation.generate=checked_generate
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(str(args.output_dir/'final'))


if __name__=='__main__': main()
