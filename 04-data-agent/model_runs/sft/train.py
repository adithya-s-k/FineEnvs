"""TRL SFT for recorded tool-use responses, with durable epoch saves."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys

from prepare import MODEL, MODEL_REVISION, digest
from eval_policy import completed_epoch, should_evaluate
from local_logging import configure as configure_local_logging, materialize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--resume', type=Path)
    p.add_argument('--learning-rate', type=float, default=3e-6)
    p.add_argument('--epochs', type=float, default=2)
    p.add_argument('--gradient-accumulation', type=int, default=8)
    p.add_argument('--save-steps', type=int, default=50)
    p.add_argument('--eval-every', type=int, help='Optional smoke override; production evaluates completed epochs only')
    p.add_argument('--eval-concurrency', type=int, default=100)
    args = p.parse_args()
    root, data, out = args.workspace.resolve(), args.data.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    configure_local_logging(out)
    materialize()
    sys.path.insert(0, str(root / 'HuggingEnvs/04-data-agent/train'))
    from checkpoint_artifacts import mark_saved, finalize_saved
    import torch
    from datasets import Dataset, Features, Json
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    proof = json.loads((data / 'preparation.json').read_text())
    model_id, model_revision = proof.get('model', MODEL), proof.get('model_revision', MODEL_REVISION)
    assert proof['train_sha256'] == digest(data / 'train.jsonl')
    assert proof['tokens_sha256'] == digest(data / 'tokens.jsonl')
    assert proof['matched_sft_tasks'] > 0
    if not proof.get('include_test_overlap'):
        assert all(proof[key] == 0 for key in ['test_task_overlap', 'test_notebook_overlap', 'test_question_overlap'])
    with (data / 'train.jsonl').open(encoding='utf-8') as source:
        rows = [json.loads(line) for line in source]
    assert len(rows) == proof['matched_sft_tasks'] == len({row['task_id'] for row in rows})
    with (data / 'tokens.jsonl').open(encoding='utf-8') as source:
        reference = {r['task_id']: r for r in map(json.loads, source)}
    if args.smoke:
        names = set(json.loads((data / 'smoke_task_ids.json').read_text()))
        rows = [r for r in rows if r['task_id'] in names]
    pretokenized = proof.get('dataset_format') == 'pretokenized_completion'
    prompt_completion = proof.get('dataset_format') == 'prompt_completion'
    if pretokenized:
        dataset = Dataset.from_list([{k: reference[r['task_id']][k] for k in ['input_ids', 'labels']} for r in rows])
    elif prompt_completion:
        columns = ['prompt', 'completion', 'tools', 'chat_template_kwargs']
        dataset = Dataset.from_list([{k: r[k] for k in columns} for r in rows],
                                    features=Features({k: Json() for k in columns}))
    else:
        dataset = Dataset.from_list([{k: r[k] for k in ['messages', 'tools', 'chat_template_kwargs']} for r in rows])
    project = 'data-agent-sft-smoke' if args.smoke else 'data-agent-rl-comparison'
    scope = f"Full dataset {proof['matched_sft_tasks']}" if proof.get('selection_scope') == 'full' else 'Matched medium+hard 907'
    if proof.get('include_test_overlap'):
        scope += ' · eval overlap'
    run_name = proof.get('run_name', 'LFM 2.6B · SFT bash · ' + scope) + (' · smoke' if args.smoke else '')
    config = SFTConfig(
        output_dir=str(out / 'run'), learning_rate=args.learning_rate,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=True, optim='paged_adamw_8bit', lr_scheduler_type='constant', warmup_steps=0,
        weight_decay=0.0, max_grad_norm=1.0, seed=42, data_seed=42,
        max_length=None, packing=False, padding_free=False, assistant_only_loss=not (pretokenized or prompt_completion),
        completion_only_loss=True if prompt_completion else None,
        chat_template_path=str(data / 'training_chat_template.jinja') if prompt_completion else None,
        loss_type='chunked_nll', dataset_num_proc=None,
        save_strategy='steps', save_steps=args.save_steps, save_total_limit=None,
        logging_steps=1, report_to='trackio', project=project, trackio_space_id=None, trackio_static_space_id=False,
        run_name=run_name, dataloader_num_workers=0,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision, local_files_only=True)
    assert tokenizer.chat_template == (data / 'chat_template.jinja').read_text()
    model_class = AutoModelForImageTextToText if model_id == 'Qwen/Qwen3.5-2B' else AutoModelForCausalLM
    model = model_class.from_pretrained(model_id, revision=model_revision,
            local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    parameter = model.get_input_embeddings().weight
    parameter_name = next(name for name, value in model.named_parameters() if value is parameter)
    initial_probe = parameter.detach().cpu().clone()

    class Audit(TrainerCallback):
        def on_train_end(self, config, state, control, **kwargs):
            if state.is_world_process_zero:
                materialize()

        def on_log(self, config, state, control, logs=None, **kwargs):
            event = {'step': state.global_step, 'resumed_from': str(args.resume) if args.resume else None, **(logs or {})}
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(event) + '\n')
            for key in ['loss', 'grad_norm']:
                if key in event and not math.isfinite(event[key]):
                    raise RuntimeError(f'Nonfinite {key}')

        def on_step_end(self, config, state, control, **kwargs):
            if args.eval_every is None and completed_epoch(state.epoch) is not None:
                control.should_save = True
            if args.stop_after and state.global_step >= args.stop_after:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, config, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            ckpt = out / 'run' / f'checkpoint-{state.global_step}'
            final = state.global_step >= state.max_steps or control.should_training_stop
            mark_saved(ckpt, state.global_step, model_id, model_revision, final=final)
            if should_evaluate(state.global_step, state.epoch, args.eval_every, final):
                requests = out / 'eval_requests'
                requests.mkdir(exist_ok=True)
                req = {'checkpoint': str(ckpt), 'step': state.global_step, 'model': model_id,
                       'base_revision': model_revision, 'epoch': completed_epoch(state.epoch), 'pass_k': 1, 'tasks': 250,
                       'harnesses': ['opencode', 'claude-code', 'codex', 'mini-swe-agent'],
                       'concurrency': args.eval_concurrency, 'status': 'requested; requires a separate eval allocation',
                       'training_test_overlap': {key: proof.get(key, 0) for key in ['test_task_overlap', 'test_notebook_overlap', 'test_question_overlap']},
                       'smoke': args.smoke}
                tmp = requests / f'checkpoint-{state.global_step}.tmp'
                tmp.write_text(json.dumps(req, indent=2) + '\n')
                tmp.replace(requests / f'checkpoint-{state.global_step}.json')

    trainer = SFTTrainer(model=model, args=config, processing_class=tokenizer,
                         train_dataset=dataset, callbacks=[Audit()])
    # Preparation is complete; checkpoints must retain the native inference template.
    tokenizer.chat_template = (data / 'chat_template.jinja').read_text()
    # Check the actual trainer preparation against the independent full-corpus token audit.
    for row, prepared in zip(rows, trainer.train_dataset):
        expected = reference[row['task_id']]
        assert prepared['input_ids'] == expected['input_ids'], row['task_id']
        assert prepared['labels'] == expected['labels'], row['task_id']
    for index in sorted({0, max(range(len(rows)), key=lambda i: len(reference[rows[i]['task_id']]['input_ids']))}):
        batch = trainer.data_collator([trainer.train_dataset[index]])
        assert torch.equal(batch['labels'][0], torch.tensor(reference[rows[index]['task_id']]['labels']))
    versions = {name: importlib.metadata.version(name) for name in ['trl','transformers','torch','datasets','accelerate','bitsandbytes','trackio']}
    provenance = {'model': model_id, 'revision': model_revision, 'configuration': config.to_dict(),
                  'data_preparation': proof, 'task_ids': [r['task_id'] for r in rows], 'versions': versions,
                  'trainer_module': sys.modules[SFTTrainer.__module__].__file__,
                  'native_trl_label_equality': True, 'loss_on_user_or_tool_tokens': False,
                  'loss_mask_source': 'Native TRL completion_only_loss' if prompt_completion else ('Explicit student-token labels, current assistant completion only' if pretokenized else 'Native assistant masks'),
                  'smoke': args.smoke, 'resume': str(args.resume) if args.resume else None}
    (out / ('config-resumed.json' if args.resume else 'config.json')).write_text(json.dumps(provenance, indent=2, default=str) + '\n')
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_state()
    step = trainer.state.global_step
    checkpoint = out / 'run' / f'checkpoint-{step}'
    if not checkpoint.exists():
        trainer._save_checkpoint(trainer.model, trial=None)
        mark_saved(checkpoint, step, model_id, model_revision, final=True)
    finalize_saved(checkpoint)
    changed = not torch.equal(initial_probe, dict(model.named_parameters())[parameter_name].detach().cpu())
    assert changed, 'No measured weight change'
    evidence = {'step': step, 'checkpoint': str(checkpoint), 'probe_parameter': parameter_name,
                'weights_changed': changed, 'metrics': result.metrics, 'task_count': len(rows),
                'native_trl_label_equality': True, 'max_sequence_length': max(len(reference[r['task_id']]['input_ids']) for r in rows),
                'checkpoint_resume': bool(args.resume)}
    (out / f'training_verified_step_{step}.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == '__main__':
    main()
