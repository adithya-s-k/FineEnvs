"""Check one full-length optimizer update using the longest selected trajectory."""
import argparse
import json
import math
from pathlib import Path
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import SFTTrainer, SFTConfig
    stat = max(map(json.loads, (args.data / 'token_statistics.jsonl').read_text().splitlines()), key=lambda r: r['tokens'])
    row = next(r for r in map(json.loads, (args.data / 'train.jsonl').read_text().splitlines()) if r['task_id'] == stat['task_id'])
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    trainer = SFTTrainer(model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_list([{k: row[k] for k in ['messages','tools','chat_template_kwargs']}]),
        args=SFTConfig(output_dir=str(args.output), max_steps=1, per_device_train_batch_size=1,
            gradient_accumulation_steps=1, learning_rate=3e-6, bf16=True, optim='paged_adamw_8bit',
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
            max_length=None, packing=False, padding_free=False, assistant_only_loss=True,
            loss_type='chunked_nll', save_strategy='no', report_to='none', logging_steps=1, seed=42))
    assert len(trainer.train_dataset[0]['input_ids']) == stat['tokens']
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    update = next(r for r in trainer.state.log_history if 'grad_norm' in r)
    assert math.isfinite(update['loss']) and math.isfinite(update['grad_norm']) and update['grad_norm'] > 0
    record = {'checkpoint': str(args.checkpoint), 'task': stat, 'max_memory_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
              'max_memory_reserved_gib': torch.cuda.max_memory_reserved()/1024**3,
              'optimizer_update': update, 'metrics': result.metrics, 'saved_new_weights': False,
              'note': 'Isolated memory qualification. Original smoke checkpoint unchanged.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'longest_verified.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
