"""Compare native TRL preparation with the independent full-corpus mask audit."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset, Features, Json
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--receipt', type=Path, required=True)
args = parser.parse_args()
proof = json.loads((args.data / 'preparation.json').read_text())
with (args.data / 'train.jsonl').open(encoding='utf-8') as source:
    rows = [json.loads(line) for line in source]
columns = ['prompt', 'completion', 'tools', 'chat_template_kwargs']
dataset = Dataset.from_list([{k: row[k] for k in columns} for row in rows],
                            features=Features({k: Json() for k in columns}))
tokenizer = AutoTokenizer.from_pretrained(proof['model'], revision=proof['model_revision'], local_files_only=True)
template = (args.data / 'training_chat_template.jinja').read_text()
config = SFTConfig(output_dir=str(args.receipt.parent / 'cpu-audit'), use_cpu=True,
                   report_to='none', bf16=False, max_length=None, packing=False,
                   completion_only_loss=True, assistant_only_loss=False, dataset_num_proc=None)
preparer = SimpleNamespace(chat_template=template, completion_only_loss=True, _tokenizer=tokenizer)
prepared = SFTTrainer._prepare_dataset(preparer, dataset, tokenizer, config, False, None, 'audit')
assert len(prepared) == len(rows) == proof['training_examples']
with (args.data / 'tokens.jsonl').open() as source:
    for row, actual, line in zip(rows, prepared, source, strict=True):
        expected = json.loads(line)
        assert row['task_id'] == expected['task_id']
        assert actual['input_ids'] == expected['input_ids'], row['task_id']
        assert actual['labels'] == expected['labels'], row['task_id']
receipt = {'passed': True, 'model': proof['model'], 'examples': len(rows),
           'native_trl_input_ids_and_labels_equal': True, 'truncated_examples': 0}
args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
