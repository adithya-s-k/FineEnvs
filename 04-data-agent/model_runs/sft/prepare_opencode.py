"""Convert published OpenCode teacher turns into explicit LFM SFT labels."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path

from prepare import MODEL, MODEL_REVISION, digest


def normalize_messages(messages):
    result = copy.deepcopy(messages)
    for message in result:
        assert message['role'] in {'system', 'user', 'assistant', 'tool'}
        assert message.get('content') is None or isinstance(message['content'], (str, list))
        if isinstance(message.get('content'), list):
            assert all(isinstance(x, dict) and x.get('type') == 'text' for x in message['content'])
        for call in message.get('tool_calls') or []:
            function = call['function']
            if isinstance(function.get('arguments'), str):
                function['arguments'] = json.loads(function['arguments'])
            assert isinstance(function['arguments'], dict)
        for key in ['reasoning', 'reasoning_content', 'thinking']:
            assert not message.get(key) or isinstance(message[key], str)
    return result


def student_tokens(tokenizer, prompt, completion, tools):
    assert len(completion) == 1 and completion[0]['role'] == 'assistant'
    kwargs = {'tools': tools, 'preserve_thinking': True}
    context = tokenizer.apply_chat_template(prompt, tokenize=False, **kwargs)
    prefix = context + '<|im_start|>assistant\n'
    text = tokenizer.apply_chat_template(prompt + completion, tokenize=False, **kwargs)
    assert text.startswith(prefix)
    encoded = tokenizer.apply_chat_template(prompt + completion, return_dict=True,
                                           return_assistant_tokens_mask=True, **kwargs)
    independently_encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, masks = encoded['input_ids'], encoded['assistant_masks']
    assert ids == independently_encoded['input_ids']
    labels = []
    for token, assistant, (start, end) in zip(ids, masks, independently_encoded['offset_mapping'], strict=True):
        assert not (start < len(prefix) < end), 'A token straddles the supervision boundary'
        current = start >= len(prefix)
        assert not current or assistant, 'Current completion must have a native assistant mask'
        labels.append(token if current else -100)
    supervised = [i for i, label in enumerate(labels) if label != -100]
    assert supervised and supervised == list(range(supervised[0], len(ids)))
    assert tokenizer.convert_tokens_to_ids('<|im_end|>') in labels[supervised[0]:]
    return {'input_ids': ids, 'labels': labels}, len(ids) - supervised[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, source, output = args.workspace.resolve(), args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import pyarrow.parquet as pq
    import numpy as np
    from dotenv import dotenv_values
    from transformers import AutoTokenizer
    import sys
    sys.path.insert(0, str(root / 'experiments/multiharness_teacher_collection/tools'))
    from publish_dataset import secrets_guard

    credentials = [v for k, v in dotenv_values(root / 'experiments/.env').items()
                   if v and len(v) >= 16 and any(x in k.upper() for x in ['TOKEN', 'SECRET', 'API_KEY', 'PASSWORD'])]
    receipt = json.loads((source / 'export-complete.json').read_text())
    publication = json.loads((source.parent.parent / 'published.json').read_text())['opencode']
    assert publication['job'] == receipt['job']
    assert receipt['successes'] == 801 and receipt['training_rows'] == 4825
    files = [r for r in receipt['files'] if r['path'].endswith('.parquet')]
    for item in files:
        assert digest(source / item['path']) == item['sha256'], item['path']
    excluded = {r['task_id'] for r in json.loads((source / 'manifests/opencode-excluded.json').read_text())}
    prior = json.loads((root / 'experiments/lfm25-sft-20260927/data/preparation.json').read_text())
    test_manifest = Path(prior['test_manifest'])
    test = json.loads(test_manifest.read_text())['tasks']
    rl_manifest = Path(prior['rl_manifest'])
    rl = {r['name']: r for r in json.loads(rl_manifest.read_text())['tasks']}
    forbidden = {r['name'] for r in test}
    forbidden_books = {r['name'].rsplit('_qa_', 1)[0] for r in test}
    def question(path):
        return ' '.join(path.read_text().split('Question:\n', 1)[1].split('\n\nWork it out', 1)[0].split())
    test_questions = {question(test_manifest.parent / 'dataset/tasks' / r['name'] / 'instruction.md') for r in test}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    (output / 'chat_template.jinja').write_text(tokenizer.chat_template)
    stats, rollouts, sample_ids = [], [], set()
    tools_seen, formats = Counter(), Counter()
    generation_prefix_mismatches = 0
    with (output / 'train.jsonl').open('w') as train, (output / 'tokens.jsonl').open('w') as tokens:
        for item in sorted(files, key=lambda x: x['path']):
            for batch in pq.ParquetFile(source / item['path']).iter_batches(batch_size=4):
                for row in batch.to_pylist():
                    task = row['task_id']
                    assert task in rl and task not in forbidden and task not in excluded
                    assert task.rsplit('_qa_', 1)[0] not in forbidden_books
                    q = question(rl_manifest.parent / 'dataset/tasks' / task / 'instruction.md')
                    assert q not in test_questions
                    assert row['correctness'] == 1 and row['rollout_weight'] == 1 and row['harness'] == 'opencode'
                    rollout = {k: row[k] for k in ['task_id', 'difficulty', 'rollout_id', 'teacher_model', 'teacher_revision', 'tokenizer_revision', 'harness_version', 'source_sha256', 'turn_count']}
                    rollouts.append(rollout)
                    assert row['turn_count'] == len(row['training_turns'])
                    for turn in row['training_turns']:
                        pids, cids, mask = turn['prompt_token_ids'], turn['completion_token_ids'], turn['loss_mask']
                        assert len(mask) == len(pids) + len(cids)
                        assert mask == [0] * len(pids) + [1] * len(cids)
                        assert cids and len(cids) == len(turn['per_token_logps'])
                        assert all(math.isfinite(x) for x in turn['per_token_logps'])
                        blob = '\n'.join(turn[k] for k in ['prompt_json', 'completion_json', 'tools_json', 'metadata_json']).encode()
                        secrets_guard(blob, credentials)
                        prompt = normalize_messages(json.loads(turn['prompt_json']))
                        completion = normalize_messages(json.loads(turn['completion_json']))
                        tools = json.loads(turn['tools_json'])
                        meta = json.loads(turn['metadata_json'])
                        sample = task + ':' + meta['node_id']
                        assert sample not in sample_ids
                        sample_ids.add(sample)
                        encoded, supervised = student_tokens(tokenizer, prompt, completion, tools)
                        generation_prompt = tokenizer.apply_chat_template(prompt, tools=tools, preserve_thinking=True,
                                                                         tokenize=False, add_generation_prompt=True)
                        full = tokenizer.apply_chat_template(prompt + completion, tools=tools, preserve_thinking=True, tokenize=False)
                        generation_prefix_mismatches += not full.startswith(generation_prompt)
                        tools_seen.update(t['function']['name'] for t in tools)
                        formats.update(['tool_call' if completion[0].get('tool_calls') else 'text'])
                        record = {'task_id': sample, 'rollout_task_id': task, 'rollout_id': row['rollout_id'],
                                  'node_id': meta['node_id'], 'difficulty': row['difficulty'], 'source_agent': 'opencode',
                                  'prompt': prompt, 'completion': completion, 'tools': tools,
                                  'chat_template_kwargs': {'preserve_thinking': True}}
                        train.write(json.dumps(record, ensure_ascii=False) + '\n')
                        tokens.write(json.dumps({'task_id': sample, **encoded}) + '\n')
                        stats.append({'task_id': sample, 'rollout_task_id': task, 'difficulty': row['difficulty'],
                                      'source_agent': 'opencode', 'tokens': len(encoded['input_ids']),
                                      'supervised_tokens': supervised, 'assistant_turns': 1})
                    if len(rollouts) % 50 == 0:
                        print(f'Audited {len(rollouts)}/801 rollouts, {len(stats)} student training examples', flush=True)
    assert len(rollouts) == len({r['task_id'] for r in rollouts}) == 801
    assert len(stats) == 4825
    (output / 'rollouts.json').write_text(json.dumps(rollouts, indent=2) + '\n')
    (output / 'token_statistics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in stats))
    smoke = []
    for tier in ['medium', 'hard']:
        smoke += [r['task_id'] for r in sorted((r for r in stats if r['difficulty'] == tier), key=lambda r: r['tokens'])[:4]]
    smoke.append(max(stats, key=lambda r: r['tokens'])['task_id'])
    (output / 'smoke_task_ids.json').write_text(json.dumps(list(dict.fromkeys(smoke)), indent=2) + '\n')
    lengths = [r['tokens'] for r in stats]
    proof = {'dataset': 'AdithyaSK/qwen38-27b-harbor-rollouts', 'subset': 'opencode',
             'dataset_revision': publication['verified_revision'],
             'source_directory': str(source), 'source_receipt_sha256': digest(source / 'export-complete.json'),
             'source_files': files, 'model': MODEL, 'model_revision': MODEL_REVISION,
             'teacher_model': rollouts[0]['teacher_model'], 'teacher_revision': rollouts[0]['teacher_revision'],
             'dataset_format': 'pretokenized_completion', 'selection_scope': 'opencode_teacher',
             'run_name': 'LFM 2.6B · SFT OpenCode teacher · 801 tasks',
             'matched_sft_tasks': len(stats), 'training_examples': len(stats), 'unique_tasks': len(rollouts),
             'difficulty_counts': dict(Counter(r['difficulty'] for r in rollouts)),
             'completion_types': dict(formats), 'tools': sorted(tools_seen),
             'test_manifest': str(test_manifest), 'test_manifest_sha256': digest(test_manifest),
             'test_task_overlap': 0, 'test_notebook_overlap': 0, 'test_question_overlap': 0,
             'rl_pool_overlap': 801, 'excluded_credential_rollouts': len(excluded),
             'only_current_assistant_completion_supervised': True,
             'teacher_ids_logprobs_used_as_student_targets': False,
             'native_generation_prefix_mismatches': generation_prefix_mismatches,
             'mask_note': 'Explicit LFM-token labels preserve the recorded current-turn-only supervision; the native generation prompt adds an absent <think> prefix.',
             'packing': False, 'truncated_trajectories': 0,
             'total_tokens': sum(lengths), 'supervised_tokens': sum(r['supervised_tokens'] for r in stats),
             'lengths': {str(q): float(np.percentile(lengths, q)) for q in [0, 50, 90, 95, 99, 100]},
             'train_sha256': digest(output / 'train.jsonl'), 'tokens_sha256': digest(output / 'tokens.jsonl'),
             'template_sha256': digest(output / 'chat_template.jinja')}
    (output / 'preparation.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({k: v for k, v in proof.items() if k != 'source_files'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
