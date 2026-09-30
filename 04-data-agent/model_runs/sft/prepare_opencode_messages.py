"""Prepare all published harness demonstrations with native student masks."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path

from prepare import digest

MODELS = {'lfm': ('LiquidAI/LFM2.5-2.6B', '654f9463ce32b05d0429d76fe1f580b27d4c1ac0'),
          'qwen': ('Qwen/Qwen3.5-2B', '15852e8c16360a2fea060d615a32b45270f8a8fc')}


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


def student_tokens(tokenizer, prompt, completion, tools, template, kwargs):
    assert len(completion) == 1 and completion[0]['role'] == 'assistant'
    kwargs = {'tools': tools, 'chat_template': template, **kwargs}
    prefix = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, **kwargs)
    text = tokenizer.apply_chat_template(prompt + completion, tokenize=False, **kwargs)
    assert text.startswith(prefix)
    prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=True, **kwargs)['input_ids']
    ids = tokenizer.apply_chat_template(prompt + completion, return_dict=True, **kwargs)['input_ids']
    independently_encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert ids == independently_encoded['input_ids']
    assert ids[:len(prompt_ids)] == prompt_ids
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
    for label, (start, end) in zip(labels, independently_encoded['offset_mapping'], strict=True):
        assert not (start < len(prefix) < end), 'A token straddles the supervision boundary'
        assert (label != -100) == (start >= len(prefix))
    supervised = [i for i, label in enumerate(labels) if label != -100]
    assert supervised and supervised == list(range(supervised[0], len(ids)))
    assert tokenizer.convert_tokens_to_ids('<|im_end|>') in labels[supervised[0]:]
    return {'input_ids': ids, 'labels': labels}, len(ids) - supervised[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=list(MODELS), required=True)
    args = parser.parse_args()
    root, source, output = args.workspace.resolve(), args.source.resolve(), args.output.resolve()
    model, revision = MODELS[args.model]
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
    publication_root = source.parent
    publication = json.loads((publication_root / 'published.json').read_text())
    status = json.loads((publication_root / 'status.json').read_text())
    assert status['result'] == 'all_published_verified'
    harnesses = ['opencode', 'claude-code', 'codex', 'mini-swe-agent']
    files, excluded, receipts = [], {}, {}
    for harness in harnesses:
        receipt_path = source / harness / 'export-complete.json'
        receipt = json.loads(receipt_path.read_text())
        receipts[harness] = {'path': str(receipt_path), 'sha256': digest(receipt_path)}
        assert publication[harness]['job'] == receipt['job']
        assert publication[harness]['successes'] == receipt['successes']
        for item in receipt['files']:
            if item['path'].endswith('.parquet'):
                item = {**item, 'path': harness + '/' + item['path']}
                assert digest(source / item['path']) == item['sha256'], item['path']
                files.append(item)
        excluded[harness] = {r['task_id'] for r in json.loads((source / harness / 'manifests' / (harness + '-excluded.json')).read_text())}
    expected_rollouts = sum(publication[h]['successes'] for h in harnesses)
    expected_examples = sum(publication[h]['training_rows'] for h in harnesses)
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
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)
    (output / 'chat_template.jinja').write_text(tokenizer.chat_template)
    template = tokenizer.chat_template
    kwargs = {'preserve_thinking': True} if args.model == 'lfm' else {'enable_thinking': False}
    if args.model == 'lfm':
        old = '{{- "<|im_start|>assistant\\n<think>" -}}'
        assert template.count(old) == 1
        template = template.replace(old, '{{- "<|im_start|>assistant\\n" -}}')
    (output / 'training_chat_template.jinja').write_text(template)
    stats, rollouts, sample_ids = [], [], set()
    tools_seen, formats = Counter(), Counter()
    generation_prefix_mismatches = 0
    with (output / 'train.jsonl').open('w') as train, (output / 'tokens.jsonl').open('w') as tokens:
        for item in sorted(files, key=lambda x: x['path']):
            for batch in pq.ParquetFile(source / item['path']).iter_batches(batch_size=4):
                for row in batch.to_pylist():
                    task, harness = row['task_id'], row['harness']
                    assert harness in harnesses
                    assert task in rl and task not in forbidden and task not in excluded[harness]
                    assert task.rsplit('_qa_', 1)[0] not in forbidden_books
                    q = question(rl_manifest.parent / 'dataset/tasks' / task / 'instruction.md')
                    assert q not in test_questions
                    assert row['correctness'] == 1 and row['rollout_weight'] == 1
                    rollout = {k: row[k] for k in ['harness', 'task_id', 'difficulty', 'rollout_id', 'teacher_model', 'teacher_revision', 'tokenizer_revision', 'harness_version', 'source_sha256', 'turn_count']}
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
                        sample = harness + ':' + task + ':' + meta['node_id']
                        assert sample not in sample_ids
                        sample_ids.add(sample)
                        example_kwargs = dict(kwargs)
                        if args.model == 'qwen':
                            answer = completion[0]
                            text_content = answer.get('content') or ''
                            example_kwargs['enable_thinking'] = bool(answer.get('reasoning_content')) or (isinstance(text_content, str) and '</think>' in text_content)
                        encoded, supervised = student_tokens(tokenizer, prompt, completion, tools, template, example_kwargs)
                        tools_seen.update(t['function']['name'] for t in tools)
                        formats.update(['tool_call' if completion[0].get('tool_calls') else 'text'])
                        record = {'task_id': sample, 'rollout_task_id': task, 'rollout_id': row['rollout_id'],
                                  'node_id': meta['node_id'], 'difficulty': row['difficulty'], 'source_agent': harness,
                                  'prompt': prompt, 'completion': completion, 'tools': tools,
                                  'chat_template_kwargs': example_kwargs}
                        train.write(json.dumps(record, ensure_ascii=False) + '\n')
                        tokens.write(json.dumps({'task_id': sample, **encoded}) + '\n')
                        stats.append({'task_id': sample, 'rollout_task_id': task, 'difficulty': row['difficulty'],
                                      'source_agent': harness, 'tokens': len(encoded['input_ids']),
                                      'supervised_tokens': supervised, 'assistant_turns': 1})
                    if len(rollouts) % 50 == 0:
                        print(f'Audited {len(rollouts)}/{expected_rollouts} rollouts, {len(stats)} student training examples', flush=True)
    assert len(rollouts) == len({(r['harness'], r['task_id']) for r in rollouts}) == expected_rollouts
    assert len(stats) == expected_examples
    (output / 'rollouts.json').write_text(json.dumps(rollouts, indent=2) + '\n')
    (output / 'token_statistics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in stats))
    smoke = []
    for harness in harnesses:
        subset = [r for r in stats if r['source_agent'] == harness]
        for tier in ['medium', 'hard']:
            smoke.append(min((r for r in subset if r['difficulty'] == tier), key=lambda r:r['tokens'])['task_id'])
        smoke.append(max(subset, key=lambda r:r['tokens'])['task_id'])
    (output / 'smoke_task_ids.json').write_text(json.dumps(list(dict.fromkeys(smoke)), indent=2) + '\n')
    lengths = [r['tokens'] for r in stats]
    proof = {'dataset': 'AdithyaSK/qwen38-27b-harbor-rollouts', 'subsets': harnesses,
             'dataset_revision': status['revision'],
             'source_directory': str(source), 'source_receipts': receipts,
             'source_files': files, 'model': model, 'model_revision': revision,
             'teacher_model': rollouts[0]['teacher_model'], 'teacher_revision': rollouts[0]['teacher_revision'],
             'dataset_format': 'prompt_completion', 'selection_scope': 'all_harness_teacher',
             'run_name': ('LFM 2.6B' if args.model == 'lfm' else 'Qwen 3.5 2B') + ' · SFT four harnesses · 888 tasks',
             'matched_sft_tasks': len(stats), 'training_examples': len(stats), 'unique_tasks': len({r['task_id'] for r in rollouts}), 'rollouts': len(rollouts), 'harness_rollouts': dict(Counter(r['harness'] for r in rollouts)),
             'difficulty_counts': dict(Counter({r['task_id']:r['difficulty'] for r in rollouts}.values())),
             'rollout_difficulty_counts': dict(Counter(r['difficulty'] for r in rollouts)),
             'harness_training_examples': dict(Counter(r['source_agent'] for r in stats)),
             'completion_types': dict(formats), 'tools': sorted(tools_seen),
             'test_manifest': str(test_manifest), 'test_manifest_sha256': digest(test_manifest),
             'test_task_overlap': 0, 'test_notebook_overlap': 0, 'test_question_overlap': 0,
             'rl_pool_overlap': len({r['task_id'] for r in rollouts}), 'excluded_credential_rollouts': sum(map(len,excluded.values())),
             'only_current_assistant_completion_supervised': True,
             'teacher_ids_logprobs_used_as_student_targets': False,
             'native_generation_prefix_mismatches': generation_prefix_mismatches,
             'mask_note': 'Native TRL completion_only_loss; recorded prompts are context only. Token arrays here are an audit reference, not the training input.',
             'training_template_note': 'LFM training-only generation prefix omits an absent <think>; inference tokenizer/template remain original.' if args.model == 'lfm' else 'Unmodified Qwen template; training prefix follows recorded reasoning presence per example. Inference remains enable_thinking=false as in the baseline.',
             'packing': False, 'truncated_trajectories': 0,
             'total_tokens': sum(lengths), 'supervised_tokens': sum(r['supervised_tokens'] for r in stats),
             'lengths': {str(q): float(np.percentile(lengths, q)) for q in [0, 50, 90, 95, 99, 100]},
             'train_sha256': digest(output / 'train.jsonl'), 'tokens_sha256': digest(output / 'tokens.jsonl'),
             'template_sha256': digest(output / 'chat_template.jinja')}
    (output / 'preparation.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({k: v for k, v in proof.items() if k != 'source_files'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
