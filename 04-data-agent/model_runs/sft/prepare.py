"""Select SFT demonstrations for the frozen LFM RL tasks and audit their masks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

DATASET = 'FineEnvs/SmolDataEnvs-sft'
REVISION = 'a9fa95ab5dff4f4522081f7eb003109e56b7a36d'
MODEL = 'LiquidAI/LFM2.5-2.6B'
MODEL_REVISION = '654f9463ce32b05d0429d76fe1f580b27d4c1ac0'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_list(value):
    value = json.loads(value) if isinstance(value, str) else value
    return [json.loads(item) if isinstance(item, str) else item for item in value]


def normalize(row, require_final_assistant=True):
    messages, tools = decode_list(row['messages']), decode_list(row['tools'])
    assert {t['function']['name'] for t in tools} == {'bash'}, row['task_id']
    calls = set()
    for message in messages:
        for call in message.get('tool_calls') or []:
            function = call['function']
            assert function['name'] == 'bash'
            if isinstance(function['arguments'], str):
                function['arguments'] = json.loads(function['arguments'])
            assert isinstance(function['arguments']['command'], str)
            assert call['id'] not in calls
            calls.add(call['id'])
        if message['role'] == 'tool':
            assert message['tool_call_id'] in calls
    assert calls
    if require_final_assistant:
        assert messages[-1]['role'] == 'assistant'
    return {'task_id': row['task_id'], 'difficulty': row['difficulty_tier'],
            'source_agent': row['source_agent'], 'messages': messages, 'tools': tools,
            'chat_template_kwargs': {'preserve_thinking': True}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=['matched', 'full'], default='matched')
    p.add_argument('--include-test-overlap', action='store_true', help='Keep overlap only for explicitly disclosed, non-held-out experiments')
    args = p.parse_args()
    root, out = args.workspace.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import numpy as np

    rl = root / 'experiments/lfm25-medium-hard1000-20260921'
    test = root / 'experiments/async_grpo_harbor_data_agent/logs/multi4-baseline-20260914'
    selected = json.loads((rl / 'manifest.json').read_text())['tasks']
    heldout = json.loads((test / 'manifest.json').read_text())['tasks']
    want, forbidden = {r['name']: r for r in selected}, {r['name'] for r in heldout}
    forbidden_books = {r['notebook'] for r in heldout}
    source = Path(hf_hub_download(DATASET, 'data/train-00000-of-00001.parquet', repo_type='dataset', revision=REVISION))
    rows = pq.read_table(source).to_pylist()
    assert len({r['task_id'] for r in rows}) == len(rows)
    by_id = {r['task_id']: r for r in rows}
    def extract_question(text):
        return ' '.join(text.split('Question:\n', 1)[1].split('\n\nWork it out', 1)[0].split())
    heldout_questions = {extract_question((test / 'dataset/tasks' / r['name'] / 'instruction.md').read_text()) for r in heldout}
    forbidden_book_ids = {r['name'].rsplit('_qa_', 1)[0] for r in heldout}
    excluded = []
    if args.scope == 'full':
        matched = []
        for raw in rows:
            row = normalize(raw, require_final_assistant=False)
            questions = [extract_question(m['content']) for m in row['messages'] if m['role'] == 'user' and 'Question:\n' in m['content']]
            assert questions, row['task_id']
            reasons = []
            if row['task_id'] in forbidden:
                reasons.append('test_task')
            if row['task_id'].rsplit('_qa_', 1)[0] in forbidden_book_ids:
                reasons.append('test_notebook')
            if set(questions) & heldout_questions:
                reasons.append('test_question')
            if reasons:
                excluded.append({'task_id': row['task_id'], 'reasons': reasons})
            if not reasons or args.include_test_overlap:
                matched.append(row)
        (out / 'test_overlap_audit.json').write_text(json.dumps(excluded, indent=2) + '\n')
    else:
        matched = [normalize(by_id[r['name']]) for r in selected if r['name'] in by_id]
        assert len(matched) == 907 and len(want) == 1000
    if not args.include_test_overlap:
        assert not ({r['task_id'] for r in matched} & forbidden)
        assert not ({r['task_id'].rsplit('_qa_', 1)[0] for r in matched} & forbidden_book_ids)
    missing = [r for r in selected if r['name'] not in by_id]
    (out / 'missing_tasks.json').write_text(json.dumps(missing, indent=2) + '\n')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    assert 'generation' in tokenizer.chat_template
    (out / 'chat_template.jinja').write_text(tokenizer.chat_template)
    token_stats = []
    heldout_questions = set()
    def question(folder):
        text = (folder / 'instruction.md').read_text()
        return ' '.join(text.split('Question:\n', 1)[1].split('\n\nWork it out', 1)[0].split())
    for task in heldout:
        heldout_questions.add(question(test / 'dataset/tasks' / task['name']))
    token_rows = []
    for i, row in enumerate(matched):
        name = row['task_id']
        if args.scope == 'matched':
            assert row['difficulty'] == want[name]['difficulty']
            q = question(rl / 'dataset/tasks' / name)
        else:
            q = extract_question(next(m['content'] for m in row['messages'] if m['role'] == 'user' and 'Question:\n' in m['content']))
        user = ' '.join(' '.join(m['content'] for m in row['messages'] if m['role'] == 'user').split())
        assert q in user, f'Question mismatch: {name}'
        if not args.include_test_overlap:
            assert q not in heldout_questions
        text = tokenizer.apply_chat_template(row['messages'], tools=row['tools'], tokenize=False, preserve_thinking=True)
        encoded = tokenizer.apply_chat_template(row['messages'], tools=row['tools'], return_dict=True,
                    return_assistant_tokens_mask=True, preserve_thinking=True)
        ids, mask = encoded['input_ids'], encoded['assistant_masks']
        offset = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        assert ids == offset['input_ids'] and len(ids) == len(mask) and any(mask)
        headers = list(re.finditer(r'<\|im_start\|>(system|user|assistant|tool)\n', text))
        assert len(headers) == len(row['messages'])
        role_at = 0
        for (start, end), supervised in zip(offset['offset_mapping'], mask):
            while role_at + 1 < len(headers) and start >= headers[role_at + 1].start():
                role_at += 1
            if supervised:
                assert headers[role_at].group(1) == 'assistant', name
                assert start >= headers[role_at].end(), name
        # Every assistant turn terminator must contribute to loss, including tool-call turns.
        eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
        assert sum(t == eos and bool(m) for t, m in zip(ids, mask)) == sum(m['role'] == 'assistant' for m in row['messages'])
        stat = {'task_id': name, 'difficulty': row['difficulty'], 'source_agent': row['source_agent'],
                'tokens': len(ids), 'supervised_tokens': sum(mask), 'assistant_turns': sum(m['role'] == 'assistant' for m in row['messages'])}
        token_stats.append(stat)
        token_rows.append({'task_id': name, 'input_ids': ids, 'labels': [t if m else -100 for t, m in zip(ids, mask)]})
        if (i + 1) % 100 == 0:
            print(f'Audited {i+1}/{len(matched)} conversations', flush=True)
    for filename, records in [('train.jsonl', matched), ('tokens.jsonl', token_rows), ('token_statistics.jsonl', token_stats)]:
        with (out / filename).open('w') as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    # Short complete conversations keep the smoke cheap; both difficulty tiers remain represented.
    smoke_ids = []
    for tier in ['medium', 'hard']:
        smoke_ids += [r['task_id'] for r in sorted((r for r in token_stats if r['difficulty'] == tier), key=lambda r: (r['tokens'], r['task_id']))[:4]]
    (out / 'smoke_task_ids.json').write_text(json.dumps(smoke_ids, indent=2) + '\n')
    lengths = [r['tokens'] for r in token_stats]
    proof = {'dataset': DATASET, 'dataset_revision': REVISION, 'dataset_parquet': str(source),
             'dataset_sha256': digest(source), 'model': MODEL, 'model_revision': MODEL_REVISION,
             'rl_manifest': str(rl / 'manifest.json'), 'rl_manifest_sha256': digest(rl / 'manifest.json'),
             'test_manifest': str(test / 'manifest.json'), 'test_manifest_sha256': digest(test / 'manifest.json'),
             'selected_rl_tasks': 1000, 'matched_sft_tasks': len(matched), 'missing_tasks': len(missing),
             'difficulty_counts': dict(Counter(r['difficulty'] for r in matched)),
             'source_agents': dict(Counter(r['source_agent'] for r in matched)),
             'ending_roles': dict(Counter(r['messages'][-1]['role'] for r in matched)),
             'test_task_overlap': sum('test_task' in x['reasons'] for x in excluded) if args.include_test_overlap else 0,
             'test_notebook_overlap': sum('test_notebook' in x['reasons'] for x in excluded) if args.include_test_overlap else 0,
             'test_question_overlap': sum('test_question' in x['reasons'] for x in excluded) if args.include_test_overlap else 0,
             'include_test_overlap': args.include_test_overlap,
             'original_question_matches': len(matched) if args.scope == 'matched' else None,
             'selection_scope': args.scope, 'source_tasks': len(rows), 'excluded_test_overlaps': 0 if args.include_test_overlap else len(excluded),
             'only_assistant_tokens_supervised': True,
             'assistant_end_tokens_supervised': True, 'tools': ['bash'], 'packing': False,
             'truncated_trajectories': 0, 'total_tokens': sum(lengths),
             'supervised_tokens': sum(r['supervised_tokens'] for r in token_stats),
             'lengths': {str(q): float(np.percentile(lengths, q)) for q in [0, 50, 90, 95, 99, 100]},
             'above_8192': sum(n > 8192 for n in lengths), 'above_32768': sum(n > 32768 for n in lengths),
             'above_65536': sum(n > 65536 for n in lengths), 'train_sha256': digest(out / 'train.jsonl'),
             'tokens_sha256': digest(out / 'tokens.jsonl'), 'template_sha256': digest(out / 'chat_template.jinja'),
             'approval': ('User explicitly approved all 4,677 demonstrations with overlap labeled, 2026-09-27.' if args.include_test_overlap else 'User requested full-dataset SFT, 2026-09-27; test overlaps excluded.') if args.scope == 'full' else 'User approved using the 907 matching tasks, 2026-09-27.'}
    (out / 'preparation.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__':
    main()
