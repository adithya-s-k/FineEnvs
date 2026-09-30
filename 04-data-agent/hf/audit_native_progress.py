"""Incrementally apply the qualified native TiTO audit on a separate CPU process."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(REPRO_ROOT=str(args.root), CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
    sys.path.insert(0, str(args.root / 'hf/runtime'))
    from common import configure, write_json
    configure()
    from training_capture_audit import positions
    from trl.experimental.async_grpo.openenv_harness import _turns_from_trace
    from trl.experimental.async_grpo.async_rollout_worker import _chain_to_sequences
    from openenv.core.harness.capture.validate import validate_training_turn
    from data_agent_env import opencode_agent_turns, to_trace_entries
    from data_agent_env.models import DataAgentRolloutResult
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / 'rollouts.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for path in sorted((args.training / 'audit/rollouts').glob('*.json')):
        stat = path.stat()
        stamp = [stat.st_size, stat.st_mtime_ns]
        if path.name in cache and cache[path.name]['stamp'] == stamp:
            continue
        encoded = path.read_bytes()
        sha = hashlib.sha256(encoded).hexdigest()
        if path.name in cache:
            if cache[path.name]['sha256'] != sha:
                raise ValueError('Previously audited rollout changed')
            cache[path.name]['stamp'] = stamp
            continue
        record = json.loads(encoded)
        if not record.get('result'):
            continue
        entries = opencode_agent_turns(to_trace_entries(DataAgentRolloutResult.model_validate(record['result'])))
        if not entries:
            continue
        expected = collections.Counter()
        for entry in entries:
            p, c, lp, mask = (entry[k] for k in ('prompt_token_ids', 'completion_token_ids', 'per_token_logps', 'loss_mask'))
            validate_training_turn(p, c, lp, mask)
            expected.update(positions(p + c, mask, [0.] * len(p) + lp))
        rows, _ = _chain_to_sequences(_turns_from_trace(entries), record['episode_id'], fork_threshold=0)
        retained = collections.Counter()
        for row in rows:
            retained.update(positions(row.input_ids, row.completion_mask, row.old_log_probs))
        if not expected or retained != expected or not any(lp < 0 for e in entries for lp in e['per_token_logps']):
            raise ValueError('Supervised token/context/logprob positions were not retained exactly')
        cache[path.name] = {'stamp': stamp, 'sha256': sha, 'episode_id': record['episode_id'],
            'rows': len(rows), 'eligible_tokens': sum(expected.values()), 'retained_tokens': sum(retained.values()),
            'tito_pass': True}
    by_id = {r['episode_id']: r for r in cache.values()}
    consumed = pending = 0
    receipt = args.training / 'audit/optimizer_rollouts.jsonl'
    if receipt.exists():
        for line in receipt.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):
                continue
            for row in json.loads(line)['rollouts']:
                if row['rollout_id'] not in by_id:
                    pending += 1  # capture can be published after this observation's directory scan
                    continue
                report = by_id[row['rollout_id']]
                if row['rows'] != report['rows'] or row['supervised_tokens'] != report['retained_tokens']:
                    raise ValueError('Optimizer receipt differs from exact captured supervision')
                consumed += 1
    write_json(cache_path, cache)
    write_json(args.out / 'summary.json', {'checked_at': time.time(), 'opencode': {
        'completed_results': len(cache), 'tito_pass': len(cache),
        'eligible_tokens': sum(r['eligible_tokens'] for r in cache.values()),
        'retained_tokens': sum(r['retained_tokens'] for r in cache.values()),
        'optimizer_rollouts_verified': consumed, 'pending_capture_publication': pending}})


if __name__ == '__main__':
    main()
