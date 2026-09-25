"""Replay saved training captures through the same lossless TITO checks as the smoke test."""

import argparse
import collections
import json
import os
from pathlib import Path

from openenv.harbor.models import HarborRolloutResult
from smoke_multiharness_tito import audit


def save_json(path, value):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def summarize_checks(checks, token_budget):
    by_harness = collections.defaultdict(list)
    for check in checks:
        by_harness[check['harness']].append(check)
    return {h: {
        'rollouts': len(rows), 'completed_results': sum(r['completed_result'] for r in rows),
        'incomplete_results': sum(not r['completed_result'] for r in rows),
        'tito_pass': sum(r['tito_pass'] for r in rows),
        'graded': sum(r.get('reward') is not None for r in rows),
        'reward_sum': sum(r.get('reward') or 0 for r in rows),
        'task_indices': sorted({r['task_index'] for r in rows}), 'token_budget': token_budget,
        'largest_row_tokens': max(r.get('largest_row_tokens', 0) for r in rows),
        **{key: sum(r.get(key, 0) for r in rows)
           for key in ('rows', 'eligible_tokens', 'retained_tokens', 'rows_over_40960', 'rows_over_token_budget')},
    } for h, rows in sorted(by_harness.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Job's audit directory")
    args = parser.parse_args()
    config_path = args.directory.parent / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    training = config.get("training", config)
    token_budget = (training['max_model_len'] if training.get('atomic_rollouts')
                    else training.get("token_budget", 40960))
    output = args.directory / "tito_checks"
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    for path in sorted((args.directory / "rollouts").glob("*.json")):
        cached = output / path.name
        if cached.exists():
            previous = json.loads(cached.read_text())
            if previous.get("token_budget") == token_budget and previous.get('audit_schema_version') == 2:
                checks.append(previous)
                continue
        record = json.loads(path.read_text())
        check = {k: record[k] for k in ("group_id", "episode_id", "harness", "task_index")}
        check.update(audit_schema_version=2, token_budget=token_budget,
                     completed_result=record.get('result') is not None)
        try:
            result = HarborRolloutResult.model_validate(record["result"])
            validation, _ = audit(result, token_budget=token_budget)
            check.update(validation, reward=result.reward, ok=result.ok, error=result.error)
        except Exception as exc:  # noqa: BLE001 - record a failed capture and audit the remaining ones
            check.update(tito_pass=False, error=f"{type(exc).__name__}: {exc}")
        save_json(cached, check)
        checks.append(check)
    summary = summarize_checks(checks, token_budget)
    save_json(args.directory / "tito_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
