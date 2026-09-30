"""Summarize saved optimizer-admitted rollouts without loading a model or launching jobs."""

import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path

import orjson
import pandas as pd


def extract(row, audit_dir):
    path = audit_dir / "rollouts" / (row["rollout_id"] + ".json")
    raw = path.read_bytes()
    saved = orjson.loads(raw)
    result = saved["result"]
    assert saved["task_index"] == row["task_index"], path
    assert saved["harness"] == row["harness"], path
    turns = [t for t in result["turns"] if t.get("trainable", True) and not t.get("discarded", False)]
    tokens, supervised, action_tokens, prompt_tokens, calls = [], 0, 0, 0, []
    for turn in turns:
        count = len(turn["completion_token_ids"])
        prompt_len = len(turn["prompt_token_ids"])
        tokens.append(count)
        prompt_tokens += prompt_len
        mask = turn.get("loss_mask")
        if mask is None:
            supervised += count
        elif len(mask) == prompt_len + count:
            supervised += sum(mask[prompt_len:])
        else:
            assert len(mask) == count, (path, len(mask), prompt_len, count)
            supervised += sum(mask)
        if turn.get("tool_calls"):
            action_tokens += count
        for call in turn.get("tool_calls") or []:
            fn = call.get("function") or call
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = orjson.loads(args)
                except orjson.JSONDecodeError:
                    pass
            signature = (fn.get("name", ""), json.dumps(args, sort_keys=True))
            calls.append(signature)
    counts = collections.Counter(calls)
    reward = result.get("reward")
    if row["run"] == "Native OpenCode":
        correctness = result.get("correctness")
        reward = float(correctness >= 1.0) if correctness is not None else None
    return {
        **row, "binary_reward": reward, "agent_turns": len(turns),
        "emitted_tool_calls": len(calls), "exact_repeated_calls": sum(n - 1 for n in counts.values()),
        "completion_tokens": sum(tokens), "masked_completion_tokens": supervised,
        "receipt_token_delta": supervised - row["supervised_tokens"],
        "completion_tokens_in_tool_turns": action_tokens,
        "completion_tokens_in_text_only_turns": sum(tokens) - action_tokens,
        "max_response_tokens": max(tokens, default=0),
        "responses_over_4096": sum(n > 4096 for n in tokens),
        "responses_finish_length": sum(t.get("finish_reason") == "length" for t in turns),
        "sum_captured_prompt_tokens": prompt_tokens,
        "last_turn_has_tool": bool(turns and turns[-1].get("tool_calls")),
        "generation_s": saved["finished_at"] - saved["started_at"],
        "tool_names": dict(collections.Counter(name for name, _ in calls)),
        "source": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="Directory containing optimizer_rollouts.csv and training_lineage.csv")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    receipts = pd.read_csv(args.evidence / "optimizer_rollouts.csv")
    lineage = pd.read_csv(args.evidence / "training_lineage.csv")
    audit_dirs = {int(r.job): Path(r.coverage_source).parent for r in lineage.itertuples()}
    cache_path = args.evidence / "admitted_training_behavior.jsonl"
    cached = {}
    if cache_path.exists():
        with cache_path.open() as stream:
            for line in stream:
                item = json.loads(line)
                cached[(item["run"], item["rollout_id"])] = item
    rows = receipts.to_dict("records")
    missing = [r for r in rows if (r["run"], r["rollout_id"]) not in cached]
    print(f"Cached {len(cached)}; extracting {len(missing)} optimizer-admitted rollouts", flush=True)
    with cache_path.open("a") as out, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = (pool.submit(extract, row, audit_dirs[row["job"]]) for row in missing)
        # Keep only a small number of raw captures in flight.
        pending = collections.deque()
        for future in futures:
            pending.append(future)
            if len(pending) < args.workers * 2:
                continue
            item = pending.popleft().result()
            out.write(json.dumps(item) + "\n")
            cached[(item["run"], item["rollout_id"])] = item
            if len(cached) % 500 == 0:
                out.flush()
                print(f"Extracted {len(cached)}/{len(rows)}", flush=True)
        for future in pending:
            item = future.result()
            out.write(json.dumps(item) + "\n")
            cached[(item["run"], item["rollout_id"])] = item
    assert len(cached) == len(rows), (len(cached), len(rows))
    frame = pd.DataFrame(cached.values())
    frame.to_csv(args.evidence / "admitted_training_behavior.csv", index=False)
    print(frame.groupby("run").agg(rollouts=("rollout_id", "size"), tool_calls=("emitted_tool_calls", "mean"), tokens=("completion_tokens", "mean"), receipt_mismatches=("receipt_token_delta", lambda x: (x != 0).sum())).to_string(), flush=True)


if __name__ == "__main__":
    main()
