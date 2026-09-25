"""Reconcile exact supervised positions with native TRL rows and optimizer receipts."""
import collections
import hashlib
import json
from pathlib import Path
import struct

from common import write_json


def positions(ids, masks, logprobs):
    result = collections.Counter()
    context = hashlib.sha256()
    for token, mask, lp in zip(ids, masks, logprobs, strict=True):
        if mask:
            result[(context.hexdigest(), token, float(lp).hex())] += 1
        context.update(struct.pack(">q", token))
    return result


def audit_async(directory, arm):
    from trl.experimental.async_grpo.openenv_harness import _turns_from_trace
    from trl.experimental.async_grpo.async_rollout_worker import _chain_to_sequences
    from openenv.core.harness.capture.validate import validate_training_turn
    directory = Path(directory)
    reports = {}
    for path in sorted((directory / "rollouts").glob("*.json")):
        record = json.loads(path.read_text())
        raw = record.get("result")
        if not raw:
            continue
        if arm == "opencode":
            from data_agent_env import opencode_agent_turns, to_trace_entries
            from data_agent_env.models import DataAgentRolloutResult
            result = DataAgentRolloutResult.model_validate(raw)
            entries = opencode_agent_turns(to_trace_entries(result))
        else:
            from openenv.harbor.models import HarborRolloutResult
            from harbor_env.harness import to_trace_entries
            result = HarborRolloutResult.model_validate(raw)
            entries = to_trace_entries(result)
        if not entries:
            continue
        expected = collections.Counter()
        for entry in entries:
            p, c, lp, mask = (entry[k] for k in ("prompt_token_ids", "completion_token_ids", "per_token_logps", "loss_mask"))
            validate_training_turn(p, c, lp, mask)
            expected.update(positions(p + c, mask, [0.] * len(p) + lp))
        rows, _ = _chain_to_sequences(_turns_from_trace(entries), record["episode_id"], fork_threshold=0)
        retained = collections.Counter()
        for row in rows:
            retained.update(positions(row.input_ids, row.completion_mask, row.old_log_probs))
        passed = bool(expected) and retained == expected and any(lp < 0 for e in entries for lp in e["per_token_logps"])
        assert passed, f"Exact supervised positions were not retained: {path.name}"
        reports[record["episode_id"]] = {"rows": len(rows), "eligible_tokens": sum(expected.values()),
            "retained_tokens": sum(retained.values()), "rows_over_token_budget": sum(len(r.input_ids) > 131072 for r in rows),
            "tito_pass": passed}
    receipts = [json.loads(line) for line in (directory / "optimizer_rollouts.jsonl").read_text().splitlines() if line.strip()]
    consumed = 0
    for receipt in receipts:
        for row in receipt["rollouts"]:
            expected = reports[row["rollout_id"]]
            assert row["rows"] == expected["rows"]
            assert row["supervised_tokens"] == expected["retained_tokens"]
            consumed += 1
    assert consumed and reports, "No audited rollouts reached the optimizer"
    summary = {arm: {"completed_results": len(reports), "tito_pass": sum(r["tito_pass"] for r in reports.values()),
        "eligible_tokens": sum(r["eligible_tokens"] for r in reports.values()),
        "retained_tokens": sum(r["retained_tokens"] for r in reports.values()),
        "rows_over_token_budget": sum(r["rows_over_token_budget"] for r in reports.values()),
        "optimizer_rollouts_verified": consumed}}
    write_json(directory / "tito_summary.json", summary)
    return summary
