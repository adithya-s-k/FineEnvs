#!/usr/bin/env python
"""Real Harbor harness smoke with exact token/context retention through TRL.

Reuses OpenEnv capture, Harbor rollout, and TRL's actual trace reader and builder.
Artifacts contain task data, so keep the output under the experiment's ignored logs/.
Prefix drift is a packing metric, never the TITO admission test.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import importlib.metadata
import json
from pathlib import Path
import secrets
import shutil
import socket
import struct
import subprocess
import time

from harbor_env.harness import to_trace_entries
from openenv.core.harness.capture.export import export_session
from openenv.core.harness.capture.forwarding import GradioForwarder
from openenv.core.harness.capture.runner import CaptureServer
from openenv.core.harness.capture.upstream import training_sampling
from openenv.core.harness.capture.validate import validate_training_turn
from openenv.harbor.rollout import run_rollout
from openenv.harbor.seams import SEAMS
from trl.experimental.async_grpo.async_rollout_worker import _chain_to_sequences
from trl.experimental.async_grpo.openenv_harness import _turns_from_trace

ROOT = Path(__file__).resolve().parents[3]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def supervised_positions(ids, masks, logprobs):
    """Multiset of exact causal context, sampled id, and behavior logprob.

    Incremental hashing bounds memory even when a harness repeats a long prompt.
    Include multiplicity: equal generated tokens are separate training positions.
    """
    out = collections.Counter()
    context = hashlib.sha256()
    for token, mask, lp in zip(ids, masks, logprobs, strict=True):
        if mask:
            out[(context.hexdigest(), token, float(lp).hex())] += 1
        context.update(struct.pack(">q", token))
    return out


def audit(result, token_budget=40960):
    entries = to_trace_entries(result)
    expected = collections.Counter()
    policy = training_sampling({"temperature": 0.8})
    for entry in entries:
        p, c, lp, mask = (entry[k] for k in (
            "prompt_token_ids", "completion_token_ids", "per_token_logps", "loss_mask"))
        validate_training_turn(p, c, lp, mask)
        assert entry["metadata"]["sampling_params"] == policy, "sampling policy mismatch"
        expected.update(supervised_positions(p + c, mask, [0.0] * len(p) + lp))
    turns = _turns_from_trace(entries)
    rows, tally = _chain_to_sequences(turns, result.session_id, fork_threshold=0)
    retained = collections.Counter()
    for row in rows:
        retained.update(supervised_positions(row.input_ids, row.completion_mask, row.old_log_probs))
    fatal = [x for x in result.findings if "[FATAL]" in x]
    checks = {
        "train_tier": result.rollout_type == "train" and result.capture_level == "tokens",
        "nonempty_supervision": bool(expected),
        "exact_context_ids_logprobs_masks_retained": retained == expected,
        "no_capture_fatal": not fatal,
        "token_count_matches_export": sum(expected.values()) == result.n_trainable_tokens,
        "nonzero_logprobs_present": any(lp < 0 for e in entries for lp in e["per_token_logps"]),
    }
    action_entries = [e for e in entries if e["response"]["choices"][0]["message"].get("tool_calls")]
    return {
        "checks": checks, "tito_pass": all(checks.values()), "fatal": fatal,
        "entries": len(entries), "rows": len(rows), "eligible_tokens": sum(expected.values()),
        "retained_tokens": sum(retained.values()),
        "packed_tokens": sum(len(row.input_ids) for row in rows),
        "largest_row_tokens": max((len(row.input_ids) for row in rows), default=0),
        "rows_over_40960": sum(len(row.input_ids) > 40960 for row in rows),
        "token_budget": token_budget,
        "rows_over_token_budget": sum(len(row.input_ids) > token_budget for row in rows),
        "action_entries": len(action_entries),
        "action_tokens": sum(sum(e["loss_mask"]) for e in action_entries), "drift": tally,
    }, entries


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = []
    manifests = []
    source_tasks = sorted(p for p in args.tasks.iterdir() if (p / "task.toml").exists())
    for index in args.indices:
        source = source_tasks[index]
        target = args.output / "tasks" / source.name
        if not target.exists():
            shutil.copytree(source, target)
            # Isolated resource override, recorded below. Never modify the cached dataset.
            config = target / "task.toml"
            config.write_text(config.read_text().replace("memory_mb = 1024", "memory_mb = 4096"))
        files = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(source.rglob("*")) if p.is_file()}
        manifests.append({"index": index, "source": str(source), "files_sha256": files,
                          "resource_override": {"memory_mb": 4096},
                          "effective_task_sha256": hashlib.sha256((target / "task.toml").read_bytes()).hexdigest()})
        tasks.append((index, target))
    heads = {repo: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / repo, text=True).strip()
             for repo in ("OpenEnv", "trl", "HuggingEnvs")}
    write(args.output / "manifest.json", {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model, "inference": args.inference, "git_heads": heads,
        "working_tree_diff_sha256": {repo: hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=ROOT / repo)).hexdigest() for repo in heads},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "local_changes": True, "harbor_version": importlib.metadata.version("harbor"),
        "harnesses": args.harnesses, "tasks": manifests, "concurrency": args.concurrency,
        "sampling": training_sampling({"temperature": 0.8}), "model_call_limit": 17,
        "max_output_tokens": 4096, "agent_timeout_sec": 600,
        "scope": "Frozen-policy smoke; not optimizer, weight sync, or scale certification.",
    })
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]
    capture = CaptureServer(llm_url=args.inference, model=args.model, port=port,
                            max_output_tokens=4096, admin_key=secrets.token_hex(32))
    capture.start()
    forwarder = GradioForwarder()
    url = forwarder.start(port)
    print(json.dumps({"event": "proxy_ready", "port": port, "url": url}), flush=True)
    # Record the upstream response before ingest and dialect replay. No credentials or headers.
    completion = capture.inference.completion
    proxy_loop = None
    raw_dir = args.output / "engine_responses"
    raw_dir.mkdir(exist_ok=True)

    async def observed(request):
        nonlocal proxy_loop
        proxy_loop = asyncio.get_running_loop()
        response = await completion(request)
        key = hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()
        write(raw_dir / f"{key}.json", {"request": request, "response": response})
        return response

    capture.inference.completion = observed
    # Save graph evidence before run_rollout's finally releases each session, including cancellation.
    delete = capture.registry.delete

    def save_and_delete(sid):
        session = capture.registry.get(sid)
        if session is not None:
            doc = export_session(session, include_messages=True, include_discarded=True)
            write(args.output / f"capture-{sid}.json", doc)
        return delete(sid)

    capture.registry.delete = save_and_delete
    results = []
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run(harness, index, task):
        async with semaphore:
            case = f"{harness}-{index}"
            print(json.dumps({"event": "start", "case": case}), flush=True)
            record = {"harness": harness, "task_index": index, "task": task.name}
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(run_rollout(
                    task_dir=task, harness=harness, sandbox="e2b", registry=capture.registry,
                    intercept_url=url, model=args.model, trials_dir=args.output / "trials",
                    dataset="AdithyaSK/data_agent_rl_environment_train (local pinned tasks)",
                    agent_timeout_sec=600, agent_step_limit=17, session_prefix="tito-smoke",
                    inference=capture.inference, sampling={"temperature": 0.8}), timeout=1200)
                write(args.output / f"result-{case}.json", result.model_dump())
                record.update(ok=result.ok, reward=result.reward, error=result.error,
                              exception_type=result.exception_type, turns=result.n_turns,
                              roots=result.n_roots, findings=result.findings,
                              session_id=result.session_id, trial_name=result.trial_name)
                audited, entries = audit(result)
                record.update(audited)
                write(args.output / f"trace-{case}.json", entries)
            except Exception as exc:
                record.update(tito_pass=False, error=f"{type(exc).__name__}: {exc}")
            record["wall_s"] = round(time.monotonic() - started, 2)
            results.append(record)
            write(args.output / "matrix.json", results)
            print(json.dumps({"event": "done", **record}), flush=True)

    try:
        cases = [(h, i, p) for h in args.harnesses for i, p in tasks]
        # Warm the shared template once before concurrent trials can race its first build.
        await run(*cases[0])
        await asyncio.gather(*(run(*case) for case in cases[1:]))
    finally:
        write(args.output / "cleanup.json", {"remaining_sessions": capture.registry.list_ids(),
                                             "completed_cases": len(results)})
        # Harbor shields sandbox deletion. Let its already-scheduled cleanup finish before closing
        # this loop; only the capture client belongs to the server's separate loop.
        await asyncio.sleep(2)
        if proxy_loop is not None:
            future = asyncio.run_coroutine_threadsafe(capture.inference.aclose(), proxy_loop)
            await asyncio.wrap_future(future)
        forwarder.stop()
        capture.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--tasks", type=Path, default=Path("/admin/home/adithyaskolavi/.cache/openenv/harbor-datasets/AdithyaSK__data_agent_rl_environment_train/tasks"))
    p.add_argument("--indices", nargs="+", type=int, default=[8, 9])
    p.add_argument("--harnesses", nargs="+", default=[k for k, v in SEAMS.items() if v.status == "validated"])
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
