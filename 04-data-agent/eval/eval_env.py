# Copyright 2026 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a model through either data-agent environment, keeping every trace.

    python eval_env.py --env opencode --server http://127.0.0.1:8200 \
        --llm-url $VLLM/v1 --model Qwen/Qwen3.5-2B --k 4

WHAT THIS REPORTS, AND WHY IN THIS ORDER
The exclusion count comes BEFORE the score, because a reward of `None` is not a zero. One suite once
emitted a null reward, the type rejected it, and 86 of 250 tasks vanished from scoring while the run
printed clean numbers over a third of the data. An ungraded rollout means the infrastructure failed;
a zero means the policy was wrong, and collapsing the two flatters or damns a model for no reason.

It also reports WHICH tasks were measured, not just how many. Coverage as a bare fraction hides
whether the missing items are random or a contiguous block: on an easy-to-hard ordered suite a 9B
once scored 0.830 on indices 0-146 and 0.553 on 147-249, so "140/250" was really an easy-prefix
score. Dispatch therefore runs in a FIXED SHUFFLED order, seeded, so that a partial run is an
unbiased sample of the split rather than its beginning.

And it reports mean turns next to the score, which diagnoses an agent faster than the score does:
9-11 turns is working, under 7 means it quit before reading the data, over 25 means it thrashed. Two
models can score alike for opposite reasons.

Records are written incrementally. A run that is killed keeps everything it had finished.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics as st
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def _opencode_client(server: str):
    from data_agent_env import DataAgentEnv

    return DataAgentEnv(server)


def _harbor_client(server: str):
    from openenv.harbor.client import HarborEnv

    return HarborEnv(server)


def _run_opencode(client, *, split, index, llm_url, model, sandbox, step_limit, timeout_s):
    r = client.run_rollout(
        split=split, index=index, llm_url=llm_url, model=model, sandbox=sandbox,
        agent_step_limit=step_limit, agent_timeout_s=timeout_s,
        # An EVAL rollout does not need token ids -- a text-only endpoint is a fine eval backend, and
        # refusing it would rule out every hosted provider. Ids still come back when the engine has
        # them, which is why the traces below are usable for training diagnostics too.
        require_tokens=False,
    )
    turns = [
        {
            "turn": t.turn,
            "n_prompt_ids": len(t.prompt_token_ids),
            "n_completion_ids": len(t.completion_token_ids),
            "n_logps": len(t.per_token_logps),
            "trainable": t.trainable,
            "finish_reason": t.finish_reason,
            "n_tool_calls": len(t.tool_calls),
            "text": t.text,
            "prompt_token_ids": list(t.prompt_token_ids),
            "completion_token_ids": list(t.completion_token_ids),
            "per_token_logps": list(t.per_token_logps),
        }
        for t in r.turns
    ]
    return {
        "reward": r.reward,
        "correctness": r.correctness,
        "answer": r.answer,
        "answer_source": r.answer_source,
        "graded_by": r.graded_by,
        "rollout_type": r.rollout_type,
        "n_turns": len(r.turns),
        "n_tool_calls": r.n_tool_calls,
        "timed_out": r.timed_out,
        "findings": (r.metadata or {}).get("capture_findings") or [],
        "error": (r.metadata or {}).get("error"),
        "metadata": r.metadata,
    }, turns


def _run_harbor(client, *, split, index, llm_url, model, sandbox, step_limit, timeout_s):
    r = client.run_rollout(
        split=split, task_index=index, harness="opencode", sandbox=sandbox,
        llm_url=llm_url, model=model, agent_step_limit=step_limit,
        agent_timeout_sec=timeout_s,
    )
    turns = [
        {
            "turn": getattr(t, "turn", i),
            "n_prompt_ids": len(getattr(t, "prompt_token_ids", []) or []),
            "n_completion_ids": len(getattr(t, "completion_token_ids", []) or []),
            "n_logps": len(getattr(t, "per_token_logps", []) or []),
            "trainable": bool(getattr(t, "trainable", False)),
            "finish_reason": getattr(t, "finish_reason", None),
            "prompt_token_ids": list(getattr(t, "prompt_token_ids", []) or []),
            "completion_token_ids": list(getattr(t, "completion_token_ids", []) or []),
            "per_token_logps": list(getattr(t, "per_token_logps", []) or []),
        }
        for i, t in enumerate(getattr(r, "turns", []) or [])
    ]
    return {
        "reward": r.reward,
        "correctness": r.reward,  # harbor grades with the task's own verifier; reward IS the grade
        "answer": None,
        "answer_source": "harbor-verifier",
        "graded_by": "task",
        "rollout_type": getattr(r, "rollout_type", ""),
        "n_turns": getattr(r, "n_turns", len(turns)),
        "n_roots": getattr(r, "n_roots", None),
        "n_tool_calls": None,
        "timed_out": None,
        "findings": list(getattr(r, "findings", []) or []),
        "error": getattr(r, "error", None),
        "metadata": {"trial_name": getattr(r, "trial_name", None),
                     "capture_level": getattr(r, "capture_level", None)},
    }, turns


ADAPTERS = {
    "opencode": (_opencode_client, _run_opencode),
    "harbor": (_harbor_client, _run_harbor),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=sorted(ADAPTERS), required=True)
    p.add_argument("--server", required=True)
    p.add_argument("--llm-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="eval")
    p.add_argument("--k", type=int, default=4, help="independent passes per task")
    p.add_argument("--n-tasks", type=int, default=0, help="0 = the whole split")
    p.add_argument("--seed", type=int, default=0, help="fixes the shuffled dispatch order")
    p.add_argument("--sandbox", default="e2b")
    p.add_argument("--step-limit", type=int, default=10)
    p.add_argument("--timeout-s", type=float, default=600.0)
    # Well under the ~200 where the capture proxy's /health starves and the 320 where it crashed.
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--out", default="")
    args = p.parse_args()

    make_client, run_one = ADAPTERS[args.env]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = pathlib.Path(args.out or f"logs/eval-{args.env}-{stamp}")
    (out / "traces").mkdir(parents=True, exist_ok=True)
    records_path = out / "records.jsonl"

    probe = make_client(args.server)
    try:
        n_total = probe.num_tasks(args.split)
    finally:
        probe.close()
    n = args.n_tasks or n_total

    # FIXED SHUFFLED ORDER. A partial run must be an unbiased sample of the split, not its prefix.
    order = list(range(n_total))
    random.Random(args.seed).shuffle(order)
    indices = order[:n]

    # DISPATCHED IN PASSES, WITH A BARRIER BETWEEN THEM, and that is not for tidiness.
    #
    # E2B builds a template per task on first use, and CONCURRENT FIRST USE RACES THAT BUILD: the
    # losers get `404: tag 'default' does not exist for template ...`. Running all k samples of a task
    # at once is exactly that race -- k-1 of them fail in ~7 s and the whole split burns without
    # scoring anything. Measured at concurrency 64 on a cold suite.
    #
    # Pass 0 therefore warms every template, and only then do passes 1..k-1 run. Same total work, no
    # intra-task race, and it has a second benefit: after pass 0 there is already a complete pass@1
    # over the split, so a run killed partway still yields a usable number.
    passes = [[(i, s) for i in indices] for s in range(args.k)]
    print(f"env        {args.env}  server {args.server}")
    print(f"model      {args.model}  via {args.llm_url}")
    print(f"split      {args.split}: {n} of {n_total} tasks, k={args.k} -> {n * args.k} rollouts"
          f" in {args.k} pass(es)")
    print(f"dispatch   shuffled, seed {args.seed}; first indices {indices[:8]}")
    print(f"out        {out}")
    print(f"concurrency {args.concurrency}\n", flush=True)

    lock = threading.Lock()
    done = [0]
    started = time.time()

    def work(job):
        index, sample = job
        client = make_client(args.server)  # one client per rollout; a shared one cannot multiplex
        rec: dict[str, Any] = {"task_index": index, "sample": sample, "model": args.model,
                               "env": args.env, "split": args.split}
        t0 = time.time()
        try:
            result, turns = run_one(
                client, split=args.split, index=index, llm_url=args.llm_url, model=args.model,
                sandbox=args.sandbox, step_limit=args.step_limit, timeout_s=args.timeout_s,
            )
            rec.update(result)
            (out / "traces" / f"{index:04d}-{sample}.json").write_text(
                json.dumps({"record": {k: v for k, v in rec.items() if k != "metadata"},
                            "turns": turns}, indent=None)
            )
        except Exception as exc:  # noqa: BLE001 -- one bad rollout must not end the eval
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["reward"] = None
        finally:
            try:
                client.close()
            except Exception:
                pass
        rec["seconds"] = round(time.time() - t0, 1)
        with lock:
            with open(records_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            done[0] += 1
            if done[0] % 10 == 0 or done[0] == n * args.k:
                rate = done[0] / max(1e-9, time.time() - started) * 60
                print(f"  {done[0]}/{n * args.k} rollouts  ({rate:.1f}/min)", flush=True)
        return rec

    for pass_no, jobs_in_pass in enumerate(passes):
        note = " (warms every E2B template; later passes cannot race it)" if pass_no == 0 else ""
        print(f"\n  -- pass {pass_no + 1}/{len(passes)}{note}", flush=True)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(work, j) for j in jobs_in_pass]
            for f in as_completed(futures):
                f.result()

    return report(records_path, out)


def report(records_path: pathlib.Path, out: pathlib.Path) -> int:
    recs = [json.loads(l) for l in open(records_path) if l.strip()]
    # EXCLUSIONS FIRST. reward=None is UNGRADED, never a zero.
    excluded = [r for r in recs if r.get("reward") is None]
    scored = [r for r in recs if r.get("reward") is not None]
    print("\n" + "=" * 78)
    print(f"  rollouts        {len(recs)}")
    print(f"  EXCLUDED        {len(excluded)} ({len(excluded)/max(1,len(recs)):.1%}) -- ungraded, not zero")
    reasons: dict[str, int] = {}
    for r in excluded:
        key = str(r.get("error") or "unknown")[:70]
        reasons[key] = reasons.get(key, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])[:6]:
        print(f"                    {v:4d}  {k}")
    if not scored:
        print("\n  nothing was graded; no score to report.")
        return 1

    by_task: dict[int, list[float]] = {}
    for r in scored:
        by_task.setdefault(r["task_index"], []).append(float(r["reward"]))
    idx = sorted(by_task)
    pass1 = st.mean([st.mean(v) for v in by_task.values()])
    passk = st.mean([1.0 if max(v) > 0 else 0.0 for v in by_task.values()])
    print(f"\n  tasks measured  {len(by_task)}   indices {idx[0]}..{idx[-1]} "
          f"(shuffled dispatch, so this is an unbiased sample)")
    print(f"  pass@1 (mean)   {pass1:.4f}")
    print(f"  pass@k (any>0)  {passk:.4f}")
    turns = [r["n_turns"] for r in scored if r.get("n_turns") is not None]
    if turns:
        print(f"  mean turns      {st.mean(turns):.2f}   "
              f"(9-11 working, <7 quit early, >25 thrashing)")
    tc = [r["n_tool_calls"] for r in scored if r.get("n_tool_calls") is not None]
    if tc:
        print(f"  mean tool calls {st.mean(tc):.2f}")
    noans = sum(1 for r in scored if not r.get("answer") and r.get("answer_source") in (None, "none"))
    if noans:
        print(f"  no answer filed {noans}/{len(scored)} ({noans/len(scored):.0%})")
    tiers = {r.get("rollout_type") for r in scored}
    print(f"  rollout_type    {sorted(t for t in tiers if t)}")
    finds: dict[str, int] = {}
    for r in recs:
        for f in r.get("findings") or []:
            key = f.split(":")[0][:60]
            finds[key] = finds.get(key, 0) + 1
    if finds:
        print("  capture findings")
        for k, v in sorted(finds.items(), key=lambda kv: -kv[1])[:5]:
            print(f"                    {v:4d}  {k}")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n  traces          {len(list((out/'traces').glob('*.json')))} files, "
          f"{size/1e6:.1f} MB under {out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
