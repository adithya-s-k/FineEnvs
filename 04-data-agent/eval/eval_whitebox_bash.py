# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paired base-vs-checkpoint evaluation for the white-box bash/SETA environment.

WHY THIS EXISTS RATHER THAN REUSING eval_env.py
That harness drives `opencode` or `harbor`, both of which own their own agent loop, and it scores ONE
model at a time. This environment has no agent: TRL owns the loop during training, so an evaluation
has to supply one. The loop here is deliberately the same shape TRL uses -- generate, parse tool
calls, execute, append results, repeat -- so the policy is measured under the conditions it was
trained in. Evaluating it any other way measures a different thing and the number is not comparable.

PAIRED, ON A COMMON ITEM SET
Base and checkpoint run the SAME task indices with the SAME seed, and only tasks where BOTH produced
a grade are counted. Comparing two runs over different subsets is not a comparison; the black-box
evals in this repo are paired for the same reason and that is what makes the numbers sit beside each
other.

THE BASE ARM IS THE VALIDITY CHECK
Base is a fixed reference. If it scores far from its known value the run measured infrastructure, not
the policy -- two q3i evals in this project reported a "collapse" that was 976 E2B 404s, with the
base arm reading 0.0000 against its true 0.06-0.08. `--expect-base` fails the run loudly instead of
publishing a plausible wrong number.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def tool_schemas(env) -> list[dict]:
    """OpenAI tool schemas for the env's public methods, derived the way TRL derives them."""
    import inspect

    from transformers.utils import get_json_schema

    out = []
    for name, member in inspect.getmembers(env, predicate=inspect.ismethod):
        if name in {"reset", "get_reward"} or name.startswith("_"):
            continue
        out.append(get_json_schema(member))
    return out


def run_episode(make_env, *, split, index, llm_url, model, max_turns, timeout_s) -> dict:
    """One episode: drive the env with `model` until it submits or runs out of turns."""
    import openai

    env = make_env()
    client = openai.OpenAI(base_url=llm_url.rstrip("/") + "/v1", api_key="unused", timeout=timeout_s)
    rec = {"split": split, "index": index, "turns": 0, "tool_calls": 0,
           "submitted": None, "reward": None, "error": ""}
    try:
        task_text = env.reset(split=split, index=index)
        tools = tool_schemas(env)
        messages = [
            {"role": "system", "content":
             "You are a data-analysis agent working in a sandbox. Use the tools to inspect the "
             "files and compute the answer. When confident, call submit_solution with the value "
             "itself -- not the command that would produce it."},
            {"role": "user", "content": task_text},
        ]
        for _ in range(max_turns):
            rec["turns"] += 1
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=tools, tool_choice="auto",
                temperature=0.8, top_p=0.95, max_tokens=1024,
            )
            msg = resp.choices[0].message
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])]})
            if not msg.tool_calls:
                break                      # prose with no tool call: the agent has stopped working
            for tc in msg.tool_calls:
                rec["tool_calls"] += 1
                fn = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (TypeError, ValueError):
                    args = {}
                method = getattr(env, fn, None)
                result = (f"[error] no such tool {fn}" if method is None
                          else str(method(**args)))
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": fn,
                                 "content": result})
                if fn == "submit_solution":
                    rec["submitted"] = args.get("answer")
            if rec["submitted"] is not None:
                break
        rec["reward"] = env.get_reward()
    except Exception as exc:                # one dead episode must not take the pass with it
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec


def pass1(records: list[dict]) -> float:
    """Mean reward over GRADED records. An errored episode is excluded, never scored 0."""
    vals = [r["reward"] for r in records if r.get("reward") is not None and not r.get("error")]
    return sum(1.0 for v in vals if v and v >= 1.0) / len(vals) if vals else 0.0


def paired(base: dict, ckpt: dict) -> dict:
    """Paired comparison on the COMMON item set, with a normal-approx CI and a sign test."""
    keys = sorted(set(base) & set(ckpt))
    b = [base[k] for k in keys]
    c = [ckpt[k] for k in keys]
    n = len(keys)
    if n == 0:
        return {"n_tasks": 0, "diff": 0.0, "note": "no common graded tasks"}
    diffs = [ci - bi for bi, ci in zip(b, c)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
    se = math.sqrt(var / n)
    better = sum(1 for d in diffs if d > 0)
    worse = sum(1 for d in diffs if d < 0)
    # Two-sided sign test over the discordant pairs only.
    m = better + worse
    p = 1.0
    if m:
        tail = sum(math.comb(m, i) for i in range(0, min(better, worse) + 1)) / (2 ** m)
        p = min(1.0, 2 * tail)
    return {"n_tasks": n, "base_pass1": sum(b) / n, "ckpt_pass1": sum(c) / n,
            "diff": mean, "se": se, "ci_lo": mean - 1.96 * se, "ci_hi": mean + 1.96 * se,
            "better": better, "worse": worse, "tied": n - m, "sign_p": round(p, 4)}


def evaluate(tag, llm_url, model, indices, k, args) -> tuple[dict, list[dict]]:
    """Run every (task, sample) for one model and return per-task mean scores plus raw records."""
    from whitebox_bash import white_box_bash_env

    make_env = white_box_bash_env(args.server, toolsets=args.toolsets, step_limit=args.step_limit)
    jobs = [(i, s) for i in indices for s in range(k)]
    random.Random(args.seed).shuffle(jobs)   # unbiased partial coverage if the pass is cut short
    records: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_episode, make_env, split=args.split, index=i, llm_url=llm_url,
                            model=model, max_turns=args.max_turns,
                            timeout_s=args.timeout_s): (i, s) for i, s in jobs}
        for fut in as_completed(futs):
            records.append(fut.result())
            done += 1
            if done % 25 == 0:
                ex = sum(1 for r in records if r.get("error") or r.get("reward") is None)
                print(f"[{tag}] {done}/{len(jobs)}  excluded so far: {ex}", flush=True)
    per_task: dict[int, float] = {}
    for i in indices:
        rs = [r for r in records if r["index"] == i and not r.get("error")
              and r.get("reward") is not None]
        if rs:
            per_task[i] = sum(1.0 for r in rs if r["reward"] >= 1.0) / len(rs)
    return per_task, records


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default=os.environ.get("WHITE_BOX_BASH_URL", "http://127.0.0.1:8000"))
    p.add_argument("--base-url", required=True, help="vLLM serving the BASE model")
    p.add_argument("--base-model", required=True)
    p.add_argument("--ckpt-url", required=True, help="vLLM serving the CHECKPOINT")
    p.add_argument("--ckpt-model", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--n-tasks", type=int, default=60)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--toolsets", default="bash,seta")
    p.add_argument("--step-limit", type=int, default=14)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="")
    # Base is a FIXED reference. If it lands far from this, the run measured infrastructure.
    p.add_argument("--expect-base", type=float, default=0.0,
                   help="fail if base pass@1 is below this (0 disables)")
    args = p.parse_args()

    indices = list(range(args.n_tasks))
    t0 = time.time()
    base_scores, base_recs = evaluate("base", args.base_url, args.base_model, indices, args.k, args)
    ckpt_scores, ckpt_recs = evaluate("ckpt", args.ckpt_url, args.ckpt_model, indices, args.k, args)
    res = paired(base_scores, ckpt_scores)
    res.update({"split": args.split, "k": args.k, "seed": args.seed,
                "n_requested": args.n_tasks, "elapsed_s": round(time.time() - t0),
                "base_model": args.base_model, "ckpt_model": args.ckpt_model,
                "excluded_base": sum(1 for r in base_recs if r.get("error") or r.get("reward") is None),
                "excluded_ckpt": sum(1 for r in ckpt_recs if r.get("error") or r.get("reward") is None)})

    print("\n=== paired base vs checkpoint ===", flush=True)
    print(f"  common tasks : {res['n_tasks']} of {args.n_tasks} requested "
          f"(excluded: base {res['excluded_base']}, ckpt {res['excluded_ckpt']} rollouts)")
    print(f"  base  pass@1 : {res.get('base_pass1', 0):.4f}")
    print(f"  ckpt  pass@1 : {res.get('ckpt_pass1', 0):.4f}")
    print(f"  diff         : {res.get('diff', 0):+.4f}  "
          f"CI[{res.get('ci_lo', 0):+.3f},{res.get('ci_hi', 0):+.3f}]  p={res.get('sign_p')}")
    print(f"  better/worse/tied: {res.get('better')}/{res.get('worse')}/{res.get('tied')}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"summary": res, "base": base_recs, "ckpt": ckpt_recs}, fh)
        print(f"  wrote {args.out}")

    if args.expect_base and res.get("base_pass1", 0) < args.expect_base:
        print(f"\nFATAL: base pass@1 {res.get('base_pass1', 0):.4f} < expected {args.expect_base}. "
              "Base is a fixed reference -- this run measured infrastructure, not the policy. "
              "Discard it.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
