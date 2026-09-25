# Copyright 2026 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One rollout against a running data-agent environment, with the checks that matter.

    ./serve.sh &
    uv run python rollout.py --llm-url http://127.0.0.1:8455/v1 --model Qwen/Qwen3.5-2B

WHY THIS CHECKS RATHER THAN JUST PRINTS
Every failure it looks for is SILENT. An engine served without `--return-tokens-as-token-ids`
produces rollouts with a reward, a transcript and a plausible turn count that carry nothing to train
on. A consumer that re-renders prompts produces training rows that look fine and describe a
conversation the model never had -- measured on Qwen3.5-4B, a re-rendered prompt matched the engine
on 0 of 28 turns. Both read as healthy right up until the first weight update.
"""

from __future__ import annotations

import argparse
import sys

from data_agent_env import DataAgentEnv


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default="http://127.0.0.1:8200")
    p.add_argument("--llm-url", default="", help="engine for this rollout; omit for the server's")
    p.add_argument("--model", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--sandbox", default="e2b", choices=["e2b", "hf"])
    p.add_argument("--step-limit", type=int, default=10)
    p.add_argument(
        "--eval",
        action="store_true",
        help="allow a text-only engine. Rollouts come back scored but not trainable.",
    )
    args = p.parse_args()

    env = DataAgentEnv(args.server)
    try:
        caps = env.capabilities()
        usable = [n for n, ok in (caps.get("sandboxes") or {}).get("usable", {}).items() if ok]
        print(f"server     {args.server}")
        print(f"splits     {[s.get('name') for s in caps.get('splits') or []]}")
        print(f"sandboxes  usable here: {usable or 'NONE'}")
        print(f"concurrency {(caps.get('concurrency') or {}).get('max_concurrent_rollouts')}")
        if args.sandbox not in usable:
            print(
                f"\n{args.sandbox!r} is not usable here -- its SDK or its credential is missing. "
                "A rollout would fail after paying for a sandbox that could never have started.",
                file=sys.stderr,
            )
            return 2

        task = env.get_task(args.split, args.index)
        print(f"\ntask       {args.split}[{args.index}]  tier={task.get('difficulty_tier')}")
        print(f"           {str(task.get('instruction'))[:160]}...")

        print(f"\nrunning opencode in a {args.sandbox} sandbox ...")
        result = env.run_rollout(
            split=args.split,
            index=args.index,
            llm_url=args.llm_url,
            model=args.model,
            sandbox=args.sandbox,
            agent_step_limit=args.step_limit,
            require_tokens=not args.eval,
        )
    finally:
        env.close()

    print(
        f"\nreward {result.reward}  correctness {result.correctness}  "
        f"answer {result.answer!r} (from {result.answer_source}, graded_by {result.graded_by})"
    )
    print(f"turns {len(result.turns)}  tool calls {result.n_tool_calls}  type {result.rollout_type}")
    return _check(result, want_trainable=not args.eval)


def _check(result, *, want_trainable: bool) -> int:
    failures: list[str] = []

    # `reward=None` is UNGRADED, not zero. Reported, never counted as a loss: a trainer drops an
    # ungraded rollout from the group baseline, whereas a zero says the policy was wrong. Collapsing
    # the two turns a flaky sandbox into a training signal.
    if result.reward is None:
        print(f"\nUNGRADED: {result.metadata.get('error', 'the verifier did not run')}")
        return 1

    if not want_trainable:
        print("\nOK (eval): scored, not trainable -- as asked for with --eval.")
        return 0

    if result.rollout_type != "train":
        failures.append(
            "rollout_type is 'eval': the engine returned no token ids. Serve it with "
            "--return-tokens-as-token-ids --logprobs-mode processed_logprobs."
        )
    turns = [t for t in result.turns if t.trainable]
    if not turns:
        failures.append("no trainable turns came back")
    if any(not t.prompt_token_ids for t in turns):
        failures.append("a turn carried no prompt_token_ids -- a consumer would have to re-render it")

    # CHAINING. Capture already decided this; do not re-derive it.
    #
    # The tempting check is `turn k+1's prompt == turn k's prompt + completion`, exactly. That is
    # wrong as a pass/fail and it fails legitimate rollouts: a harness that re-sends a `messages`
    # list gets the engine's tokenisation of the RECONSTRUCTED history, and Qwen3.5's template does
    # not round-trip -- it emits `<think>\n\n</think>\n\n` for the turn being generated and strips
    # it from history. Measured live, that drifts 6-8 tokens per transition, which is small,
    # legitimate, and indistinguishable by eye from the real failure.
    #
    # Capture reports the real answer directly, as `per_turn_capture_only`: "every turn is its own
    # root ... this harness re-renders its prompt rather than appending, so rows are single-turn.
    # Tokens and logprobs are exact; multi-turn credit assignment is not available." That is a
    # structural fact about the graph, not an inference from token counts.
    #
    # It is a WARNING, not a failure. The rollout still trains -- just as N single-turn rows rather
    # than one multi-turn row. Real opencode rollouts do chain: 60 steps measured at
    # drift_tokens_mean 0.26, fork_frac 0.0000, 8.19 turns collapsing into 1.00 sample.
    findings = result.metadata.get("capture_findings") or []
    for f in findings:
        print(f"capture    {f}")
    if any("per_turn_capture_only" in f for f in findings):
        print(
            "\n  NOTE: turns did not chain, so this rollout trains as single-turn rows. Expected for "
            "a harness that re-renders its prompt; unexpected for opencode, which appends."
        )

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    total = sum(len(t.completion_token_ids) for t in turns)
    print(f"\nOK: {len(turns)} trainable turns, {total} completion tokens, prefixes chain exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
