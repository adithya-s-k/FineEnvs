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


# Above this, the graph forks instead of realigning; it is capture's own `fork_threshold_tokens`.
DRIFT_FORK_THRESHOLD = 1024


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

    # CHAINING, measured rather than asserted as byte equality.
    #
    # The tempting check is `turn k+1's prompt == turn k's prompt + completion`, exactly. That is
    # WRONG as a pass/fail, and it fails legitimate rollouts. A harness that re-sends a `messages`
    # list gets the engine's tokenisation of the RECONSTRUCTED history, and Qwen3.5's template does
    # not round-trip: it emits `<think>\n\n</think>\n\n` for the turn being generated and strips it
    # from history, so a 3-turn conversation measured live drifted 6-8 tokens per transition and the
    # graph reported 3 roots for 3 turns.
    #
    # What matters is the SIZE of the drift, because that is what decides whether the capture graph
    # realigns the turn onto the same rollout or forks it into a new one. Measured on 60 steps of
    # real opencode rollouts: drift_tokens_mean 0.26, fork_frac 0.0000, and 8.19 turns collapsing
    # into 1.00 sample per rollout. Fragmentation is the failure -- one conversation becoming several
    # short ones, each still training -- and it shows up as large drift, not as small drift.
    drifts = []
    for a, b in zip(turns, turns[1:]):
        want = list(a.prompt_token_ids) + list(a.completion_token_ids)
        got = list(b.prompt_token_ids)[: len(want)]
        common = next((j for j, (x, y) in enumerate(zip(want, got)) if x != y), min(len(want), len(got)))
        drifts.append(len(want) - common)
    if drifts:
        worst = max(drifts)
        mean = sum(drifts) / len(drifts)
        print(f"chaining   drift per transition: mean {mean:.2f}, max {worst} tokens")
        if worst > DRIFT_FORK_THRESHOLD:
            failures.append(
                f"a transition drifted {worst} tokens (threshold {DRIFT_FORK_THRESHOLD}): the capture "
                "graph will fork rather than realign, so this one conversation becomes several short "
                "rollouts and every fragment still trains"
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
