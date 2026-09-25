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

"""Run one Harbor rollout and check it is actually trainable.

WHY THIS SCRIPT EXISTS RATHER THAN A GLANCE AT THE LOG
Every failure it checks for is SILENT. An engine served without `--return-tokens-as-token-ids`
produces rollouts with a reward, a transcript and a plausible turn count that carry nothing to train
on; a consumer that re-renders prompts produces training rows that look fine and describe a
conversation the model never had. Both read as healthy until the loss step, and one of them cost two
production runs a night each.

    uv run python rollout.py --server http://127.0.0.1:8000 \\
        --llm-url http://127.0.0.1:8455/v1 --model Qwen/Qwen3.5-2B
"""

from __future__ import annotations

import argparse
import sys

# `openenv.harbor.client`, not the `harbor_env` env package. That package lives in OpenEnv's
# envs/ tree and is not published to PyPI, so it is unreachable from an installed environment --
# the same reason this project vendors its sandbox backends rather than importing opencode_env's.
# `harbor_env/__init__.py` only re-exports this class anyway.
from openenv.harbor.client import HarborEnv


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default="http://127.0.0.1:8000", help="a running harbor_env server")
    p.add_argument("--llm-url", default="", help="engine for this rollout; omit for the server's")
    p.add_argument("--model", default="", help="served model id")
    p.add_argument("--split", default="", help="dataset; omit for the server's first")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--harness", default="opencode")
    p.add_argument("--sandbox", default="e2b")
    p.add_argument("--step-limit", type=int, default=10)
    args = p.parse_args()

    env = HarborEnv(args.server)
    try:
        caps = env.capabilities()
        llm = caps.get("llm") or {}
        print(f"server     {args.server}")
        print(f"datasets   {caps.get('datasets')}")
        print(f"engine     {llm.get('url') or args.llm_url}  capture_level={llm.get('capture_level')}")
        if llm.get("capture_level") not in ("tokens", None) and not args.llm_url:
            print(
                "\nWARNING: the server's default engine is EVAL TIER. Rollouts will carry a reward "
                "and a trace but nothing trainable. Serve it with --return-tokens-as-token-ids "
                "--logprobs-mode processed_logprobs, or name a train-tier engine with --llm-url.",
                file=sys.stderr,
            )

        print(f"\nrunning {args.harness} on {args.sandbox}, task index {args.index} ...")
        result = env.run_rollout(
            split=args.split,
            task_index=args.index,
            harness=args.harness,
            sandbox=args.sandbox,
            llm_url=args.llm_url,
            model=args.model,
            agent_step_limit=args.step_limit,
        )
    finally:
        env.close()

    print(
        f"\nreward {result.reward}  rollout_type {result.rollout_type}  "
        f"turns {result.n_turns}  roots {result.n_roots}  capture_level {result.capture_level}"
    )
    if result.findings:
        print("findings:")
        for f in result.findings[:10]:
            print(f"  {f}")

    return _check(result)


# Above this, the capture graph forks instead of realigning; it is its own `fork_threshold_tokens`.
DRIFT_FORK_THRESHOLD = 1024


def _check(result) -> int:
    """The three properties that are silent when wrong. Returns a process exit code."""
    failures: list[str] = []

    # 1. Tier. `reward=None` is an UNGRADED rollout, not a zero, so it is reported separately: a
    #    trainer drops an ungraded rollout from the group baseline rather than treating it as a loss.
    if result.rollout_type != "train":
        failures.append(
            f"rollout_type is {result.rollout_type!r}: the engine did not return token ids, so this "
            "rollout carries nothing to train on"
        )
    if result.reward is None:
        print("\nnote: reward is None -- UNGRADED, not zero. The task's verifier did not run.")

    turns = [t for t in result.turns if getattr(t, "trainable", False)]
    if not turns:
        failures.append("no trainable turns came back")

    # 2. Every turn carries the ENGINE's tokenisation. Without it a consumer has to re-render the
    #    prompt, which matched the engine on 0 of 28 measured turns on Qwen3.5-4B.
    missing = [i for i, t in enumerate(turns) if not getattr(t, "prompt_token_ids", None)]
    if missing:
        failures.append(f"turns {missing[:5]} carried no prompt_token_ids")

    # 3. CHAINING, measured rather than asserted as byte equality.
    #
    # The tempting check is `turn k+1's prompt == turn k's prompt + completion`, exactly. It fails
    # legitimate rollouts: a harness that re-sends a `messages` list gets the engine's tokenisation
    # of the RECONSTRUCTED history, and Qwen3.5's template does not round-trip -- it emits
    # `<think>\n\n</think>\n\n` for the turn being generated and strips it from history. Measured
    # live, that drifted 6-8 tokens per transition and produced 3 graph roots for 3 turns.
    #
    # The size of the drift is what decides realign-versus-fork, and fork is the real failure: one
    # conversation becoming several short rollouts, each still training. Real opencode rollouts over
    # 60 steps: drift_tokens_mean 0.26, fork_frac 0.0000, 8.19 turns into 1.00 sample.
    drifts = []
    for a, b in zip(turns, turns[1:]):
        want = list(a.prompt_token_ids) + list(a.completion_token_ids)
        got = list(b.prompt_token_ids)[: len(want)]
        common = next((j for j, (x, y) in enumerate(zip(want, got)) if x != y), min(len(want), len(got)))
        drifts.append(len(want) - common)
    if drifts:
        print(f"chaining   drift per transition: mean {sum(drifts)/len(drifts):.2f}, max {max(drifts)}")
        if max(drifts) > DRIFT_FORK_THRESHOLD:
            failures.append(
                f"a transition drifted {max(drifts)} tokens (threshold {DRIFT_FORK_THRESHOLD}): the "
                "capture graph forks rather than realigns, fragmenting one rollout into several"
            )

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"\nOK: {len(turns)} trainable turns, engine tokenisation intact, prefixes chain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
