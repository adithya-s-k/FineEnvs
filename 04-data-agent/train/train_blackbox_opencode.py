# Copyright 2026 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AsyncGRPO on `blackbox-opencode`. TRL hosts vLLM and orchestrates; the env does the rest.

WHAT IS ABSENT IS THE POINT. The run this reproduces needed two monkeypatches and got one of them
silently wrong:

  * `prompt_ids_patch` rebound TRL's `_turns_from_trace` to prefer the engine's prompt ids. It never
    took effect. The rollout loop runs in a multiprocessing child created with `spawn`, which
    re-imports every module, so a parent-side rebind of a module function is simply lost -- the patch
    logged "installed" and its counters stayed at zero across 12,000 rollouts. Both production runs
    trained on re-tokenised prompts for a night.
  * `think_template_patch` mutated the tokenizer OBJECT, which IS pickled into the child, so that one
    survived. That asymmetry is the whole reason the first went unnoticed.

Neither is here. The ids come from the environment on the wire, and TRL raises rather than falling
back to a re-render. There is also no `chat_template_kwargs`: in loop-owning mode `_sample_turn`
never runs, so nothing applies a chat template at all.

The engine MUST be the trainer's own vLLM. That is what makes the rollouts on-policy -- the agent
calls the same weights the optimizer is updating, through the capture proxy.
"""

from __future__ import annotations

import argparse
import os

from data_agent_env import DataAgentSessionFactory, opencode_agent_turns
from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker, has_tool_call


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default="http://127.0.0.1:8200", help="a running blackbox-opencode")
    p.add_argument("--vllm-url", required=True, help="the trainer's OWN vLLM; on-policy depends on it")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--split", default="train")
    # The +0.2028 arm: 125 easy prompts first, then medium with hard sprinkled through. Without
    # it the run meets the hard tiers at step 0, where a group of all-zero rollouts gives no
    # gradient at all.
    p.add_argument("--curriculum", default="warmup:125")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sandbox", default="e2b", choices=["e2b", "hf"])
    # The arm that produced +0.2028 [+0.146,+0.259] at step 200 and held it at 400.
    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-staleness", type=int, default=4)
    # 17: the value the reference +0.2028 run used, read off its own launch line --
    #     STEP_PERSIST_CAP=0 STEP_SOFT_CAP=0 STEP_HARD_CAP=17 STEP_FORCE_TOOL=
    # -- where the hard cap fired 4,627 times, so it bound constantly rather than sitting unused.
    #
    # It has to sit at or below the reward's step_budget of 30: above it there is a band where the
    # agent is allowed to act and punished for acting, and the policy escapes by not acting at all,
    # which under train_turn_fn=has_tool_call yields no rows and therefore no gradient.
    #
    # 10 was far too tight -- 197 of 224 eval rollouts (88%) cut off mid-task, turns pinned at 9. But
    # 25 is not the reference either, and the difference shows up in the DYNAMICS rather than as an
    # error: longer rollouts mean fewer complete per optimizer step, measured at 9.8 samples/step
    # against the reference's 16.1, which is a different effective batch and a different rate of
    # consuming the curriculum.
    #
    # The reference deliberately ran with BOTH nudges off (SOFT_CAP=0, PERSIST_CAP=0), so the absence
    # of prompt injection here matches it rather than departing from it.
    p.add_argument("--agent-step-limit", type=int, default=17)
    # PINNED, and not optional. Unset, token_budget defaults to the vLLM server's max_model_len --
    # 131072 here -- which tripled the trained row and killed job 69906 with torch.OutOfMemoryError in
    # fla/ops/gated_delta_rule/chunk.py before step 1. At 40960 the rows already reach 40,870 (99.8%),
    # so this is the measured ceiling for a 4B-class model on one 80 GB card, not a safety margin.
    p.add_argument("--token-budget", type=int, default=40960)
    # 900, against agent_timeout_s=600. The default 300 killed job 69319 with "heartbeat stale: 302s >
    # 300s; child is hung" on a worker that was not hung but BUSY: the worker ticks its heartbeat at
    # the top of the dispatch loop, which does not re-iterate while every max_inflight slot is full.
    p.add_argument("--heartbeat-stale-after-s", type=float, default=900.0)
    # opencode asks for 32,000 output tokens and capture clamps it to 8192; the TRL default is 2048.
    p.add_argument("--max-completion-length", type=int, default=16384)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--optim", default="paged_adamw_8bit")
    # bfloat16, to MATCH THE SERVER. AsyncGRPOConfig defaults dtype="float32" deliberately -- TRL
    # prefers fp32 on the trainer because the training-inference mismatch is sensitive to it -- but its
    # own docstring adds that closing that gap end to end "also requires serving the vLLM server in the
    # same dtype", and a precision GAP BIASES THE IMPORTANCE RATIO
    # (https://huggingface.co/papers/2510.26788).
    #
    # TRL's preferred direction, serving fp32, is impossible here and that was measured rather than
    # assumed: Qwen3.5 is hybrid Gated-DeltaNet and vLLM asserts
    #     ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16.
    # (qwen_gdn_linear_attn.py:1165, job 72978). So the match is made on the trainer's side.
    #
    # Left at the default, job 72939 warned "serves in bfloat16 but the weights sent to it are
    # float32" with embed_tokens.weight at 2.54 GB against a 1 GB transfer buffer. Halving the
    # optimizer state is a side benefit on a card this work has already OOMed.
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--output-dir", default="")
    p.add_argument("--run-name", default="")
    p.add_argument("--project", default="data-agent-blackbox")
    args = p.parse_args()

    # Trackio keys a run by name inside a project, so two relaunches of one config land on top of each
    # other and the earlier metrics read as part of the later run's history -- worst exactly when
    # relaunching after a crash. Stamping with the job id keeps them apart.
    stamp = os.environ.get("SLURM_JOB_ID", "local")
    run_name = args.run_name or f"{args.model.split('/')[-1]}-lr{args.learning_rate:g}-{stamp}"
    output_dir = args.output_dir or f"runs/{run_name}"

    factory = DataAgentSessionFactory(
        args.server,
        split=args.split,
        llm_url=args.vllm_url,
        model=args.model,
        sandbox=args.sandbox,
        agent_step_limit=args.agent_step_limit,
        curriculum=args.curriculum,
        seed=args.seed,
    )
    # Built FROM THE FACTORY so the instruction the trainer sends is one the server can resolve back
    # to a task. All `num_generations` rollouts of a group share a row, so they get the same task and
    # the group baseline is well formed without any seed plumbing.
    dataset = Dataset.from_list(factory.prompt_rows())
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"server    {args.server}")
    print(f"vllm      {args.vllm_url}   model {args.model}")
    print(f"tasks     {len(dataset)} from {args.split} [{args.curriculum or 'shuffled'}], sandbox {args.sandbox}")
    print(f"run       {run_name} -> {output_dir}")
    print(f"budgets   token_budget={args.token_budget} max_completion={args.max_completion_length} "
          f"heartbeat={args.heartbeat_stale_after_s:g}s agent_steps={args.agent_step_limit} "
          f"dtype={args.dtype}")

    worker = HarnessRolloutWorker(
        harness_session_factory=factory,
        harness_adapter=None,  # loop-owning: the agent drives itself; we read what it did
        # Reinforce turns that took an ACTION rather than prose -- right for an agent whose job is to
        # inspect data and write a file. It works only because the env hands TRL tool calls in the
        # nested OpenAI shape; flattened, `has_tool_call` is False for every turn and the whole
        # rollout is silently discarded.
        train_turn_fn=has_tool_call,
        # Drop opencode's own title/summarizer calls. An earlier revision left this out on the theory
        # that capture removes aux roots structurally -- it does not, and the run that assumed so
        # collapsed: fork_frac 0.02-0.06 (reference: 0), drift_tokens_max 32,770 (reference: 0),
        # samples_per_rollout up to 1.31 (reference: exactly 1.0), and the policy trained on title and
        # summary tokens carrying the task's advantage. See `opencode_agent_turns` for the full
        # measurement.
        agent_turn_fn=opencode_agent_turns,
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],  # the environment's verify() is the reward
        processing_class=tokenizer,
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        log_completions=True,
        num_completions_to_print=2,
    )

    AsyncGRPOTrainer(
        model=args.model,
        args=AsyncGRPOConfig(
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=args.save_steps,
            # Keep every checkpoint: the eval watcher picks them up asynchronously, and a
            # save_total_limit would delete one out from under a queued evaluation.
            save_total_limit=None,
            num_generations=args.num_generations,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
            max_completion_length=args.max_completion_length,
            token_budget=args.token_budget,
            heartbeat_stale_after_s=args.heartbeat_stale_after_s,
            optim=args.optim,
            dtype=args.dtype,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            max_staleness=args.max_staleness,
            vllm_server_base_url=args.vllm_url,
            bf16=True,
            gradient_checkpointing=True,
            # Required: the reentrant checkpointer does not see inputs that reach a block through
            # anything but positional args.
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="trackio",
            project=args.project,
            run_name=run_name,
            # Every rollout costs a sandbox and minutes, so nothing is logged in arrears.
            logging_steps=1,
            log_completions=True,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_worker=worker,
    ).train()


if __name__ == "__main__":
    main()
