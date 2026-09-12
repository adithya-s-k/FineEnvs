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

from data_agent_env import DataAgentSessionFactory
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
    p.add_argument("--agent-step-limit", type=int, default=10)
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

    worker = HarnessRolloutWorker(
        harness_session_factory=factory,
        harness_adapter=None,  # loop-owning: the agent drives itself; we read what it did
        # Reinforce turns that took an ACTION rather than prose -- right for an agent whose job is to
        # inspect data and write a file. It works only because the env hands TRL tool calls in the
        # nested OpenAI shape; flattened, `has_tool_call` is False for every turn and the whole
        # rollout is silently discarded.
        train_turn_fn=has_tool_call,
        # No `agent_turn_fn`: capture already dropped auxiliary calls and de-duplicated forked paths
        # structurally, which a flat trace cannot do.
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],  # the environment's verify() is the reward
        processing_class=tokenizer,
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
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
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
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
