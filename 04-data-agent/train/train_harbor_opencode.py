"""Async GRPO on the data-agent task set via OpenEnv x Harbor, opencode harness.

A deliberate replication of HuggingEnvs/04-data-agent/train/train_blackbox_opencode.py -- the arm
that reached +0.2343 (CI [+0.178, +0.291], p=0.0) on Qwen3.5-2B -- with ONE variable changed: the
environment is Harbor through the OpenEnv capture proxy instead of the bespoke blackbox-opencode env.
Every training knob below is the reference's value, so a difference in outcome is attributable to the
environment and not to the recipe.

TOKEN-IN-TOKEN-OUT, NO RE-RENDER. `to_trace_entries` (envs/harbor_env/harness.py) carries the
engine's own `prompt_token_ids`, and TRL's `_turns_from_trace` reads them and RAISES if absent. A
local re-render matched the engine on 0 of 28 measured turns, so this path is not optional.
Verify from the metrics, not from reading this file: rollout/fork_frac == 0,
rollout/samples_per_rollout == 1.00, rollout/drift_tokens_max == 0.

WHY NO agent_turn_fn, AND HOW TO KNOW IF THAT IS WRONG. The reference passes `opencode_agent_turns`
to strip opencode's title/summarizer calls, anchored on the first tool-enabled turn's SYSTEM PROMPT.
That filter cannot be ported: Harbor's TraceEntry carries no `request`, and HarborTurn has no
system_digest. It should not need to be. Harbor assigns roles structurally in capture/export.py --
a path that never uses tools is AUXILIARY, `trainable` requires role == AGENT, and
`to_trace_entries` skips anything not trainable. So the aux calls are dropped BEFORE TRL sees them.

That is a claim about Harbor, and claims get checked: if `rollout/fork_frac` is non-zero AT STEP 1
(structural, not a later collapse) or `samples_per_rollout` != 1.00, the drop did not happen and
`--agent-turn-filter tools` supplies a fallback that keeps only turns that called a tool.

THE TRAPDOOR THIS SCRIPT IS SAFE FROM, AND WHY IT IS WORTH KNOWING ANYWAY. Jobs 72452/72473 wedged
at step 7 and 10 of 100, having spent 4,076 E2B sandboxes on 11 productive groups (294 and 201 EMPTY
groups). Two individually reasonable settings combined:

    reward        = correctness + 0.1 * clamp(1 - n_tool_calls/15, 0, 1)   # UNGATED efficiency
    train_turn_fn = has_tool_call

Zero tool calls scores the MAXIMUM efficiency bonus, so inaction is the best move available to a
policy that cannot solve the task (0.100 vs 0.033 for a real attempt that fails). The policy learns
to stop calling tools -- and `has_tool_call` then yields NO trainable turns, so the group is empty
and the run starves while the logs keep moving.

The first leg is absent here: the `_train` suite emits a single float from `grader.py` with no
efficiency term at all (verified: zero occurrences across all 2,238 task graders), so `reward_funcs=[]`
means pure correctness. The SECOND leg is still present. A model that makes no tool calls for any
reason -- including simply being too weak for the task -- still produces empty groups. That is what
banded tasks are for: pick indices the model can SOMETIMES solve, so reward_std > 0 and the group
carries gradient. Watch `reward_std` and the empty-group count from step 1.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker, has_tool_call

TRAIN_SPLIT = "AdithyaSK/data_agent_rl_environment_train"


def tool_calling_turns(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fallback `agent_turn_fn`: keep only turns that actually called a tool.

    The Harbor-native stand-in for the reference's system-prompt anchor, which cannot be ported
    (no `request` on a Harbor TraceEntry). opencode's bookkeeping calls -- the conversation-title
    generator and the context summarizer -- use no tools, so `metadata.n_tools > 0` separates them
    from real agent steps. Weaker than anchoring on the system prompt, which is why it is OFF by
    default and gated on measured fork_frac rather than switched on out of caution.
    """
    return [e for e in trace if ((e.get("metadata") or {}).get("n_tools") or 0) > 0]


def build(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8200", help="a running `openenv harbor serve`")
    p.add_argument("--vllm-url", required=True, help="the engine AsyncGRPO also syncs weights into")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--split", default=TRAIN_SPLIT)
    p.add_argument("--harness", default="opencode")
    p.add_argument("--sandbox", default="e2b")
    p.add_argument("--reward-key", default="", help="'' lets the server pick; required on a multi-reward suite")
    p.add_argument("--task-indices", default="", help="comma-separated, or @file")
    p.add_argument("--n-tasks", type=int, default=0, help="0 = the whole split")
    p.add_argument("--agent-turn-filter", default="none", choices=["none", "tools"],
                   help="'tools' keeps only turns with n_tools>0; use ONLY if fork_frac != 0 at step 1")

    # ---- the reference's values, unchanged ---------------------------------------------------
    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--max-staleness", type=int, default=4)
    p.add_argument("--optim", default="paged_adamw_8bit")
    # Pinned, and the SAME value must reach `vllm serve --override-generation-config`. opencode sends
    # no sampling params and Qwen3.5 ships no generation_config.json, so an unpinned engine samples at
    # 1.0 while the trainer divides logits by this -- gradients against a distribution that never
    # produced the samples. Measured unpinned: entropy 0.229 -> 0.587 over 24 steps, reward 0.592 -> 0.216.
    p.add_argument("--temperature", type=float, default=0.8)
    # NEUTRAL, and this is a deliberate DEPARTURE from the reference's 0.95. processed_logprobs are
    # taken AFTER truncation, so a truncating top_p renormalises every captured logprob over the kept
    # set while the trainer recomputes full-vocab; the step-0 importance ratio then lands at
    # kept_mass rather than 1. Validated: `ratio` moved from the reference's 0.985-0.993 signature to
    # 0.9984-0.9999 once this was 1.0.
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    # 17, read off the reference run's own STEP_HARD_CAP, which fired 4,627 times. Must sit at or
    # below the reward's step_budget, else there is a band where acting is allowed and punished and
    # the policy escapes by not acting -- which under has_tool_call yields no rows and no gradient.
    p.add_argument("--agent-step-limit", type=int, default=17)
    p.add_argument("--agent-timeout", type=float, default=600.0)
    # PINNED, not optional: unset, token_budget falls back to the engine's max_model_len, which
    # tripled the trained row and OOMed job 69906 in fla/ops/gated_delta_rule/chunk.py before step 1.
    p.add_argument("--token-budget", type=int, default=40960)
    # 900 against agent_timeout 600. The 300 default killed job 69319 on a worker that was BUSY, not hung.
    p.add_argument("--heartbeat-stale-after-s", type=float, default=900.0)
    p.add_argument("--max-completion-length", type=int, default=16384)
    # MATCH THE SERVER. AsyncGRPOConfig defaults to float32; a precision gap biases the importance ratio.
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--output-dir", default="")
    p.add_argument("--run-name", default="")
    p.add_argument("--project", default="data-agent-harbor-opencode")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def indices_of(spec: str) -> list[int] | None:
    """`@file` form exists because `sbatch --export=ALL,VAR=a,b,c` truncates at the first comma,
    silently -- the job runs with a task list it was never given."""
    if not spec:
        return None
    if spec.startswith("@"):
        spec = open(spec[1:]).read()
    out, seen = [], set()
    for tok in spec.replace("\n", ",").split(","):
        tok = tok.strip()
        if tok and int(tok) not in seen:
            seen.add(int(tok)); out.append(int(tok))
    return out or None


def main() -> None:
    args = build()
    from harbor_env.harness import HarborSessionFactory

    factory = HarborSessionFactory(
        args.server,
        split=args.split,
        harness=args.harness,
        sandbox=args.sandbox,
        # THE SAME engine the trainer syncs weights into. That is what makes the rollouts on-policy:
        # the agent's calls and the weight updates go to one vLLM. It must be the node's ROUTABLE
        # address -- the harbor server probes it from ANOTHER host, and with localhost the probe
        # fails, the tier grades `text`, and every rollout comes back with no trainable turns.
        llm_url=args.vllm_url,
        model=args.model,
        reward_key=args.reward_key,
        agent_timeout_sec=args.agent_timeout,
        agent_step_limit=args.agent_step_limit,
        indices=indices_of(args.task_indices),
        num_tasks=args.n_tasks or None,
    )

    # Built FROM the factory so the instruction TRL sends is one the server can resolve: `create()`
    # hashes the prompt back to a task index and RAISES on a miss rather than silently running task 0.
    rows = factory.prompt_rows()
    dataset = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    run_name = args.run_name or (
        f"{args.model.split('/')[-1]}-{args.harness}-harbor-{args.max_steps}steps"
        # Stamped with the job id: trackio keys a run by name inside a project, so relaunches
        # otherwise stack on top of each other.
        f"-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    out_dir = args.output_dir or f"/fsx/{os.environ.get('USER','x')}/runs/agrpo_harbor/{run_name}"

    print(f"model     {args.model}")
    print(f"server    {args.server}   vllm {args.vllm_url}")
    print(f"rollouts  {args.harness} on {args.sandbox}, {args.num_generations}x{args.max_inflight}")
    print(f"tasks     {len(dataset)} from {args.split}")
    print(f"sampling  temperature={args.temperature} top_p={args.top_p} top_k={args.top_k}"
          f"   <-- the SAME values must be on `vllm serve --override-generation-config`")
    print(f"budgets   token_budget={args.token_budget} max_completion={args.max_completion_length} "
          f"heartbeat={args.heartbeat_stale_after_s:g}s agent_steps={args.agent_step_limit} dtype={args.dtype}")
    print(f"aux       agent_turn_fn={args.agent_turn_filter}  (Harbor drops AUXILIARY-role turns "
          f"server-side; check rollout/fork_frac at STEP 1)")
    print(f"output    {out_dir}")

    worker = HarnessRolloutWorker(
        harness_session_factory=factory,
        harness_adapter=None,  # loop-owning: the agent drives itself; we read what it did
        # Reinforce turns that took an ACTION rather than prose. Works only because the env hands TRL
        # tool calls in the NESTED OpenAI shape; flattened, this is False for every turn and the whole
        # rollout is discarded with no error anywhere.
        train_turn_fn=has_tool_call,
        agent_turn_fn=tool_calling_turns if args.agent_turn_filter == "tools" else None,
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],  # the environment's verify() IS the reward; None means UNSCORED, never 0.0
        processing_class=tokenizer,
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        log_completions=True,
        num_completions_to_print=2,
    )

    config = AsyncGRPOConfig(
        output_dir=out_dir,
        save_strategy="steps" if args.save_steps else "no",
        save_steps=args.save_steps or 500,
        save_total_limit=None,  # never rob an eval watcher of a checkpoint
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_staleness=args.max_staleness,
        vllm_server_base_url=args.vllm_url,
        optim=args.optim,
        bf16=True,
        dtype=args.dtype,
        trust_remote_code=True,  # Qwen3_5ForConditionalGeneration is a custom arch
        token_budget=args.token_budget,
        heartbeat_stale_after_s=args.heartbeat_stale_after_s,
        gradient_checkpointing=True,
        # Required: the reentrant checkpointer does not see inputs arriving through anything but
        # positional args, and the hybrid-attention path passes state that way.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="trackio",
        project=args.project,
        run_name=run_name,
        log_completions=True,
        logging_steps=1,  # every rollout costs a sandbox and minutes; nothing is logged in arrears
        seed=args.seed,
    )

    AsyncGRPOTrainer(
        model=args.model, args=config, train_dataset=dataset, rollout_worker=worker
    ).train()


if __name__ == "__main__":
    main()
