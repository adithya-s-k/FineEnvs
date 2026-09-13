"""Async GRPO on the data-agent task set via OpenEnv x Harbor, SEVERAL harnesses in one run.

Identical to train_harbor_opencode.py in every training knob; the only change is that each GRPO
GROUP is routed to a harness (see multi_harness.py). Harness is constant WITHIN a group and varies
BETWEEN groups, which keeps the advantage encoding "which action" rather than "which harness" --
measured pass@4 across harnesses on this suite spans 0.320 to 0.020.

ADMISSION IS NOT OPTIONAL. Run tools/tito_matrix.py first. Measured 2026-09-13, Qwen3.5-4B:

    opencode         extends 4/4, roots 1  -> TITO PASS, admit
    mini-swe-agent   extends 4/4, roots 1  -> TITO PASS, admit (an earlier run showed an
                                             intermittent aux call; confirm at N~25)
    codex            extends 2/4           -> FAIL. The Responses transformer splits one model turn
                                             into TWO assistant messages (prose, then tool_call), so
                                             the next prompt renders a turn boundary the model never
                                             emitted, ~40 tok/turn. Fixable upstream; do not train
                                             on it until fixed.
    claude-code      extends 0/4, roots=N  -> FAIL structurally (per_turn_capture_only: it re-renders
                                             its whole prompt every turn). Eval only.

A harness that fails T8 still has exact tokens and logprobs. What it loses is CROSS-TURN credit
assignment, because one rollout stops being one training sample.

TOKEN-IN-TOKEN-OUT. `to_trace_entries` carries the engine's own `prompt_token_ids`, and TRL's
`_turns_from_trace` reads them and RAISES if absent -- a local re-render matched the engine on 0 of
28 measured turns. Verify from metrics, not from this file: rollout/fork_frac == 0,
rollout/samples_per_rollout == 1.00, rollout/drift_tokens_max == 0, all AT STEP 1.

THE TRAPDOOR. Jobs 72452/72473 wedged at step 7 and 10 of 100, spending 4,076 E2B sandboxes on 11
productive groups, because an UNGATED efficiency term made zero tool calls the highest-scoring move
while `train_turn_fn=has_tool_call` then yielded no trainable turns. The `_train` suite emits a
single float with no efficiency term (verified across all 2,238 graders), so the first leg is absent
here -- but the second is not. A model too weak for a task still produces empty groups. Band the
task indices so reward_std > 0, and watch it from step 1.
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
    # '+'-separated, never commas: `sbatch --export=ALL,VAR=a,b,c` truncates at the first comma
    # SILENTLY, and the job then runs a harness set it was never given.
    p.add_argument("--harnesses", default="opencode+mini-swe-agent")
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
    p.add_argument("--project", default="data-agent-harbor-multi")
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
    from multi_harness import MultiHarborSessionFactory, pair_rows

    harnesses = [h.strip() for h in args.harnesses.replace(",", "+").split("+") if h.strip()]
    factory = MultiHarborSessionFactory(
        args.server,
        harnesses=harnesses,
        split=args.split,
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
    # pair_rows pads so gcd(len(rows), n_harnesses) == 1. Without it, group->row and group->harness
    # stay in lockstep and each task meets only ONE harness: at 40 tasks and 2 harnesses, 0 of 40
    # tasks meet both. The run looks multi-harness and is a disjoint partition.
    rows = pair_rows(factory)
    dataset = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    run_name = args.run_name or (
        f"{args.model.split('/')[-1]}-multi{len(harnesses)}-harbor-{args.max_steps}steps"
        # Stamped with the job id: trackio keys a run by name inside a project, so relaunches
        # otherwise stack on top of each other.
        f"-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    out_dir = args.output_dir or f"/fsx/{os.environ.get('USER','x')}/runs/agrpo_harbor/{run_name}"

    print(f"model     {args.model}")
    print(f"server    {args.server}   vllm {args.vllm_url}")
    print(f"rollouts  {'+'.join(harnesses)} on {args.sandbox}, {args.num_generations}x{args.max_inflight}")
    print(f"routing   harness = harnesses[group_id % {len(harnesses)}] (constant WITHIN a group)")
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
