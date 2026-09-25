"""Async GRPO on the data-agent task set via OpenEnv x Harbor, SEVERAL harnesses in one run.

Identical to train_harbor_opencode.py in every training knob; the only change is that each GRPO
GROUP is routed to a harness (see multi_harness.py). Harness is constant WITHIN a group and varies
BETWEEN groups, which keeps the advantage encoding "which action" rather than "which harness" --
measured pass@4 across harnesses on this suite spans 0.320 to 0.020.

ADMISSION. Run the 10-harness smoke and inspect exact engine token ids, sampled logprobs,
per-token masks, retained supervision, and sampling policy. Prefix drift increases packed context;
it does not invalidate per-call TITO or rollout-level rewards. The loop-owning worker uses lossless
reconciliation: exact prefixes merge, every rewritten history starts a new row.

The default trains all retained agent turns. Some harnesses (for example Terminus) express actions
as text, so a universal `has_tool_call` filter would silently remove their entire training signal.
`--train-turn-filter tool_calls` is an explicit native-tool-call-only ablation. Auxiliary calls are
already removed by Harbor's capture/ATIF reconciliation.

THE TRAPDOOR. Jobs 72452/72473 wedged at step 7 and 10 of 100, spending 4,076 E2B sandboxes on 11
productive groups, because an UNGATED efficiency term made zero tool calls the highest-scoring move
while `train_turn_fn=has_tool_call` then yielded no trainable turns. The `_train` suite emits a
single float with no efficiency term (verified across all 2,238 graders), so the first leg is absent
here. The default all-agent-turn filter also avoids dropping text-action rollouts. Constant-reward
groups can still have no advantage signal: band the task indices and watch reward_std from step 1.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker, has_tool_call

TRAIN_SPLIT = "AdithyaSK/data_agent_rl_environment_train"


def tool_calling_turns(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Optional filter for turns whose request offered native tools.

    Harbor already removes auxiliary calls. This additional restriction is an ablation and must
    not be used for text-action harnesses; prefix drift alone is not evidence of an auxiliary call.
    """
    return [e for e in trace if ((e.get("metadata") or {}).get("n_tools") or 0) > 0]


def build(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="http://127.0.0.1:8200", help="a running `openenv harbor serve`")
    p.add_argument("--vllm-url", required=True, help="the engine AsyncGRPO also syncs weights into")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--model-revision", default=None)
    p.add_argument("--resume-from-checkpoint", default="", help="Completed local checkpoint including optimizer and rollout cursor")
    p.add_argument("--split", default=TRAIN_SPLIT)
    # '+'-separated, never commas: `sbatch --export=ALL,VAR=a,b,c` truncates at the first comma
    # SILENTLY, and the job then runs a harness set it was never given.
    p.add_argument("--harnesses", default="opencode+mini-swe-agent")
    p.add_argument("--sandbox", default="e2b")
    p.add_argument("--reward-key", default="", help="'' lets the server pick; required on a multi-reward suite")
    p.add_argument("--task-indices", default="", help="comma-separated, or @file")
    p.add_argument("--n-tasks", type=int, default=0, help="0 = the whole split")
    p.add_argument("--all-task-harness-pairs", action="store_true",
                   help="Schedule every task under every harness before repeating the dataset")
    p.add_argument("--harness-schedule", default="", help="Frozen one-harness-per-task rotation JSON")
    p.add_argument("--agent-turn-filter", default="none", choices=["none", "tools"],
                   help="Optional tool-manifest filter; incompatible with harnesses that express actions as text")
    p.add_argument("--train-turn-filter", default="all", choices=["all", "tool_calls"],
                   help="Train all selected agent turns, or explicitly restrict to native tool-call turns")

    # ---- the reference's values, unchanged ---------------------------------------------------
    p.add_argument("--learning-rate", type=float, default=3e-6)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--atomic-rollouts", action="store_true",
                   help="Keep all rows of each admitted rollout in one update (single dense trainer GPU)")
    p.add_argument("--max-outstanding-rollouts", type=int, default=0,
                   help="Atomic recipe: bound generating plus queued rollouts until optimizer consumption")
    p.add_argument("--max-row-tokens", type=int, default=131072,
                   help="Hard context limit for atomic rollout forwards; token-budget is the packing target")
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--max-train-seconds", type=float, default=0,
                   help="If positive, save and stop at the first update boundary after this duration")
    p.add_argument("--coverage-min-steps", type=int, default=0,
                   help="If positive, stop once this many updates and every task/harness pair are covered; max-steps remains a hard ceiling")
    p.add_argument("--audit-dir", default="", help="Save per-rollout capture results and pair coverage locally")
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
    p.add_argument("--checkpoint-max-seconds", type=float, default=0,
                   help="Also save at the first optimizer boundary after this interval; 0 disables")
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


def main(argv=None, *, session_factory_class=None, agent_turn_selector=None) -> None:
    args = build(argv)
    from multi_harness import MultiHarborSessionFactory, pair_rows
    resume = None
    if args.resume_from_checkpoint:
        from checkpoint_artifacts import resume_info
        resume = resume_info(args.resume_from_checkpoint, args.model, args.model_revision)
    group_offset = resume['group_offset'] if resume else 0

    harnesses = [h.strip() for h in args.harnesses.replace(",", "+").split("+") if h.strip()]
    if args.harness_schedule and args.all_task_harness_pairs:
        raise ValueError('Choose either a rotating schedule or Cartesian scheduling')
    schedule = None
    if args.harness_schedule:
        with open(args.harness_schedule) as stream:
            schedule = json.load(stream)
    factory_class = session_factory_class or MultiHarborSessionFactory
    factory = factory_class(
        args.server,
        harnesses=harnesses,
        schedule=schedule,
        group_offset=group_offset,
        split=args.split,
        sandbox=args.sandbox,
        # THE SAME engine the trainer syncs weights into. That is what makes the rollouts on-policy:
        # the agent's calls and the weight updates go to one vLLM. It must be the node's ROUTABLE
        # address -- the harbor server probes it from ANOTHER host, and with localhost the probe
        # fails, the tier grades `text`, and every rollout comes back with no trainable turns.
        llm_url=os.environ.get("ROLLOUT_LLM_URL", args.vllm_url),
        api_key=os.environ.get("ROLLOUT_LLM_API_KEY", ""),
        model=args.model,
        sampling={"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
        reward_key=args.reward_key,
        agent_timeout_sec=args.agent_timeout,
        agent_step_limit=args.agent_step_limit,
        indices=indices_of(args.task_indices),
        num_tasks=args.n_tasks or None,
    )
    if args.coverage_min_steps and not 0 < args.coverage_min_steps <= args.max_steps:
        raise ValueError("coverage-min-steps must be between 1 and max-steps")
    if args.audit_dir:
        from training_audit import AuditedFactory
        factory = AuditedFactory(factory, args.audit_dir)

    # Built FROM the factory so the instruction TRL sends is one the server can resolve: `create()`
    # hashes the prompt back to a task index and RAISES on a miss rather than silently running task 0.
    # pair_rows pads so gcd(len(rows), n_harnesses) == 1. Without it, group->row and group->harness
    # stay in lockstep and each task meets only ONE harness: at 40 tasks and 2 harnesses, 0 of 40
    # tasks meet both. The run looks multi-harness and is a disjoint partition.
    rows = pair_rows(factory, all_pairs=args.all_task_harness_pairs)
    dataset = Dataset.from_list(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, trust_remote_code=True)

    implementation = "standalone-opencode" if session_factory_class else "harbor"
    run_name = args.run_name or (
        f"{args.model.split('/')[-1]}-multi{len(harnesses)}-{implementation}-{args.max_steps}steps"
        # Stamped with the job id: trackio keys a run by name inside a project, so relaunches
        # otherwise stack on top of each other.
        f"-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    out_dir = args.output_dir or f"/fsx/{os.environ.get('USER','x')}/runs/agrpo_harbor/{run_name}"
    if resume:
        from training_audit import write_json
        write_json(os.path.join(args.audit_dir or out_dir, 'resume.json'), resume)
        print(f"resume    checkpoint step={resume['step']}, next schedule group={group_offset}", flush=True)

    print(f"model     {args.model}")
    print(f"server    {args.server}   vllm {args.vllm_url}")
    print(f"rollouts  {'+'.join(harnesses)} on {args.sandbox}, {args.num_generations}x{args.max_inflight}")
    print(f"routing   {'frozen rotation' if schedule else 'modulo harness routing'}; constant within each group")
    print(f"tasks     {len(dataset)} from {args.split}")
    print(f"sampling  temperature={args.temperature} top_p={args.top_p} top_k={args.top_k}"
          f"   (explicit capture session policy; checked against trainer recompute)")
    print(f"budgets   token_budget={args.token_budget} max_completion={args.max_completion_length} "
          f"heartbeat={args.heartbeat_stale_after_s:g}s agent_steps={args.agent_step_limit} dtype={args.dtype}")
    print(f"admission atomic={args.atomic_rollouts} max_outstanding_rollouts={args.max_outstanding_rollouts}")
    print(f"aux       agent_turn_fn={agent_turn_selector.__name__ if agent_turn_selector else args.agent_turn_filter}; "
          f"train_turn_filter={args.train_turn_filter}; implementation={implementation}")
    print(f"output    {out_dir}")

    worker_class, trainer_class = HarnessRolloutWorker, AsyncGRPOTrainer
    if args.atomic_rollouts:
        from atomic_rollouts import AtomicHarnessWorker, AtomicRolloutTrainer
        worker_class, trainer_class = AtomicHarnessWorker, AtomicRolloutTrainer
    worker = worker_class(
        **({"max_outstanding_rollouts": args.max_outstanding_rollouts} if args.atomic_rollouts else {}),
        harness_session_factory=factory,
        harness_adapter=None,  # loop-owning: the agent drives itself; we read what it did
        # Text-action harnesses have no native tool_calls. Keep their supervision by default.
        train_turn_fn=has_tool_call if args.train_turn_filter == "tool_calls" else None,
        lossless_capture=True,
        fork_threshold_tokens=0,
        agent_turn_fn=agent_turn_selector or (tool_calling_turns if args.agent_turn_filter == "tools" else None),
        model_name=args.model,
        dataset=dataset,
        reward_funcs=[],  # the environment's verify() IS the reward; None means UNSCORED, never 0.0
        processing_class=tokenizer,
        num_generations=args.num_generations,
        max_inflight_tasks=args.max_inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
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
        max_inflight_tasks=args.max_inflight,
        fork_threshold_tokens=0,
        vllm_server_base_url=args.vllm_url,
        optim=args.optim,
        bf16=True,
        dtype=args.dtype,
        trust_remote_code=True,  # Qwen3_5ForConditionalGeneration is a custom arch
        model_init_kwargs={"revision": args.model_revision} if args.model_revision else None,
        token_budget=args.token_budget,
        heartbeat_stale_after_s=args.heartbeat_stale_after_s,
        gradient_checkpointing=True,
        # Required: the reentrant checkpointer does not see inputs arriving through anything but
        # positional args, and the hybrid-attention path passes state that way.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="trackio",
        project=args.project,
        run_name=run_name,
        trackio_space_id=None,
        trackio_bucket_id=None,
        trackio_static_space_id=False,  # CPU logger owns online sync; never publish/freeze from trainer
        log_completions=True,
        logging_steps=1,  # every rollout costs a sandbox and minutes; nothing is logged in arrears
        seed=args.seed,
    )

    trainer_kwargs = ({"max_row_tokens": args.max_row_tokens, "admission_dir": args.audit_dir}
                      if args.atomic_rollouts else {})
    trainer = trainer_class(
        model=args.model, args=config, train_dataset=dataset, rollout_worker=worker,
        **trainer_kwargs,
    )
    from training_audit import CheckpointReadyCallback
    trainer.add_callback(CheckpointReadyCallback(args.model, args.model_revision))
    if args.checkpoint_max_seconds > 0:
        from training_audit import PeriodicCheckpointCallback
        trainer.add_callback(PeriodicCheckpointCallback(args.checkpoint_max_seconds))
    if args.max_train_seconds:
        from training_audit import WallTimeCallback
        trainer.add_callback(WallTimeCallback(args.max_train_seconds))
    if args.audit_dir or args.coverage_min_steps:
        from training_audit import PairCoverageCallback
        trainer.add_callback(PairCoverageCallback(
            trainer, len(rows), harnesses, args.coverage_min_steps, args.audit_dir or out_dir,
            all_pairs=args.all_task_harness_pairs,
            schedule=schedule,
            group_offset=group_offset,
        ))
    trainer.train(resume_from_checkpoint=resume['checkpoint'] if resume else None)
    trainer.save_state()
    trainer.save_model(os.path.join(out_dir, "final"))


if __name__ == "__main__":
    main()
