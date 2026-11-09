"""Train a retrosynthesis agent on RetroEnv with AsyncGRPO. Read the numbered sections in order.

The experiment is sync_grpo.py's: same model, tasks, environment, adapter and reward. Only the
trainer differs. AsyncGRPO runs rollouts in a separate process that keeps generating while the
trainer steps, so a slow episode (sixteen tool turns against the server) no longer stalls the
optimizer. Episodes may be up to `max_staleness` policy versions old when they are trained on.

    python -m train.async_grpo --server http://127.0.0.1:8000 --vllm-url http://127.0.0.1:8001

train/jobs/ starts the RetroEnv server and vLLM and runs this on HF Jobs (see train/README.md).
"""

# %% 1-5. Model, tokenizer, tasks, environment, reward and adapter: shared with sync_grpo.py.
import json
import os
from pathlib import Path

from train.sync_grpo import (
    MODEL_REVISIONS,
    arguments,
    environment_factory,
    lora_config,
    patch_turn_requests,
    task_dataset,
    tokenizer_for,
)


# %% 6. Configure AsyncGRPO. Batch, lengths, sampling and clipping match sync_grpo.py.
#
# The rollout process scores each group itself, so the solved/submitted columns of sync_grpo.py
# are not logged here; episodes.jsonl records both for every episode. AsyncGRPO trains on vLLM's
# own log-probabilities as the behaviour policy and clips per token, so it needs no separate
# importance-sampling correction.
def main():
    args = arguments(__doc__, output="runs/rl/async_grpo")
    if not args.smoke and args.steps % args.save_steps:
        raise SystemExit(f"--steps {args.steps} must be a multiple of --save-steps {args.save_steps}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(output.resolve() / "trackio")
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer

    dataset = task_dataset(args.server, args.max_depth, tasks=args.train_tasks, pool=args.task_pool)
    tokenizer = tokenizer_for(args.model, thinking=args.thinking)
    if args.thinking:
        # Thinking keeps earlier turns' thinking in the history (see sync_grpo.py), so a whole
        # episode is one row and must fit the context. The template's default reasoning effort
        # (xhigh) asks for long deliberation: the first turn of the thinking eval ran ~8.5k tokens.
        # Medium (the default here) adds no instruction; low asks for brevity, and a turn that
        # thinks past the 16k cap ends its episode unsubmitted. Sampling is untruncated (no top-p/top-k): on vLLM 0.25
        # TRL cannot correct the importance ratios for truncated sampling.
        sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}
        per_turn, budget = args.turn_tokens, args.context
        template = {"enable_thinking": True, "reasoning_effort": args.reasoning_effort}
    else:
        sampling = {"temperature": 0.8, "top_p": 1.0, "top_k": -1}  # as the difficulty sweep sampled
        per_turn, budget = args.turn_tokens, min(40960, args.context)
        template = {"enable_thinking": False, "preserve_thinking": True}
    if args.train_tasks or args.task_pool:
        # A pool run is sized like the overfitting run that climbed: a round is --round-tasks
        # groups (4 by default) of --generations episodes, split into --updates-per-round updates.
        # An overfitting run trains on one round per update: a group of every task, one sample per
        # row per GPU (no packing). Forked episodes add a sample now and then, so a step can end a
        # few episodes before or after its round.
        # `--updates-per-round` splits a round into that many optimizer steps.
        # A group's version is the policy that started it, and a group runs for minutes while
        # generation never stops, so a round mixes a few versions: at most one stale dropped 112 of
        # a step's samples. Each extra update per round ages the queue by one more version.
        tasks = len(json.loads(Path(args.train_tasks).read_text())) if args.train_tasks else args.round_tasks
        rounds = tasks * (4 if args.smoke else args.generations)
        trainers = int(os.environ.get("WORLD_SIZE", "1"))  # set by accelerate: one per training GPU
        # A thinking episode trains as ~2.2 rows: a later turn that re-renders differently from what
        # was generated forks a new row (samples_per_rollout 1.6-2.7, think3 x 32). Sized for one
        # row per episode, three updates took 96 of a round's ~210 rows; the backlog aged to the
        # staleness limit and was dropped (356 rows at once), and each update saw ~14 episodes.
        rows = args.rows_per_episode or (2.2 if args.thinking else 1.0)
        accumulation = max(1, round(rounds * rows / (trainers * args.updates_per_round)))
        # A limit at 2 + updates per round made the trainer drop a stale backlog and then sit idle
        # waiting for fresh episodes (110 s idle against 38 s of compute); stale samples were barely
        # clipped (~0.5% of tokens), so they stay useful and the limit is twice as loose.
        budget, staleness = 0, 2 + 2 * args.updates_per_round
    else:
        # Rows are packed to `token_budget`: 4 rows per GPU held about 22 eight-attempt episodes on
        # two GPUs; 8 hold about three sixteen-attempt groups.
        accumulation = 4 if args.smoke or args.generations <= 8 else 8
        staleness = 2 if args.thinking else 4  # long thinking episodes: keep samples near on-policy
    # vLLM's --max-model-len (serve_model.py gets the same --context); the rollout process inherits it.
    os.environ["RETROENV_MAX_MODEL_LEN"] = str(args.context)
    patch_turn_requests()
    config = AsyncGRPOConfig(
        output_dir=str(output),
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_steps=2 if args.smoke else args.steps,
        per_device_train_batch_size=1 if args.smoke or args.train_tasks or args.task_pool else 4,
        gradient_accumulation_steps=accumulation,
        num_generations=4 if args.smoke else args.generations,
        max_completion_length=per_turn,  # per model turn: AsyncGRPO caps each generation call
        max_tool_calling_iterations=20,  # 16 in the prompt, 4 to spare for the submission
        **sampling,
        epsilon_high=0.28,
        chat_template_kwargs=template,
        # Episodes in flight against vLLM and the server, and how stale a trained episode may be.
        max_inflight_tasks=4 if args.smoke else args.inflight,  # run.py sets 32 per vLLM replica
        max_staleness=staleness,
        token_budget=budget,  # tokens packed into one forward pass (0: one sample per row); longer is dropped
        # A turn whose re-rendered history drifts from the generated tokens by fewer than this
        # many tokens is realigned (those tokens become context); a larger drift starts a new row.
        # At 0 every few-token wobble forked a row repeating the whole prefix: 1.2-1.8 rows per
        # episode, more trainer time, and episodes counted several times in a step's reward.
        fork_threshold_tokens=1024,
        optim="adamw_torch_fused",  # LoRA state is small; bitsandbytes 8-bit Adam rejects FSDP2 tensors
        bf16=True,
        dtype="bfloat16",  # vLLM serves bf16 too; the default float32 would double the trainer's memory
        model_init_kwargs={"revision": MODEL_REVISIONS[args.model]},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        vllm_server_base_url=args.vllm_url,
        vllm_server_timeout=1800,
        # One turn's request. TRL's 600 s cut off thinking turns (up to 16k tokens with 128
        # episodes sharing four replicas), and a timed-out turn is generated again from scratch.
        request_timeout=3600,
        # A thinking group of 32 waits on its slowest episode (25+ minutes), and the first batch on
        # its first group: at the default 30 minutes the other ranks' broadcast timed out waiting
        # for rank 0 (gloo recv) and the run died before its first update.
        heartbeat_stale_after_s=3 * 3600,
        ddp_timeout=3 * 3600,
        save_strategy="steps",
        save_steps=1 if args.smoke else args.save_steps,
        save_total_limit=None,
        logging_steps=1,
        log_completions=args.thinking,  # thinking traces in trackio, to read what the model thinks
        report_to="trackio",
        project="retroenv-rl",
        run_name=output.parent.name if output.name == "train" else output.name,
        trackio_space_id=args.space_id,
        seed=0,
    )
    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset,
        environment_factory=environment_factory(args.server, args.partial_credit, args.reward, args.mask_truncated),
        peft_config=lora_config(),
    )
    if args.init_adapter:
        # Start from a trained adapter (same LoRA config): its weights replace the fresh ones before
        # training wraps the model, and vLLM gets them at the first weight sync. With --lr 0 this
        # replays a checkpoint in the training harness.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load

        # Read whole, then parse: load_file memory-maps the file, and a page fault on the bucket's
        # FUSE mount killed a rank with SIGBUS.
        state = load((args.init_adapter / "adapter_model.safetensors").read_bytes())
        loaded = set_peft_model_state_dict(trainer.model, state)
        missing = [key for key in loaded.missing_keys if "lora_" in key]
        if loaded.unexpected_keys or missing:
            raise RuntimeError(
                f"{args.init_adapter} does not fit this adapter: {loaded.unexpected_keys[:3]} {missing[:3]}"
            )
        print(f"Started from {args.init_adapter}: {len(state)} adapter tensors", flush=True)

    # %% 7. Train. The last checkpoint is the final adapter.
    #
    # max_steps is a multiple of save_steps, so checkpoint-<max_steps> is written inside the
    # loop. A save_model() after train() is a collective under FSDP2, and in the h200x4 smoke
    # rank 0 waited in it for half an hour with rank 1 already gone.
    (output / "training_config.json").write_text(json.dumps(config.to_dict(), indent=2, default=str))
    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    main()
