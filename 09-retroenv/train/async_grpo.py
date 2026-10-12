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

from train.sync_grpo import MODEL_REVISIONS, arguments, environment_factory, lora_config, task_dataset, tokenizer_for


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

    dataset = task_dataset(args.server, args.max_depth)
    tokenizer = tokenizer_for(args.model)
    config = AsyncGRPOConfig(
        output_dir=str(output),
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_steps=2 if args.smoke else args.steps,
        # One optimizer step sees 16 episodes, as in sync_grpo.py: two tasks, eight attempts each.
        per_device_train_batch_size=1 if args.smoke else 4,
        gradient_accumulation_steps=4 if args.smoke else 4,
        num_generations=4 if args.smoke else 8,
        max_completion_length=16384,
        max_tool_calling_iterations=20,  # 16 in the prompt, 4 to spare for the submission
        temperature=0.8,
        top_p=1.0,
        top_k=-1,
        epsilon_high=0.28,
        chat_template_kwargs={"enable_thinking": False, "preserve_thinking": True},
        # Episodes in flight against vLLM and the server, and how stale a trained episode may be.
        max_inflight_tasks=4 if args.smoke else args.inflight,  # run.py sets 32 per vLLM replica
        max_staleness=4,
        token_budget=40960,  # tokens packed into one forward pass
        fork_threshold_tokens=0,
        optim="adamw_torch_fused",  # LoRA state is small; bitsandbytes 8-bit Adam rejects FSDP2 tensors
        bf16=True,
        dtype="bfloat16",  # vLLM serves bf16 too; the default float32 would double the trainer's memory
        model_init_kwargs={"revision": MODEL_REVISIONS[args.model]},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        vllm_server_base_url=args.vllm_url,
        vllm_server_timeout=1800,
        heartbeat_stale_after_s=1800,
        save_strategy="steps",
        save_steps=1 if args.smoke else args.save_steps,
        save_total_limit=None,
        logging_steps=1,
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
        environment_factory=environment_factory(args.server, args.partial_credit),  # rebuilt in the rollout process
        peft_config=lora_config(),
    )

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
