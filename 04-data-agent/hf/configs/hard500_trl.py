"""Copyable settings for the checkpoint-500 hard-task continuation.

Use with the finite atomic Harbor worker/trainer. Task epochs are bounded by
1,000 scheduled groups; native Trainer num_train_epochs is not that boundary.
"""

MODEL = {
    "name": "Qwen/Qwen3.5-2B",
    "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "parent_step": 500,
    "restore_optimizer": True,
    "restore_scheduler": True,
    "new_curriculum_cursor": 0,
}

TRL_CONFIG = {
    "output_dir": "./outputs/hard500/run",
    "learning_rate": 3e-6,
    "lr_scheduler_type": "constant",
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "optim": "paged_adamw_8bit",
    "epsilon": 0.2,
    "epsilon_high": 0.2,
    "bf16": True,
    "dtype": "bfloat16",
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_generations": 8,
    "max_inflight_tasks": 32,
    "max_staleness": 4,
    "token_budget": 40960,
    "max_completion_length": 4096,
    "fork_threshold_tokens": 0,
    "temperature": 0.8,
    "top_p": 1.0,
    "top_k": 0,
    "vllm_server_base_url": "http://127.0.0.1:8000",
    "request_timeout": 600,
    "weight_sync_timeout": 1800,
    "heartbeat_stale_after_s": 900,
    "max_steps": 2501,  # Safety ceiling including parent step500; the finite worker stops earlier.
    "ignore_data_skip": True,
    "save_strategy": "steps",
    "save_steps": 50,
    "save_total_limit": None,
    "logging_steps": 1,
    "log_completions": True,
    "report_to": "trackio",
    "project": "multi4-hard500-2epochs-from500-20260917",
    "run_name": "multi4-hard500-2epochs-from500-20260917",
    "trackio_space_id": None,  # Separate logger publishes to the comparison Space.
    "trackio_bucket_id": None,
    "trackio_static_space_id": False,
    "trust_remote_code": True,
    "model_init_kwargs": {"revision": MODEL["revision"]},
    "seed": 1,
}

CURRICULUM = {
    "tasks": 500,
    "difficulty": "hard",
    "task_epochs": 2,
    "total_groups": 1000,
    "requested_rollouts": 8000,
    "selection_seed": 1,
    "one_harness_per_task_per_epoch": True,
    "rotate_harness_between_epochs": True,
    "harnesses": ["opencode", "claude-code", "codex", "mini-swe-agent"],
    "harness_versions": {"opencode": "1.18.31", "claude-code": "2.1.270", "codex": "0.154.0", "mini-swe-agent": "2.4.6"},
}

HARNESS_CONFIG = {
    "server": "http://127.0.0.1:8200",
    "sandbox": "e2b",
    "sandbox_cpu": 1,
    "sandbox_memory_mb": 4096,
    "reward_key": "correctness,reward",
    "agent_step_limit": 17,
    "agent_timeout_seconds": 600,
    "atomic_rollouts": True,
    "max_outstanding_rollouts": 16,
    "max_rollouts_per_packing_unit": 2,
    "max_row_tokens": 131072,
    "loss_normalization": "update_supervised_token_mean",
    "lossless_capture": True,
    "agent_turn_filter": "none",
    "train_turn_filter": "all",
}

SERVING = {
    "tp": 1,
    "dp": 1,
    "max_model_len": 131072,
    "max_output_tokens_per_response": 4096,
    "enable_thinking": False,
    "tool_call_parser": "qwen3_xml",
    "reasoning_parser": "qwen3",
    "gdn_prefill_backend": "triton",
    "logprobs_mode": "processed_logprobs",
    "return_tokens_as_token_ids": True,
    "prefix_caching": False,
    "training_weight_transfer_backend": "nccl",
    "training_gpu_memory_utilization": 0.85,
    "eval_gpu_memory_utilization": 0.90,
}

EVALUATION = {
    "interval_optimizer_steps": 100,
    "first_global_step": 600,
    "include_final_checkpoint": True,
    "test_tasks": 250,
    "difficulty_counts": {"easy": 33, "medium": 118, "hard": 99},
    "harnesses": CURRICULUM["harnesses"],
    "pass_k": 1,
    "evaluations_per_checkpoint": 1000,
    "concurrency": 50,
    "max_active_eval_jobs": 1,
    "queue_all_due_checkpoints": True,
    "max_retries": 3,
    "retry_policy": "ungraded failures only; preserve the first grade, including zero",
}

HF_JOBS = {
    "namespace": "HuggingEnvs",
    "training_flavor": "h200x2",
    "training_timeout": "72h",
    "training_gpu": 1,
    "rollout_inference_gpu": 0,
    "eval_flavor": "a100-large",
    "eval_timeout": "4h",
    "coordinator_flavor": "cpu-upgrade",
    "coordinator_timeout": "96h",
    "checkpoint_storage": "HF Bucket; full checkpoint manifest uploaded last",
    "trackio_space": "HuggingEnvs/data-agent-training-comparison-trackio",
    "trackio_comparison_project": "qwen35-2b-harbor-vs-opencode-20260916",
    "offline_logging": True,
    "online_logging": True,
}


def make_trl_config(**overrides):
    from trl.experimental.async_grpo import AsyncGRPOConfig
    return AsyncGRPOConfig(**{**TRL_CONFIG, **overrides})
