"""Train a small model to write olympiad proofs through our Harbor harness, with AsyncGRPO.

Adapted from 05-multi-harness-rl/train/multi_harness.py. What changes: one harness, ours (the FineProof harness: write a
proof, check it, rewrite it), and the reward is the rubric judge's score / 7 from the task's verifier, unchanged.
Read the numbered sections in order.
"""

# %% 1. Choose the model and run settings.
import argparse
import json
import os
from functools import partial
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from tasks import TaskFactory, dataset_dir, load_tasks

# The pinned revision 05-multi-harness-rl validated. LFM2.5 needs 05's packed-convolution fix, not ported here.
MODEL_REVISIONS = {
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--data", default="prepared")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--env-llm-url", help="Inference URL reachable by a remote environment server")
    parser.add_argument("--server", default="http://127.0.0.1:8200")
    parser.add_argument("--trials", default="runs/trials")
    parser.add_argument("--output", default="runs/train")
    parser.add_argument("--smoke", action="store_true", help="Two updates, two rollouts per group, a small batch")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--inflight", type=int, default=32)
    parser.add_argument("--space-id", default=None, help="Optional Trackio Space; local logs are always kept")
    parser.add_argument("--sandbox", default="daytona", choices=["daytona", "hf-sandbox"],
                        help="Where Harbor runs each task's sandboxes (hf-sandbox: HF Jobs, needs a docker_image task)")
    return parser.parse_args()


# %% 2. Load the tokenizer in non-thinking mode (the same template vLLM serves, see jobs/serve_model.py).
def tokenizer_for(model):
    from copy import deepcopy

    from trl.chat_template_utils import qwen3_5_template

    tokenizer = AutoTokenizer.from_pretrained(model, revision=MODEL_REVISIONS[model])
    tokenizer.chat_template = "{%- set enable_thinking = false -%}\n" + tokenizer.chat_template
    tokenizer.response_template = deepcopy(qwen3_5_template)
    return tokenizer


# %% 3. The reward is the verifier's: rubric score / 7. A missing grade (judge unreachable, sandbox failure) is
# excluded from the group rather than counted as a wrong proof.
def reward(outcome):
    return None if outcome.env_reward is None else float(outcome.env_reward)


# %% 4. Configure AsyncGRPO. These are the experiment's hyperparameters.
def main():
    args = arguments()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(output.resolve() / "trackio")
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
    from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker

    tasks = load_tasks(dataset_dir(args.data, "train"))
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": task["instruction"]}]} for task in tasks]).shuffle(seed=0)
    tokenizer = tokenizer_for(args.model)
    generations = 2 if args.smoke else 8
    inflight = min(args.inflight, 4) if args.smoke else args.inflight
    config = AsyncGRPOConfig(
        output_dir=str(output),
        learning_rate=2e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_steps=2 if args.smoke else args.steps,
        per_device_train_batch_size=1 if args.smoke else 4,
        gradient_accumulation_steps=2 if args.smoke else 4,
        num_generations=generations,
        max_completion_length=16384,
        temperature=0.8,
        top_p=1.0,
        top_k=-1,
        max_inflight_tasks=inflight,
        max_staleness=4,
        token_budget=40960,
        fork_threshold_tokens=0,
        optim="paged_adamw_8bit",
        bf16=True,
        dtype="bfloat16",
        model_init_kwargs={"revision": MODEL_REVISIONS[args.model]},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        vllm_server_base_url=args.vllm_url,
        heartbeat_stale_after_s=1800,
        save_strategy="steps",
        save_steps=1 if args.smoke else args.save_steps,
        save_total_limit=None,
        logging_steps=1,
        report_to="trackio",
        project="fineproof-rl",
        run_name=output.parent.name if output.name == "train" else output.name,
        trackio_space_id=args.space_id,
        seed=0,
    )
    worker = HarnessRolloutWorker(
        harness_session_factory=partial(TaskFactory, args, tasks, split="train"),
        harness_adapter=None,  # our Harbor agent owns the conversation; OpenEnv captures its model calls
        rollout_reward_fn=reward,
        lossless_capture=True,
        model_name=args.model,
        processing_class=tokenizer,
        dataset=dataset,
        reward_funcs=[],
        num_generations=generations,
        max_inflight_tasks=inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=16384,
        temperature=0.8,
        top_p=1.0,
        top_k=-1,
        chat_template_kwargs={"enable_thinking": False},
    )
    trainer = AsyncGRPOTrainer(
        model=args.model, args=config, processing_class=tokenizer, train_dataset=dataset, rollout_worker=worker)

    # %% 5. Train, then save weights and tokenizer for evaluation.
    (output / "training_config.json").write_text(json.dumps(config.to_dict(), indent=2))
    (output / "task_names.json").write_text(json.dumps([t["name"] for t in tasks]))
    trainer.train()
    trainer.save_model(str(output / "final"))
    tokenizer.save_pretrained(output / "final")
    trainer.save_state()


if __name__ == "__main__":
    main()
