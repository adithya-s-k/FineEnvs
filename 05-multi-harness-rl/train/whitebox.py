"""Train a bash/SETA agent with synchronous GRPO. Read the numbered sections in order."""

# %% 1. Choose the model and run settings.
import argparse
import json
import os
from functools import partial
from pathlib import Path

from datasets import Dataset
from smoldataenv_whitebox.tasks import load_tasks
from transformers import AutoTokenizer

MODEL_REVISIONS = {
    "LiquidAI/LFM2.5-2.6B": "654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    parser.add_argument("--server", help="Optional local or Space OpenEnv server")
    parser.add_argument("--data", default="prepared")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="runs/whitebox")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two updates, two rollouts per group and a small batch",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--space-id",
        default=None,
        help="Optional Trackio Space; local logs are always kept",
    )
    return parser.parse_args()


# %% 2. Load the tokenizer in non-thinking mode.
def tokenizer_for(model):
    from copy import deepcopy

    from trl.chat_template_utils import lfm2_2_5_template, qwen3_5_template

    tokenizer = AutoTokenizer.from_pretrained(model, revision=MODEL_REVISIONS[model])
    if "lfm" in model.lower():
        # LFM's template opens <think> even when enable_thinking=False.
        start = r'{{- "<|im_start|>assistant\n<think>" -}}'
        if tokenizer.chat_template.count(start) != 1:
            raise ValueError(
                "LFM template changed; check its non-thinking generation prefix"
            )
        tokenizer.chat_template = tokenizer.chat_template.replace(
            start, r'{{- "<|im_start|>assistant\n<think></think>" -}}'
        )
        tokenizer.response_template = deepcopy(lfm2_2_5_template)
    else:
        tokenizer.chat_template = (
            "{%- set enable_thinking = false -%}\n" + tokenizer.chat_template
        )
        tokenizer.response_template = deepcopy(qwen3_5_template)
    return tokenizer


# %% 3. Give TRL a Python environment. Public methods become model tools.
import socket

from smoldataenv_whitebox.environment import SYSTEM, BashEnvironment


# %% 4. Configure GRPO. The environment supplies the correctness and tool reward.
def main():
    args = arguments()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(output.resolve() / "trackio")
    from trl import GRPOConfig, GRPOTrainer

    tasks = load_tasks(args.data, "train")
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "Solve the task."},
                ],
                "folder": task["folder"],
            }
            for task in tasks
        ]
    ).shuffle(seed=0)
    tokenizer = tokenizer_for(args.model)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        weight_sync_port = listener.getsockname()[1]
    config = GRPOConfig(
        output_dir=str(output),
        learning_rate=3e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_steps=2 if args.smoke else args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2 if args.smoke else 16,
        num_generations=2 if args.smoke else 8,
        max_completion_length=16384,
        max_tool_calling_iterations=16,
        temperature=0.8,
        top_p=1.0,
        top_k=0,
        beta=0.0,
        loss_type="dapo",
        chat_template_kwargs={"enable_thinking": False, "preserve_thinking": True},
        generation_kwargs={"max_tokens": 4096},
        optim="paged_adamw_8bit",
        bf16=True,
        model_init_kwargs={
            "dtype": "bfloat16",
            "revision": MODEL_REVISIONS[args.model],
        },
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url=args.vllm_url,
        vllm_server_timeout=900,
        vllm_group_port=weight_sync_port,
        vllm_max_model_length=131072,
        save_strategy="steps",
        save_steps=1 if args.smoke else args.save_steps,
        save_total_limit=None,
        logging_steps=1,
        report_to="trackio",
        project="smoldataenv-rl",
        run_name=output.parent.name if output.name == "train" else output.name,
        trackio_space_id=args.space_id,
        seed=0,
    )
    from smoldataenv_whitebox.client import RemoteBashEnvironment

    environment_factory = (
        partial(RemoteBashEnvironment, args.server) if args.server else BashEnvironment
    )
    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[],
        environment_factory=environment_factory,
    )
    # TRL's text-only tool loop reads the context limit from the outer config.
    text_config = trainer.model.config.get_text_config()
    trainer.model.config.max_position_embeddings = text_config.max_position_embeddings

    # 5. Train, then save weights and tokenizer for evaluation.
    (output / "training_config.json").write_text(json.dumps(config.to_dict(), indent=2))
    (output / "task_names.json").write_text(json.dumps([t["name"] for t in tasks]))
    try:
        trainer.train()
        trainer.save_model(str(output / "final"))
        tokenizer.save_pretrained(output / "final")
        trainer.save_state()
    finally:
        # Environments are also closed after grading; this covers interrupted rollouts.
        for environment in trainer.environments or []:
            environment._close()


if __name__ == "__main__":
    main()
