#!/usr/bin/env python3
"""Minimal multi-turn GRPO smoke run for the six-task RetroEnv sample.

Run this from the project checkout in a TRL image/environment. It is deliberately
small: prove the base model can use the graph tools and overfit a few indexed
tasks before creating a large job.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from retroenv.store import TaskStore
from retroenv.training import RetroRouteTrainingEnv


MODEL = os.getenv("MODEL", "Qwen/Qwen3.5-4B")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/retroenv-grpo-smoke")
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "4"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "10"))
SEED = int(os.getenv("SEED", "17"))


def build_dataset() -> Dataset:
    root = Path(__file__).resolve().parents[1]
    store = TaskStore(
        os.getenv("RETROENV_TASKS_DIR", root / "sample/tasks-private"),
        os.getenv("RETROENV_STOCKS_DIR", root / "sample/stocks"),
    )
    indices = list(range(len(store.tasks("train"))))
    random.Random(SEED).shuffle(indices)
    return Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "user", "content": [{"type": "text", "text": ""}]}
                ],
                "split": "train",
                "index": index,
            }
            for index in indices
        ]
    )


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer = GRPOTrainer(
        model=MODEL,
        train_dataset=build_dataset(),
        environment_factory=RetroRouteTrainingEnv,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
        args=GRPOConfig(
            output_dir=OUTPUT_DIR,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=float(os.getenv("VLLM_MEM", "0.25")),
            num_generations=NUM_GENERATIONS,
            generation_batch_size=NUM_GENERATIONS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_tool_calling_iterations=32,
            max_completion_length=int(os.getenv("MAX_COMPLETION", "8192")),
            max_steps=MAX_STEPS,
            learning_rate=float(os.getenv("LR", "1e-5")),
            beta=0.0,
            loss_type="dr_grpo",
            scale_rewards="group",
            logging_steps=1,
            save_steps=5,
            seed=SEED,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
