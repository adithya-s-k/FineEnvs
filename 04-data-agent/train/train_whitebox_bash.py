# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train the white-box bash/SETA agent with TRL's SYNCHRONOUS GRPOTrainer.

HOW THE DATASET SELECTS THE TASK
TRL forwards the whole dataset row to `reset()` as kwargs
(`grpo_trainer.py`: `reset_kwargs = x; environment.reset(**reset_kwargs)`), so a row carrying
`split` and `index` lands on `WhiteBoxBashEnv.reset(split=..., index=...)` and starts that exact
episode. That is the entire task-selection mechanism -- there is no side channel, and the row is the
single place a task is chosen.

`prompt` is the role framing only. The task text arrives from `reset()`, which TRL appends to the
last user message; the gold answer never leaves the server, so the trainer cannot see it.

WHY NO REWARD FUNCTION
The environment owns its reward through `get_reward()`, so `reward_funcs` is empty. TRL exposes the
env's reward as a column named after the env class. Passing a reward function as well would add a
second, unweighted reward source and quietly change the objective.

ONE GPU
vLLM runs in COLOCATE mode, in-process on the training GPU, so the smoke needs no separate server and
no 2-GPU allocation. `vllm_gpu_memory_utilization` is the knob that makes that fit -- the trainer and
the engine share one card, and the default 0.9 would leave nothing for the optimizer states.
"""

from __future__ import annotations

import argparse
import os

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from whitebox_bash import exposed_tool_names, white_box_bash_env


SYSTEM = (
    "You are a terminal agent working in a sandbox. Use the available tools to inspect the "
    "filesystem and solve the task. Work step by step: look before you act. When you are confident, "
    "call submit_solution with the final answer and nothing else -- not the command that would "
    "produce it."
)


def build_dataset(server: str, split: str, limit: int = 0) -> Dataset:
    """One row per task: the role framing plus the coordinates `reset()` needs.

    Task text is deliberately NOT baked in here. It comes from `reset()` at rollout time, which keeps
    one source of truth (the server) and means the trainer never holds anything the agent should have
    had to discover.
    """
    import httpx

    # Only the COUNT is needed: the row carries `split`/`index`, and the task text comes from
    # `reset()`. The Task API routes are POST with a JSON body, not GET with query params.
    r = httpx.post(f"{server.rstrip('/')}/white_box_bash/num_tasks",
                   json={"split": split}, timeout=60.0)
    r.raise_for_status()
    payload = r.json()
    n = int(payload if isinstance(payload, int) else payload.get("num_tasks", payload.get("count", 0)))
    if limit:
        n = min(n, limit)
    if n <= 0:
        raise SystemExit(f"split {split!r} has no tasks; nothing to train on")
    return Dataset.from_list([
        {
            "prompt": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": "Solve the task."}],
            # Read by `reset()` via TRL's row-as-kwargs forwarding. The names must match the
            # signature exactly: a misspelling is silently dropped and every rollout gets index 0.
            "split": split,
            "index": i,
        }
        for i in range(n)
    ])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen3.5-2B"))
    p.add_argument("--server", default=os.environ.get("WHITE_BOX_BASH_URL", "http://127.0.0.1:8412"))
    p.add_argument("--split", default=os.environ.get("SPLIT", "train"))
    p.add_argument("--toolsets", default=os.environ.get("TOOLSETS", "bash,seta"))
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "runs/whitebox-bash"))
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "4")))
    p.add_argument("--num-generations", type=int, default=int(os.environ.get("NUM_GENERATIONS", "4")))
    p.add_argument("--per-device-train-batch-size", type=int,
                   default=int(os.environ.get("PER_DEVICE_BS", "4")))
    # 1e-6, not 3e-6, and beta 0.04 rather than 0. Both come from a previous SYNC GRPO run on this
    # same dataset (experiments/rollout_control/harbor_trl, FAILURE_MODES.md §E2): with `beta=0` and
    # LR 2e-6 BOTH Qwen3.5-2B runs collapsed -- entropy fell 0.4 -> 0.09 by step 40, the policy
    # degenerated into malformed repetition, and reward went to 0. Nothing anchors a small policy
    # without a KL term. The 4B tolerated it; the 2B does not.
    p.add_argument("--learning-rate", type=float, default=float(os.environ.get("LR", "1e-6")))
    p.add_argument("--kl-beta", type=float, default=float(os.environ.get("KL_BETA", "0.04")))
    p.add_argument("--warmup-steps", type=int, default=int(os.environ.get("WARMUP_STEPS", "10")))
    # The LOGITS batch, and the thing that OOMs. The loss materialises
    # `per_device_batch x seq_len x vocab` in fp32, and Qwen3.5's vocab is ~152k: at batch 8 and a
    # 4096 completion budget that is ~24 GB and the run dies in `logits / temperature` (job 76717).
    # Keep this small and recover the effective batch with gradient accumulation -- GRPO only needs
    # the GROUP intact, and `per_device_bs * grad_accum` still covers `num_generations`.
    p.add_argument("--gradient-accumulation-steps", type=int,
                   default=int(os.environ.get("GRAD_ACCUM", "4")))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", "0.8")))
    # Bounds the WHOLE multi-turn completion, tool-result tokens included -- they sit in
    # `completion_ids` (masked out of the loss and out of the length metric, but still counted here).
    # 1024 was too small for real data: one clipped CSV read consumed it and the agent could never
    # take a second turn.
    p.add_argument("--max-completion-length", type=int,
                   default=int(os.environ.get("MAX_COMPLETION_LENGTH", "4096")))
    # The multi-turn cap on the SYNC path. Each iteration is one generate + one tool call, so this is
    # the real bound on episode length; the client's own step_limit backs it up from the other side.
    p.add_argument("--max-tool-calling-iterations", type=int,
                   default=int(os.environ.get("MAX_TOOL_ITERS", "8")))
    p.add_argument("--step-limit", type=int, default=int(os.environ.get("STEP_LIMIT", "12")))
    p.add_argument("--limit-tasks", type=int, default=int(os.environ.get("LIMIT_TASKS", "0")))
    p.add_argument("--report-to", default=os.environ.get("REPORT_TO", "none"))
    # Saving is OFF by default because the 4-step smoke has nothing worth keeping; a real run turns
    # it on. `save_steps` is in TRAINER steps, and with num_generations completions per prompt a step
    # is one prompt, so 100 steps is one pass over a 100-task split.
    p.add_argument("--save-steps", type=int, default=int(os.environ.get("SAVE_STEPS", "0")))
    args = p.parse_args()

    dataset = build_dataset(args.server, args.split, args.limit_tasks)
    factory = white_box_bash_env(
        args.server, split=args.split, toolsets=args.toolsets, step_limit=args.step_limit,
    )
    print(f"[whitebox-bash] model={args.model} server={args.server} split={args.split} "
          f"tasks={len(dataset)} tools={exposed_tool_names(args.toolsets)}", flush=True)

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        # KL anchor to the reference model. Costs a second model in memory, which is why
        # vllm_gpu_memory_utilization is kept low.
        beta=args.kl_beta,
        warmup_steps=args.warmup_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        temperature=args.temperature,
        # Qwen3.5's template opens <think> by default on the 4B and closes it on the 2B -- INVERTED
        # between two models of the same family. Pinning it here removes that as a variable.
        chat_template_kwargs={"enable_thinking": False},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        log_completions=True,
        num_completions_to_print=1,
        logging_steps=1,
        save_strategy="steps" if args.save_steps > 0 else "no",
        **({"save_steps": args.save_steps, "save_total_limit": None} if args.save_steps > 0 else {}),
        report_to=args.report_to,
        # vLLM in-process on the training GPU. 0.3 leaves room for the optimizer states; the default
        # 0.9 would OOM the trainer on the same card.
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.3")),
        vllm_max_model_length=int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384")),
    )

    GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        environment_factory=factory,
        reward_funcs=[],  # the environment owns the reward via get_reward()
    ).train()


if __name__ == "__main__":
    main()
