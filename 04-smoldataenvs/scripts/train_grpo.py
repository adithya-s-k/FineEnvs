# /// script
# dependencies = [
#   "trl>=0.27.0",
#   "transformers>=4.56.0",
#   "datasets>=3.0.0",
#   "huggingface_hub>=1.22.0",
#   "math-verify",
#   "trackio",
# ]
# ///
"""GRPO on SmolDataEnvs. One turn, one sandbox, one deterministic reward.

No environment framework. The loop is:

    prompt  ->  model writes a Python program  ->  Hugging Face Sandbox runs it
            ->  SmolDataEnvs' grader compares the printed value to the gold answer

`rollout.py` next to this file is the whole environment; this file is only the
training wiring. Generation is colocated in the trainer process (vLLM inside the
same GPU, no separate server to stand up).

Local dry run, no GPU, no training — checks prompts, sandbox and rewards:

    uv run train_grpo.py --dry-run

On a Hugging Face Jobs GPU (see run_on_hf_jobs.sh):

    hf jobs uv run --flavor a10g-large --timeout 4h --secrets HF_TOKEN \\
      -e HUB_MODEL_ID=you/smoldataenvs-grpo-2b train_grpo.py
"""

from __future__ import annotations

import os
import sys
import urllib.request

# ── the environment lives in rollout.py ──────────────────────────────────────
# Jobs runs a single file fetched by URL, so a sibling import will not resolve
# there. Use the local copy when running from the repo, otherwise pull the same
# file from the branch this script came from.
ROLLOUT_URL = os.environ.get(
    "ROLLOUT_URL",
    "https://raw.githubusercontent.com/adithya-s-k/FineEnvs/main/04-smoldataenvs/scripts/rollout.py",
)
if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout.py")):
    with open("rollout.py", "wb") as fh:
        fh.write(urllib.request.urlopen(ROLLOUT_URL).read())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from rollout import SandboxRunner, build_prompt, rollout  # noqa: E402

# ── configuration ────────────────────────────────────────────────────────────
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
DATASET = os.environ.get("DATASET", "FineEnvs/SmolDataEnvs")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "")
RUN_NAME = os.environ.get("RUN_NAME", "smoldataenvs-grpo")

# Hyperparameters follow the 2B run the dataset card's curves come from
# (04-data-agent/train/launch_harbor_multi.slurm): lr 3e-6, temperature 0.8,
# top_p 1.0, 8 generations, grad-accum 4, per-device batch 4, bf16. What does not
# carry over is the completion budget: that run was a 17-step agent loop with a
# 16k completion window, this one is a single program.
NUM_TASKS = int(os.environ.get("NUM_TASKS", 256))  # tasks drawn from the train split
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", 8))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 200))
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", 1536))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 3e-6))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.8))
TOP_P = float(os.environ.get("TOP_P", 1.0))
PER_DEVICE_BATCH = int(os.environ.get("PER_DEVICE_BATCH", 4))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 4))
# vLLM shares the GPU with the model being trained, so it gets a slice, not the card.
VLLM_MEM = float(os.environ.get("VLLM_MEM", 0.3))

RUNNER = SandboxRunner()

# ── reward ───────────────────────────────────────────────────────────────────
# TRL hands every reward function the batch of completions plus each dataset
# column as a list. GRPO asks for NUM_GENERATIONS completions of the same task in
# a row, so the runner's per-task data pull happens once and is then reused.


def _grade_batch(completions, **columns) -> list[dict]:
    rows = [
        {k: columns[k][i] for k in ("answer", "reward_mode", "atol", "rtol", "bucket_prefix")}
        for i in range(len(completions))
    ]
    return [rollout(RUNNER, row, c) for row, c in zip(rows, completions)]


_cache: dict[int, list[dict]] = {}


def _results(completions, **columns) -> list[dict]:
    """Both reward functions see the same batch; run the sandbox once for it."""
    key = id(completions)
    if key not in _cache:
        _cache.clear()
        _cache[key] = _grade_batch(completions, **columns)
    return _cache[key]


def reward_correct(completions, **columns) -> list[float]:
    """1.0 when the printed value matches the gold answer under the dataset's grader."""
    return [r["reward"] for r in _results(completions, **columns)]


def reward_ran(completions, **columns) -> list[float]:
    """0.1 for a program that ran and printed something.

    Shaping, not scoring: early in training almost everything is wrong, and
    without this the advantage inside a group is zero for every completion and
    there is nothing to learn from.
    """
    return [0.1 * r["ran"] for r in _results(completions, **columns)]


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train").shuffle(seed=42).select(range(NUM_TASKS))
    ds = ds.map(lambda row: {"prompt": build_prompt(row)})
    keep = {"prompt", "answer", "reward_mode", "atol", "rtol", "bucket_prefix"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    print(f"{DATASET}: {len(ds)} training tasks, {NUM_GENERATIONS} generations each")

    if "--dry-run" in sys.argv:
        _dry_run(ds)
        return

    from trl import GRPOConfig, GRPOTrainer

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[reward_correct, reward_ran],
        train_dataset=ds,
        args=GRPOConfig(
            output_dir=RUN_NAME,
            # generation runs inside the trainer process: no vLLM server to stand
            # up, no second GPU, nothing to keep in sync.
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=VLLM_MEM,
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION_LENGTH,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            bf16=True,
            logging_steps=1,
            save_steps=50,
            log_completions=True,
            report_to="trackio",
            project="smoldataenvs",
            run_name=RUN_NAME,
            push_to_hub=bool(HUB_MODEL_ID),
            hub_model_id=HUB_MODEL_ID or None,
        ),
    )
    try:
        trainer.train()
        if HUB_MODEL_ID:
            trainer.push_to_hub()
            print(f"pushed → https://huggingface.co/{HUB_MODEL_ID}")
    finally:
        RUNNER.close()


def _dry_run(ds) -> None:
    """Exercise everything except the GPU: prompts, sandbox, grader, reward shapes."""
    n = int(os.environ.get("DRY_RUN_TASKS", 2))
    batch = ds.select(range(n))
    columns = {k: batch[k] for k in ("answer", "reward_mode", "atol", "rtol", "bucket_prefix")}
    # stand in for the policy: one program that is right, one that crashes
    completions = []
    for i in range(n):
        completions.append(
            f"```python\nprint({columns['answer'][i]!r})\n```" if i % 2 == 0 else "```python\n1/0\n```"
        )
    try:
        correct = reward_correct(completions, **columns)
        ran = reward_ran(completions, **columns)
    finally:
        RUNNER.close()
    print("\nprompt sent to the model:\n" + "-" * 70)
    print(batch[0]["prompt"][-1]["content"][:600])
    print("-" * 70)
    for i in range(n):
        print(f"  completion {i}: correct={correct[i]:.1f} ran={ran[i]:.1f} gold={columns['answer'][i]!r}")
    assert correct[0] == 1.0 and ran[0] == 0.1, "a correct program should score 1.0 and run"
    assert correct[1] == 0.0, "a crashing program should score 0.0"
    print("\ndry run ok: rewards line up, sandbox torn down")


if __name__ == "__main__":
    main()
