# /// script
# dependencies = [
#   "trl>=0.27.0",
#   "transformers>=4.56.0",
#   # torchvision: AutoProcessor for Qwen3.5 resolves a VL video processor that
#   #   imports it, even though nothing here touches video.
#   # vllm: colocate mode runs vLLM in the trainer process. uv builds the env from
#   #   this header, not from the image, so it has to be declared here.
#   "torchvision",
#   "vllm>=0.22.0",
#   "accelerate",
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

# 10GB sat reserved-but-unallocated in the run that OOMed; expandable segments
# hand that back instead of fragmenting it away.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
# Programs that actually finish run 250-900 tokens; the rest are a 2B repeating
# itself until the cap. A tighter cap costs few real answers and halves the time
# spent generating rubbish, and truncated completions are masked out anyway.
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", 1024))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 3e-6))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.8))
TOP_P = float(os.environ.get("TOP_P", 1.0))
# The reference run had per-device 4 x accum 4 across two GPUs. Here it is one
# GPU shared with vLLM, so the same effective batch of 16 is reached with a
# smaller per-device step: 2 x 8. The optimiser update is identical; only the
# activation peak changes, and that peak is what ran the A100 out of memory.
PER_DEVICE_BATCH = int(os.environ.get("PER_DEVICE_BATCH", 2))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 8))
# vLLM shares the GPU with the model being trained, so it gets a slice, not the card.
VLLM_MEM = float(os.environ.get("VLLM_MEM", 0.22))

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


def reward_correct(completions, **columns) -> list[float | None]:
    """1.0 when the printed value matches the gold answer under the dataset's grader.

    `None` for a rollout the sandbox could not run: TRL turns it into NaN and drops
    it from the group baseline, so a dead sandbox is not scored as a wrong answer.
    """
    return [r["reward"] for r in _results(completions, **columns)]


def reward_ran(completions, **columns) -> list[float]:
    """Diagnostic only: did the program run and print anything?

    Carried at weight 0 (see reward_weights below) so it shows up in the logs
    without steering the policy. It was weighted 0.1 as a shaping term, and that
    turned out to reward verbosity: when almost nothing is correct, the only way
    to earn anything is to print *something*, and a long exploratory program does
    that more reliably than a short exact one. Completion length climbed 300 ->
    900 tokens in five steps with a quarter of them hitting the cap. Correctness
    alone gives a usable signal -- the first step of every run scores around 0.3.
    """
    return [r["ran"] for r in _results(completions, **columns)]


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train").shuffle(seed=42).select(range(NUM_TASKS))
    # load_from_cache_file=False: datasets fingerprints the lambda, not the
    # prompt text it closes over, so editing the prompt silently reuses the old
    # mapped copy -- and a local dry run then tests a prompt that no longer exists.
    ds = ds.map(lambda row: {"prompt": build_prompt(row)}, load_from_cache_file=False)
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
            # Qwen3.5 thinks by default. A single program does not need a
            # reasoning preamble, and the tokens it costs come out of the
            # completion budget, so the whole run is non-thinking.
            chat_template_kwargs={"enable_thinking": False},
            vllm_gpu_memory_utilization=VLLM_MEM,
            # correctness is the reward; "it ran" is logged, not optimised
            reward_weights=[1.0, 0.0],
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION_LENGTH,
            # A completion cut off at the cap has no closing fence, so it cannot
            # compile and scores zero -- but without this flag it still receives
            # gradient as though it were a finished answer, which teaches the
            # model to ramble. The first attempt drifted 276 -> 1331 tokens in six
            # steps with three quarters of completions truncated.
            mask_truncated_completions=True,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            # 44% of completions were hitting the cap without terminating -- a
            # repetition loop, not a long answer. This is the cheapest brake.
            repetition_penalty=float(os.environ.get("REPETITION_PENALTY", 1.05)),
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            bf16=True,
            # 2B full fine-tune + Adam states + a colocated vLLM is most of an
            # 80GB card before a single activation exists.
            gradient_checkpointing=True,
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
        print(f"  completion {i}: correct={correct[i]:.1f} ran={ran[i]:.1f} (ran is logged, weight 0) gold={columns['answer'][i]!r}")
    assert correct[0] == 1.0 and ran[0] == 1.0, "a correct program should score 1.0 and run"
    assert correct[1] == 0.0, "a crashing program should score 0.0"
    print("\ndry run ok: rewards line up, sandbox torn down")


if __name__ == "__main__":
    main()
