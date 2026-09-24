# /// script
# dependencies = [
#   "trl>=0.12.0",
#   "peft>=0.7.0",
#   "transformers>=4.45.0",
#   "datasets>=3.0.0",
#   "trackio",
# ]
# ///
"""SFT a small model on SmolDataEnvs-sft.

The simplest thing that trains on SmolDataEnvs. `SmolDataEnvs-sft` ships
conversational `messages` plus the `bash` tool schema in `tools`, which is exactly
what TRL's SFTTrainer wants, so there is no preprocessing step here and none is
hiding in a helper: load, train, push.

Every trajectory in that dataset reached the correct answer under the deterministic
grader, so this is imitation of known-correct work rather than of plausible-looking
work. It is the warm start for the RL run, not a replacement for it.

Run it locally:

    uv run train_sft.py

Run it on a Hugging Face Jobs GPU (see run_on_hf_jobs.sh for the full command):

    hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN \\
      --image huggingface/trl train_sft.py

Everything is set through environment variables so the same file works in both
places without editing it.
"""

import os

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# ── configuration ────────────────────────────────────────────────────────────
# Qwen3.5-2B is the model the RL run uses, so SFT here is a warm start for that
# rather than a separate experiment. MODEL=HuggingFaceTB/SmolLM2-360M-Instruct
# runs anywhere in minutes if you only want to prove the path.
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
DATASET = os.environ.get("DATASET", "FineEnvs/SmolDataEnvs-sft")
RUN_NAME = os.environ.get("RUN_NAME", f"smoldataenvs-sft-{MODEL.split('/')[-1]}")
HUB_MODEL_ID = os.environ.get("HUB_MODEL_ID", "")  # e.g. you/smoldataenvs-sft-360m
TRACKIO_SPACE = os.environ.get("TRACKIO_SPACE", "")  # e.g. you/trackio

EPOCHS = float(os.environ.get("EPOCHS", 1))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 8192))  # these trajectories are long
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 8))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-5))
# Set MAX_SAMPLES to a small number for a smoke test: 32 finishes on CPU.
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 0))
USE_LORA = os.environ.get("USE_LORA", "1") != "0"

# ── data ─────────────────────────────────────────────────────────────────────
ds = load_dataset(DATASET, split="train")
if MAX_SAMPLES:
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

# A held-out slice so the loss curve has something to be checked against. It is
# cut from the same pool on purpose: the real held-out measurement is pass@1 on
# SmolDataEnvs-harbor-test, which needs a sandbox and is a separate run.
split = ds.train_test_split(test_size=0.05, seed=42)
print(f"{DATASET}: {len(split['train'])} train, {len(split['test'])} eval trajectories")

if TRACKIO_SPACE:
    os.environ.setdefault("TRACKIO_SPACE_ID", TRACKIO_SPACE)

# ── train ────────────────────────────────────────────────────────────────────
args = SFTConfig(
    output_dir=RUN_NAME,
    num_train_epochs=EPOCHS,
    max_length=MAX_LENGTH,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    gradient_checkpointing=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    bf16=True,
    report_to="trackio",
    project="smoldataenvs",
    run_name=RUN_NAME,
    # The Jobs environment is deleted when the job ends. Without this the run is lost.
    push_to_hub=bool(HUB_MODEL_ID),
    hub_model_id=HUB_MODEL_ID or None,
    hub_strategy="every_save",
)

trainer = SFTTrainer(
    model=MODEL,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05) if USE_LORA else None,
    args=args,
)

trainer.train()

if HUB_MODEL_ID:
    trainer.push_to_hub()
    print(f"pushed → https://huggingface.co/{HUB_MODEL_ID}")
else:
    trainer.save_model(RUN_NAME)
    print(f"saved locally → {RUN_NAME}/  (set HUB_MODEL_ID to push instead)")
