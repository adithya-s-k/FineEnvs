# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "trl[vllm]>=1.12.0,<1.13",
#   "transformers>=5.2.0",
#   "peft",
#   "trackio",
#   "datasets",
#   "accelerate",
#   "numpy",
# ]
# ///
"""Multi-turn GRPO on Wordle, in-process.

Wordle is the cheapest multi-turn env in this repo and nobody had trained
against it. The environment is a Python function, so a hosted Space is not
a dependency and a failed step costs a few seconds rather than a GPU-hour
of waiting on HTTP.

What this script changes relative to a stock TRL GRPO run, and why:

- Reward is realised information gain, not win/loss. Early in training
  almost every group of 8 is eight failures; under a sparse reward that
  is eight zeros and a skipped gradient. The tiny-policy ablation in
  `tiny_grpo.py` is the measurement.
- `get_reward` records the group and classifies it as live / cliff /
  collapse. Cliff tasks go into a replay queue; `reset` draws from it with
  probability `REPLAY_MIX` instead of rewriting a random other row (that
  row may already have been consumed). GeoGuesser's 3,452-task split was
  visited once and never repeated; the zeros were the episodes that needed
  the repeats.
- The training signal is centered ranks over the group of 8, via a
  `reward_func` that reads `env._raw_reward`. `get_reward` returns 0 in
  that mode so TRL does not sum the raw scalar on top. `SCALE_REWARDS=none`
  then leaves the ranks as advantages.
- Completions are logged. `frac_reward_zero_std` is the number to watch,
  same as GeoGuesser. Collapse (identical guess sequences) is the number
  that says stop.

The skip-the-backward-pass half of alive GRPO is not in this file. TRL
does not expose dynamic sampling; implementing it here would mean forking
`GRPOTrainer`. `tiny_grpo.py` owns that loop. This file is the LLM recipe
that uses everything TRL will let us do without a fork.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import random
import sys
import threading
import time

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TRAIN = pathlib.Path(__file__).resolve().parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from alive import (  # noqa: E402
    AliveStats,
    ReplayQueue,
    choose_task,
    classify_group,
    group_rank_rewards,
)
from envs.wordle.core.game import WordleGame  # noqa: E402
from envs.wordle.core.rewards import reward_for  # noqa: E402
from envs.wordle.core.tasks import train_tasks  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("grpo_wordle")

MODEL = os.getenv("MODEL", "Qwen/Qwen3.5-4B")
MAX_TURNS = int(os.getenv("MAX_TURNS", "6"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "200"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "8"))
GENERATION_BATCH_SIZE = int(os.getenv("GENERATION_BATCH_SIZE", "8"))
SEED = int(os.getenv("SEED", "0"))
REWARD_SHAPE = os.getenv("REWARD_SHAPE", "process")
# `rank` is the tiny-policy default. `raw` hands TRL the process scalar
# and lets `SCALE_REWARDS` do GRPO/Dr.GRPO. Rank uses a batch reward_func;
# get_reward then returns 0 so the two sources do not sum.
ADVANTAGE = os.getenv("ADVANTAGE", "rank")
REPLAY_MIX = float(os.getenv("REPLAY_MIX", "0.5"))
NPROC = int(os.getenv("NPROC", "1"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", str(NPROC)))

RUN_NAME = os.getenv(
    "RUN_NAME",
    "wordle-{}-{}".format(
        time.strftime("%Y%m%d-%H%M", time.gmtime()),
        os.getenv("JOB_ID", "local")[:6],
    ),
)
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", os.getenv("OUTPUT_DIR", "runs"))
OUTPUT_DIR = f"{OUTPUT_ROOT.rstrip('/')}/{RUN_NAME}"
HUB_MODEL_ID = os.getenv("HUB_MODEL_ID") or (
    f"{os.getenv('HUB_ORG', 'HuggingEnvs')}/wordle-{RUN_NAME}"
)
TRACKIO_PROJECT = os.getenv("TRACKIO_PROJECT", "wordle")
TRACKIO_SPACE = os.getenv("TRACKIO_SPACE", "HuggingEnvs/wordle-trackio")

INSTRUCTION = """Play Wordle. Guess the hidden 5-letter word.

Call `guess` with a 5-letter English word. You get coloured feedback:
- green: letter is in the correct position
- yellow: letter is in the word, different position
- black: letter is not in the word

You have {max_turns} guesses. Use the feedback. Do not repeat a letter you
have already been told is black. You MUST guess every turn — an episode
that emits no guess scores zero.
"""

_TRACE_LOCK = threading.Lock()
_STATS = AliveStats()
_REPLAY = ReplayQueue()
_GROUP_BUFFER: list[tuple[int, float, tuple]] = []
_ANSWERS = list(train_tasks())
_TASK_RNG = np.random.default_rng(SEED)


class _DeadGroupCallback(TrainerCallback):
    """Publish alive-GRPO stats next to TRL's own `frac_reward_zero_std`."""

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D102
        if logs is None:
            return
        logs.update({f"alive/{k}": v for k, v in _STATS.as_dict().items()})


def _flush_group() -> None:
    if len(_GROUP_BUFFER) < NUM_GENERATIONS:
        return
    chunk = _GROUP_BUFFER[:NUM_GENERATIONS]
    del _GROUP_BUFFER[:NUM_GENERATIONS]
    task_id = chunk[0][0]
    rewards = [row[1] for row in chunk]
    fingerprints = [row[2] for row in chunk]
    kind = classify_group(rewards, fingerprints)
    _STATS.record(kind)
    _REPLAY.observe(task_id, kind)


class WordleTrainingEnv:
    """One Wordle episode, exposed to TRL as a single tool."""

    def __init__(self) -> None:
        self._game: WordleGame | None = None
        self._task_id = 0
        self._fingerprint: list[str] = []

    def reset(self, index: int = 0, **kwargs):
        task_id = choose_task(
            int(index), len(_ANSWERS), _REPLAY, _TASK_RNG, REPLAY_MIX
        )
        self._task_id = task_id
        self._fingerprint = []
        self._raw_reward = 0.0
        self._game = WordleGame(answer=_ANSWERS[task_id], max_guesses=MAX_TURNS)
        return [
            {
                "type": "text",
                "text": INSTRUCTION.format(max_turns=MAX_TURNS)
                + "\n\n"
                + self._game.observe(),
            }
        ]

    def guess(self, word: str) -> str:
        """Submit a 5-letter guess and get coloured feedback.

        Args:
            word: A 5-letter English word.
        """
        if self._game is None:
            raise RuntimeError("reset() before guess()")
        self._fingerprint.append(word.lower().strip())
        return self._game.guess(word)

    def get_reward(self) -> float:
        game = self._game
        if game is None:
            self._raw_reward = 0.0
            return 0.0
        reward = reward_for(
            REWARD_SHAPE,
            won=game.won,
            n_guesses=len(game.guesses),
            guesses=list(game.guesses),
            patterns=list(game.patterns),
            max_guesses=game.max_guesses,
            invalid_count=game.invalid_count,
        )
        self._raw_reward = float(reward)
        _GROUP_BUFFER.append((self._task_id, float(reward), tuple(self._fingerprint)))
        _flush_group()
        # Rank mode: TRL sums get_reward with reward_funcs. Return 0 here
        # so the centered ranks from `training_rewards` are the signal.
        if ADVANTAGE == "rank":
            return 0.0
        return float(reward)


def training_rewards(completions, environments, **kwargs):
    """Batch ranks (or raw scalars) over the GRPO group.

    Reads `env._raw_reward` so a group of 8 can be ranked together. TRL
    calls this with one environment per completion.
    """
    raws = [float(getattr(env, "_raw_reward", 0.0)) for env in environments]
    if ADVANTAGE != "rank":
        return raws
    return group_rank_rewards(raws, NUM_GENERATIONS)


def build_dataset(seed: int) -> Dataset:
    indices = list(range(len(_ANSWERS)))
    random.Random(seed).shuffle(indices)
    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": [{"type": "text", "text": ""}]}],
                "index": index,
            }
            for index in indices
        ]
    )


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logger.info("model %s · %d train words · reward %s", MODEL, len(_ANSWERS), REWARD_SHAPE)
    for directory in ("", "completions"):
        path = pathlib.Path(OUTPUT_DIR) / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / ".keep").write_text("")

    trainer_kwargs = {}
    if ADVANTAGE == "rank":
        trainer_kwargs["reward_funcs"] = training_rewards
    trainer = GRPOTrainer(
        model=MODEL,
        train_dataset=build_dataset(SEED),
        environment_factory=WordleTrainingEnv,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
        args=GRPOConfig(
            output_dir=OUTPUT_DIR,
            project=TRACKIO_PROJECT,
            run_name=RUN_NAME,
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=float(os.getenv("VLLM_MEM", "0.25")),
            num_generations=NUM_GENERATIONS,
            generation_batch_size=GENERATION_BATCH_SIZE,
            per_device_train_batch_size=int(os.getenv("BATCH", "1")),
            gradient_accumulation_steps=int(os.getenv("ACCUM", "2")),
            max_tool_calling_iterations=MAX_TURNS,
            max_completion_length=int(os.getenv("MAX_COMPLETION", "1024")),
            max_steps=MAX_STEPS,
            learning_rate=float(os.getenv("LR", "3e-5")),
            temperature=float(os.getenv("TEMPERATURE", "1.0")),
            chat_template_kwargs={"enable_thinking": False},
            mask_truncated_completions=True,
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=int(os.getenv("SAVE_STEPS", "25")),
            log_completions=True,
            num_completions_to_print=int(os.getenv("PRINT_COMPLETIONS", "2")),
            report_to=os.getenv("REPORT_TO", "trackio"),
            trackio_space_id=os.getenv("TRACKIO_SPACE", TRACKIO_SPACE),
            trackio_static_space_id=False,
            # Default `none`: training_rewards already returned centered
            # ranks, which are zero-mean, so subtracting the mean is a no-op
            # and dividing by std would re-introduce the 60× amplifier.
            # Set SCALE_REWARDS=group to reproduce run 1 on the raw scalar
            # (also set ADVANTAGE=raw).
            scale_rewards=os.getenv("SCALE_REWARDS", "none"),
            beta=float(os.getenv("BETA", "0.0")),
            num_iterations=int(os.getenv("NUM_ITERATIONS", "1")),
            loss_type=os.getenv("LOSS_TYPE", "dapo"),
            seed=SEED,
            push_to_hub=False,
            hub_model_id=HUB_MODEL_ID,
            hub_private_repo=False,
        ),
        **trainer_kwargs,
    )
    trainer.add_callback(_DeadGroupCallback())
    trainer.train(resume_from_checkpoint=os.getenv("RESUME") or None)
    final = pathlib.Path(trainer.args.output_dir) / "final"
    trainer.save_model(str(final))
    stats_path = pathlib.Path(OUTPUT_DIR) / "alive_stats.json"
    stats_path.write_text(json.dumps(_STATS.as_dict(), indent=2), encoding="utf-8")
    logger.info("adapter saved to %s · alive stats %s", final, stats_path)


def _relaunch_distributed() -> None:
    if NPROC <= 1 or "RANK" in os.environ:
        return
    torchrun = pathlib.Path(sys.executable).with_name("torchrun")
    argv = [str(torchrun), f"--nproc_per_node={NPROC}", "--nnodes=1", str(pathlib.Path(__file__).resolve())]
    logger.info("relaunching under torchrun on %d GPUs", NPROC)
    os.execv(str(torchrun), argv)


if __name__ == "__main__":
    _relaunch_distributed()
    main()
