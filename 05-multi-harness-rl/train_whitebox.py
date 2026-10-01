"""Train a bash/SETA agent with synchronous GRPO. Read the numbered sections in order."""

# %% 1. Choose the model and run settings.
import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from envs.tasks import load_tasks

MODEL_REVISIONS = {
    "LiquidAI/LFM2.5-2.6B": "654f9463ce32b05d0429d76fe1f580b27d4c1ac0",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    parser.add_argument("--data", default="prepared")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="runs/whitebox")
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
import math
import shlex
import socket
import time

from envs.daytona import DaytonaSandboxBackend
from envs.tasks import grade_answer, stage_task

SYSTEM = "Use the sandbox tools to solve the task. Call submit_solution with the final answer itself."


class BashEnvironment:
    def __init__(self):
        self._sandbox = None
        self._calls = 0
        self._deadline = 0
        self._submitted = False

    def reset(self, folder, **kwargs):
        self._close()
        self._folder = Path(folder)
        self._calls, self._submitted = 0, False
        self._sandbox = DaytonaSandboxBackend(
            image="docker.io/savatar101/env-data-agent-train:base"
        ).create(timeout_s=900)
        try:
            stage_task(self._sandbox, self._folder)
        except BaseException:
            self._close()
            raise
        self._deadline = time.monotonic() + 600
        return (self._folder / "instruction.md").read_text()

    def _check(self):
        if self._submitted or time.monotonic() >= self._deadline:
            raise RuntimeError("Episode is finished")

    def _execute(self, command):
        self._check()
        self._calls += 1
        result = self._sandbox.exec(
            command,
            cwd="/workdir",
            timeout=min(60, max(1, self._deadline - time.monotonic())),
        )
        return (result.stdout + result.stderr)[-16000:]

    def bash(self, command: str) -> str:
        """Run a shell command in /workdir.

        Args:
            command: The shell command to execute.
        """
        return self._execute(command)

    def read(self, path: str) -> str:
        """Read a text file.

        Args:
            path: Absolute path of the file.
        """
        return self._execute("cat -- " + shlex.quote(path))

    def write(self, path: str, content: str) -> str:
        """Write a text file, creating parent directories.

        Args:
            path: Absolute path of the file.
            content: Text to write.
        """
        self._check()
        self._calls += 1
        self._sandbox.write_text(path, content)
        return "Saved."

    def edit(self, path: str, old: str, new: str) -> str:
        """Replace one exact occurrence in a file.

        Args:
            path: Absolute path of the file.
            old: Text that must occur exactly once.
            new: Replacement text.
        """
        self._check()
        self._calls += 1
        text = self._sandbox.read_text(path)
        if text.count(old) != 1:
            return "The old text must occur exactly once."
        self._sandbox.write_text(path, text.replace(old, new, 1))
        return "Saved."

    def ls(self, path: str = "/workdir") -> str:
        """List a directory.

        Args:
            path: Directory to list.
        """
        return self._execute("ls -la -- " + shlex.quote(path))

    def grep(self, pattern: str, path: str) -> str:
        """Search text files recursively.

        Args:
            pattern: Regular expression to search for.
            path: File or directory to search.
        """
        return self._execute(
            "grep -rn -- " + shlex.quote(pattern) + " " + shlex.quote(path)
        )

    def glob(self, pattern: str, path: str = "/workdir") -> str:
        """Find files by name.

        Args:
            pattern: Filename pattern, for example *.csv.
            path: Directory to search.
        """
        return self._execute(
            "find " + shlex.quote(path) + " -name " + shlex.quote(pattern)
        )

    def submit_solution(self, answer: str) -> str:
        """Save your final answer and finish the episode.

        Args:
            answer: The answer itself, not a command.
        """
        self._check()
        self._sandbox.write_text("/workdir/answer.txt", answer)
        self._submitted = True
        return "Answer saved. End your response now."

    # 4. Grade the answer. Submitting it does not count as a tool action.
    def get_reward(self):
        try:
            self._correctness = grade_answer(self._sandbox, self._folder)
            bonus = 0.1 * 15 / (15 + self._calls) if self._calls > 0 else 0.0
            return self._correctness * (1 + bonus)
        except (OSError, RuntimeError, ValueError, TimeoutError):
            return math.nan  # No grade is different from a wrong answer.
        finally:
            self._close()

    def _close(self):
        if self._sandbox is not None:
            self._sandbox.kill()
            self._sandbox = None


# %% 5. Configure GRPO. TRL generates responses and invokes the tools above.
def main():
    from trl import GRPOConfig, GRPOTrainer

    args = arguments()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(output.resolve() / "trackio")
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
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_generations=8,
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
        save_steps=args.save_steps,
        save_total_limit=None,
        logging_steps=1,
        report_to="trackio",
        project="smoldataenv-rl",
        run_name=output.name,
        trackio_space_id=args.space_id,
        seed=0,
    )
    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[],
        environment_factory=BashEnvironment,
    )
    # 6. Train, then save weights and tokenizer for evaluation.
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
