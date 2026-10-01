"""Train through four Harbor harnesses with AsyncGRPO. Read the numbered sections in order."""

# %% 1. Choose the model and run settings.
import argparse
import json
import os
from functools import partial
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
    parser.add_argument("--server", default="http://127.0.0.1:8200")
    parser.add_argument("--trials", default="runs/trials")
    parser.add_argument("--output", default="runs/multi_harness")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--inflight", type=int, default=32)
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


# %% 3. Select one harness per task. All eight rollouts use the same assignment.
from harbor_env import HarborEnv
from harbor_env.harness import HarborSession

HARNESSES = ("opencode", "claude-code", "codex", "mini-swe-agent")


class TaskSession(HarborSession):
    def fetch_training_trace(self):
        trace = super().fetch_training_trace()
        # ATIF counts actual tool actions, not model requests or training rows.
        path = self.trial_root / self.result.trial_name / "agent/trajectory.json"
        try:
            trajectory = json.loads(path.read_text())
            if not trajectory["schema_version"].startswith("ATIF-"):
                raise ValueError("Unknown trajectory format")
            ids = [
                call["tool_call_id"]
                for step in trajectory["steps"]
                if step.get("source") == "agent"
                for call in step.get("tool_calls", []) or []
            ]
            if any(not value for value in ids) or len(set(ids)) != len(ids):
                raise ValueError("Missing or repeated tool action IDs")
            calls = len(ids)
        except (OSError, ValueError, KeyError, TypeError):
            calls = None
        for turn in trace.turns:
            turn.metadata["native_tool_calls"] = calls
        return trace


class TaskFactory:
    def __init__(self, args, tasks, *, sampling):
        self.args, self.sampling = args, sampling
        self.tasks = {task["instruction"]: (i, task) for i, task in enumerate(tasks)}

    def create(self, task, seed=None, episode_id=None):
        # Match by prompt, not by the worker seed: seeds can change after a restart.
        index, row = self.tasks[task[-1]["content"]]
        session = TaskSession(
            env=HarborEnv(self.args.server, message_timeout_s=1800),
            owns_env=True,
            split=str(Path(self.args.data).resolve() / "datasets/train"),
            task_index=index,
            instruction=row["instruction"],
            harness=HARNESSES[index % len(HARNESSES)],
            sandbox="daytona",
            llm_url=self.args.vllm_url,
            model=self.args.model,
            sampling=self.sampling,
            reward_key="correctness",
            agent_step_limit=17,
            agent_timeout_sec=600,
        )
        session.trial_root = Path(self.args.trials)
        return session


# %% 4. Reward correctness, with a small bonus for fewer verified tool calls.
def reward(outcome):
    if outcome.env_reward is None:
        return None  # Infrastructure failures do not become wrong answers.
    calls = (
        outcome.trace[0]["metadata"].get("native_tool_calls") if outcome.trace else None
    )
    bonus = 0.1 * 15 / (15 + calls) if calls is not None and calls > 0 else 0.0
    return outcome.env_reward * (1 + bonus)


# %% 5. Configure public AsyncGRPO. These are the experiment's hyperparameters.
def main():
    import torch
    from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
    from trl.experimental.async_grpo.openenv_harness import HarnessRolloutWorker

    args = arguments()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TRACKIO_DIR"] = str(output.resolve() / "trackio")
    tasks = load_tasks(args.data, "train")
    dataset = Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": task["instruction"]}]}
            for task in tasks
        ]
    ).shuffle(seed=0)
    tokenizer = tokenizer_for(args.model)
    config = AsyncGRPOConfig(
        output_dir=str(output),
        learning_rate=3e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_steps=args.steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=8,
        max_completion_length=16384,
        temperature=0.8,
        top_p=1.0,
        top_k=-1,
        max_inflight_tasks=args.inflight,
        max_staleness=4,
        token_budget=40960,
        fork_threshold_tokens=0,
        optim="paged_adamw_8bit",
        bf16=True,
        dtype="bfloat16",
        model_init_kwargs={
            "revision": MODEL_REVISIONS[args.model],
            "use_kernels": "lfm" in args.model.lower(),
        },
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        vllm_server_base_url=args.vllm_url,
        heartbeat_stale_after_s=900,
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
    worker = HarnessRolloutWorker(
        harness_session_factory=partial(TaskFactory, args, tasks),
        harness_adapter=None,  # OpenEnv captures a CLI agent's own model calls.
        rollout_reward_fn=reward,
        lossless_capture=True,
        model_name=args.model,
        processing_class=tokenizer,
        dataset=dataset,
        reward_funcs=[],
        num_generations=8,
        max_inflight_tasks=args.inflight,
        vllm_server_url=args.vllm_url,
        max_tokens=16384,
        temperature=0.8,
        top_p=1.0,
        top_k=-1,
        chat_template_kwargs={"enable_thinking": False},
    )
    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        processing_class=tokenizer,
        train_dataset=dataset,
        rollout_worker=worker,
    )
    if trainer.model.config.model_type == "lfm2":
        # Packed samples reset attention positions. LFM's convolutions also need sequence IDs.
        def reset_convolutions(module, positional, keyword):
            keyword["seq_idx"] = ((keyword["position_ids"] == 0).cumsum(-1) - 1).to(
                torch.int32
            )
            return positional, keyword

        trainer.model.register_forward_pre_hook(reset_convolutions, with_kwargs=True)
        # Abort if the installed convolution kernel ignores packed sequence boundaries.
        trainer.model.eval()
        with torch.no_grad():
            ids = torch.tensor(
                [[10, 20, 30, 40, 50, 60, 70, 80]], device=trainer.model.device
            )
            separate = trainer.model(
                input_ids=ids[:, 4:],
                position_ids=torch.arange(4, device=ids.device)[None],
                use_cache=False,
            ).logits
            packed = trainer.model(
                input_ids=ids,
                position_ids=torch.arange(4, device=ids.device).repeat(2)[None],
                use_cache=False,
            ).logits[:, 4:]
            torch.testing.assert_close(packed, separate, atol=0.05, rtol=0.01)
        trainer.model.train()

    # 6. Train, then save weights and tokenizer for evaluation.
    (output / "training_config.json").write_text(json.dumps(config.to_dict(), indent=2))
    (output / "task_names.json").write_text(json.dumps([t["name"] for t in tasks]))
    trainer.train()
    trainer.save_model(str(output / "final"))
    tokenizer.save_pretrained(output / "final")
    trainer.save_state()


if __name__ == "__main__":
    main()
