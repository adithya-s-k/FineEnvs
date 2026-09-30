"""Synchronous GRPO: TRL owns generation and invokes native bash/SETA tools."""
from pathlib import Path
import json
import math

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from whitebox_bash import white_box_bash_env

from recipe import schedule, task_rows, write_json
from runtime.models import tokenizer_for
from train.logging_utils import Checkpoints, Logs, publish_checkpoint
from train.generation import protect_generation

SYSTEM = ("You are a terminal agent working in a sandbox. Use the available tools to inspect the "
          "files and solve the task. Call submit_solution with the final answer itself, "
          "not a command that would produce it.")


class TokenCheckedTrainer(GRPOTrainer):
    def _generate_single_turn(self, prompt_ids, *args, **kwargs):
        completions, logprobs = super()._generate_single_turn(prompt_ids, *args, **kwargs)
        for prompt, completion, lps in zip(prompt_ids, completions, logprobs, strict=True):
            if len(completion) != len(lps) or not all(math.isfinite(p) for p in lps):
                raise ValueError("Invalid whitebox engine token/logprob capture")
            self._captured_calls.append({"prompt_ids": prompt.copy(), "completion_ids": completion.copy(),
                                         "logprobs": lps.copy()})
        return completions, logprobs

    def _generate(self, *args, **kwargs):
        self._captured_calls = []
        return super()._generate(*args, **kwargs)

    def _tool_call_loop(self, prompts, prompt_ids, *args, **kwargs):
        from whitebox_tito import audit_rows
        result = super()._tool_call_loop(prompts, prompt_ids, *args, **kwargs)
        masks, _, completions, logps, _, _, _ = result
        checks = audit_rows(prompt_ids, completions, masks, logps, self._captured_calls)
        with (Path(self.args.output_dir) / "token-audit.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": self.state.global_step, "rows": checks,
                                     "engine_calls": len(self._captured_calls)}) + "\n")
        return result


def train(cfg, data, server, vllm, resume=None):
    from run import unused_port
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    groups = schedule(cfg, task_rows("train"))
    write_json(output / "schedule.json", groups)
    rows = [{"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Solve the task."}],
             "split": "train", "index": g["task_index"]} for g in groups]
    args = GRPOConfig(output_dir=str(output), learning_rate=cfg["learning_rate"],
        lr_scheduler_type="constant", warmup_steps=0, beta=0.0, loss_type="dapo",
        max_steps=cfg["max_steps"], num_train_epochs=1, shuffle_dataset=False,
        num_generations=cfg["num_generations"], per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg["batch_size"] * cfg["gradient_accumulation_steps"],
        max_completion_length=cfg["max_completion_length"], max_tool_calling_iterations=cfg["agent_step_limit"] - 1,
        temperature=cfg["temperature"], top_p=1.0, top_k=0,
        chat_template_kwargs={"enable_thinking": False, "preserve_thinking": True},
        generation_kwargs={"max_tokens": cfg["max_output_tokens"]},
        model_init_kwargs={"revision": cfg["profile"]["revision"], "dtype": "bfloat16"},
        optim="paged_adamw_8bit", bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True, vllm_mode="server", vllm_server_base_url=vllm,
        vllm_group_port=unused_port(),
        vllm_server_timeout=900, vllm_max_model_length=cfg["max_model_len"],
        save_strategy="steps", save_steps=cfg["save_steps"], save_total_limit=None,
        logging_steps=1, report_to="trackio", project=cfg["project"], run_name=cfg["run_name"],
        trackio_space_id=cfg["trackio_space_id"], seed=cfg["seed"])
    trainer = TokenCheckedTrainer(model=cfg["profile"]["id"], processing_class=tokenizer_for(cfg), args=args,
        train_dataset=Dataset.from_list(rows), reward_funcs=[],
        environment_factory=white_box_bash_env(server, split="train", toolsets="bash,seta", step_limit=cfg["agent_step_limit"]),
        callbacks=[Logs(), Checkpoints(cfg)])
    # This TRL version reads the outer config for text-only tool rollouts.
    text_config = trainer.model.config.get_text_config()
    trainer.model.config.max_position_embeddings = min(text_config.max_position_embeddings, cfg["max_model_len"])
    write_json(output / "trainer-config.json", args.to_dict())
    protect_generation(trainer)
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    checkpoint = output / f"checkpoint-{trainer.state.global_step}"
    if not (checkpoint / "checkpoint.saved.json").exists():
        trainer._save_checkpoint(trainer.model, None)
    publish_checkpoint(checkpoint, cfg, trainer.state.global_step, final=True)
