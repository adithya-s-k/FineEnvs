"""AsyncGRPO with atomic rollout admission and a finite, resumable task schedule."""
import json
import os
from pathlib import Path

from datasets import Dataset
from hard_curriculum_train import FiniteLoop, FiniteTrainer, FiniteWorker
from trl.experimental.async_grpo import AsyncGRPOConfig

from recipe import digest, resume_state, schedule, task_rows, write_json
from runtime.models import tokenizer_for
from train.adapters import Factory, rollout_reward
from train.logging_utils import Checkpoints, Logs, publish_checkpoint


class GuardedLoop(FiniteLoop):
    async def _score_group(self, group):
        bundles = await super()._score_group(group)
        self._ungraded_streak = 0 if bundles else getattr(self, "_ungraded_streak", 0) + 1
        if self._ungraded_streak >= 3:
            raise RuntimeError("Three consecutive groups produced no trainable rollouts; inspect capture and environment logs")
        return bundles


class Worker(FiniteWorker):
    _loop_cls = GuardedLoop


def train(cfg, data, server, vllm, resume=None):
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    groups = schedule(cfg, task_rows("train"))
    write_json(output / "schedule.json", groups)
    os.environ.update(CURRICULUM_GROUP_LIMIT=str(len(groups)), CURRICULUM_SCHEDULE_SHA256=digest(groups),
                      CURRICULUM_AUDIT=str(output / "audit"))
    if resume:
        state = json.loads((Path(resume) / "curriculum_state.json").read_text())
        state = resume_state(groups, state)
        write_json(output / "resume-curriculum.json", state)
        os.environ["CURRICULUM_RESUME_STATE"] = str(output / "resume-curriculum.json")
    else:
        os.environ.pop("CURRICULUM_RESUME_STATE", None)
    if cfg["model"] == "lfm":
        from packing import install
        install()
    factory = Factory(cfg, data, server, vllm, groups, output / "trials")
    dataset = Dataset.from_list(factory.rows())
    tokenizer = tokenizer_for(cfg)
    selector = None
    if cfg["mode"] == "opencode":
        from data_agent_env import opencode_agent_turns
        selector = opencode_agent_turns
    worker = Worker(harness_session_factory=factory, harness_adapter=None,
        train_turn_fn=None, agent_turn_fn=selector, fork_threshold_tokens=0, rollout_reward_fn=rollout_reward,
        model_name=cfg["profile"]["id"], processing_class=tokenizer, dataset=dataset, reward_funcs=[],
        num_generations=cfg["num_generations"], max_inflight_tasks=cfg["max_inflight"],
        max_outstanding_rollouts=cfg["max_outstanding_rollouts"], vllm_server_url=vllm,
        max_tokens=cfg["max_completion_length"], temperature=cfg["temperature"], top_p=1.0, top_k=0,
        log_completions=False)
    args = AsyncGRPOConfig(output_dir=str(output), learning_rate=cfg["learning_rate"],
        lr_scheduler_type="constant", warmup_steps=0, max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_generations=cfg["num_generations"], max_completion_length=cfg["max_completion_length"],
        max_inflight_tasks=cfg["max_inflight"], max_staleness=cfg["max_staleness"], fork_threshold_tokens=0,
        token_budget=cfg["token_budget"], temperature=cfg["temperature"], top_p=1.0, top_k=0,
        optim="paged_adamw_8bit", bf16=True, dtype="bfloat16", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"revision": cfg["profile"]["revision"]},
        vllm_server_base_url=vllm, heartbeat_stale_after_s=900,
        save_strategy="steps", save_steps=cfg["save_steps"], save_total_limit=None,
        logging_steps=1, report_to="trackio", project=cfg["project"], run_name=cfg["run_name"],
        trackio_space_id=cfg["trackio_space_id"], seed=cfg["seed"])
    trainer = FiniteTrainer(model=cfg["profile"]["id"], args=args, train_dataset=dataset,
        rollout_worker=worker, max_row_tokens=cfg["max_model_len"], admission_dir=output / "audit")
    trainer.add_callback(Logs())
    trainer.add_callback(Checkpoints(cfg))
    write_json(output / "trainer-config.json", args.to_dict())
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    checkpoint = output / f"checkpoint-{trainer.state.global_step}"
    if not (checkpoint / "checkpoint.saved.json").exists():
        trainer._save_checkpoint(trainer.model, None)
    publish_checkpoint(checkpoint, cfg, trainer.state.global_step, final=True)
