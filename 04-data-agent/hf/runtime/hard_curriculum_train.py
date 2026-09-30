"""Finite Harbor task passes with explicit checkpointed admission accounting."""
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import queue
import time

import atomic_rollouts as atomic
from trl.experimental.async_grpo.async_rollout_worker import RolloutGroup
from training_audit import write_json
from transformers import TrainerCallback


def pending_groups(limit, state):
    if limit <= 0 or state.get("group_limit", limit) != limit:
        raise ValueError("Curriculum group limit changed")
    settled = set(state.get("settled_groups", []))
    if any(type(g) is not int or not 0 <= g < limit for g in settled):
        raise ValueError("Invalid saved curriculum group")
    return [g for g in range(limit) if g not in settled]


class FiniteLoop(atomic.AtomicHarnessLoop):
    def __init__(self, *, curriculum_limit, curriculum_state, curriculum_audit, **kwargs):
        self.curriculum_limit = curriculum_limit
        self.curriculum_state = curriculum_state
        self.curriculum_audit = Path(curriculum_audit)
        super().__init__(**kwargs)
        if self.environment_factories or self.tools or curriculum_limit > len(self.dataset):
            raise ValueError("Finite runner requires loop-owning Harbor and a complete frozen schedule")

    async def _generate_loop(self, stop_event):
        async def generate(group_id):
            row = self.dataset[group_id]
            results = await asyncio.gather(*[
                self._generate_one(row["prompt"], {}, [], group_id)
                for _ in range(self.num_generations)
            ])
            group = RolloutGroup(
                prompts=[row["prompt"] for _ in results],
                reward_kwargs={k: [v] * len(results) for k, v in row.items()
                               if k not in {"prompt", "completion", "completion_ids"}},
                completions=[r[0] for r in results], completions_ids=[r[1] for r in results],
                completions_sequences=[r[2] for r in results],
                tool_call_counts=[r[3] for r in results], tool_failure_counts=[r[4] for r in results],
                model_version=self.model_version, group_id=group_id,
                env_rewards=[None] * len(results), rollout_rewards=[r[5] for r in results],
            )
            group.queued_at = time.monotonic()
            await self._groups_to_score.put(group)

        todo = iter(pending_groups(self.curriculum_limit, self.curriculum_state))
        pending = set()
        exhausted = False
        self._generation_start_time = time.monotonic()
        try:
            while not stop_event.is_set():
                self._heartbeat_value.value = time.time()
                while not exhausted and len(pending) < max(1, self.max_inflight_tasks // self.num_generations):
                    group_id = next(todo, None)
                    if group_id is None:
                        exhausted = True
                        break
                    pending.add(asyncio.create_task(generate(group_id)))
                self._inflight = len(pending) * self.num_generations
                if not pending:
                    break
                done, pending = await asyncio.wait(pending, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            await self._groups_to_score.put(None)
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _score_group(self, group):
        bundles = await super()._score_group(group)
        write_json(self.curriculum_audit / "groups" / f"{group.group_id:06d}.json", {
            "group_id": group.group_id, "requested": self.num_generations,
            "scorable_rollouts": len(bundles), "rollout_ids": [b.rollout_id for b in bundles],
        })
        return bundles

    async def _score_loop(self, stop_event):
        await super()._score_loop(stop_event)
        while not stop_event.is_set():
            try:
                self.rollout_buffer.put_nowait(atomic.RolloutsFinished())
                break
            except queue.Full:
                self._heartbeat_value.value = time.time()
                await asyncio.sleep(0.1)
        # Keep native process-health checks valid until the trainer consumes the marker.
        while not stop_event.is_set():
            self._heartbeat_value.value = time.time()
            await asyncio.sleep(0.1)


class FiniteWorker(atomic.AtomicHarnessWorker):
    _loop_cls = FiniteLoop

    def __init__(self, **kwargs):
        self.curriculum_limit = int(os.environ["CURRICULUM_GROUP_LIMIT"])
        state_path = os.environ.get("CURRICULUM_RESUME_STATE")
        self.curriculum_state = json.loads(Path(state_path).read_text()) if state_path else {}
        if self.curriculum_state and self.curriculum_state.get("execution_schedule_sha256") != os.environ["CURRICULUM_SCHEDULE_SHA256"]:
            raise ValueError("Resume belongs to a different curriculum schedule")
        pending_groups(self.curriculum_limit, self.curriculum_state)
        super().__init__(curriculum_limit=self.curriculum_limit,
                         curriculum_state=self.curriculum_state,
                         curriculum_audit=os.environ["CURRICULUM_AUDIT"], **kwargs)


class FinishCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer._rollout_dataset.exhausted:
            control.should_training_stop = True
            control.should_save = True


class FiniteTrainer(atomic.AtomicRolloutTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(FinishCallback(self))

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        previous = self.rollout_worker.curriculum_state
        admitted = dict(previous.get("admitted_rollouts", {}))
        receipts = self.admission_dir / "optimizer_rollouts.jsonl"
        if receipts.exists():
            for line in receipts.read_text().splitlines():
                entry = json.loads(line)
                if entry["step"] <= self.state.global_step:
                    for row in entry["rollouts"]:
                        admitted[row["rollout_id"]] = row["group_id"]
        counts = Counter(admitted.values())
        settled = set(previous.get("settled_groups", [])) | set(counts)
        for path in (self.admission_dir / "groups").glob("*.json"):
            group = json.loads(path.read_text())
            if group["scorable_rollouts"] == 0:
                settled.add(group["group_id"])
        write_json(checkpoint / "curriculum_state.json", {
            "schema_version": 1, "group_limit": self.rollout_worker.curriculum_limit,
            "schedule_exhausted": self._rollout_dataset.exhausted,
            "execution_schedule_sha256": os.environ["CURRICULUM_SCHEDULE_SHA256"],
            "settled_groups": sorted(settled), "admitted_rollouts": admitted,
            "partial_groups_not_replayed_on_resume": {str(g): n for g, n in counts.items()
                                                     if n < self.rollout_worker.num_generations},
            "resume_policy": "Do not replay groups with committed optimizer work. Unconsumed tails of partial groups are abandoned and reported; uncommitted groups may be regenerated.",
        })
        # Group identities are absolute in this runner, including after restart.
        write_json(checkpoint / "rollout_state.json", {"prompt_index": 0, "model_version": self.model_version})


def main():
    import sys
    import train_harbor_multi
    from checkpoint_artifacts import mark_saved
    if '--help' not in sys.argv and '-h' not in sys.argv:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        if qwen.chunk_gated_delta_rule is None:
            raise RuntimeError('Qwen3.5 training requires the pinned FLA kernels')
        print('Training Gated DeltaNet kernel: ' + qwen.chunk_gated_delta_rule.__module__, flush=True)
    atomic.AtomicHarnessWorker = FiniteWorker
    atomic.AtomicRolloutTrainer = FiniteTrainer
    # Ensure a final full optimizer checkpoint even when exhaustion falls exactly
    # on an accumulation boundary and is discovered by the next empty read.
    original_train = FiniteTrainer.train

    def train(self, *args, **kwargs):
        result = original_train(self, *args, **kwargs)
        self._save_checkpoint(self.model, None)
        mark_saved(Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}",
                   self.state.global_step, "Qwen/Qwen3.5-2B",
                   "15852e8c16360a2fea060d615a32b45270f8a8fc", final=True)
        return result

    FiniteTrainer.train = train
    train_harbor_multi.main()


if __name__ == "__main__":
    main()
