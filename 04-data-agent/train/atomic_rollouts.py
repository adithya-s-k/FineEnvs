"""Single-GPU AsyncGRPO recipe that consumes every row of an admitted rollout.

The reference uses four token-packed batches per update. A forked rollout can
exceed one batch: keep it as one admission unit and stream its exact rows through
bounded forwards before the optimizer changes weights. No TRL source patch or
token rewriting is needed. The loss is a token mean over the entire update.
"""

from __future__ import annotations

import asyncio
import json
import queue
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import TrainerCallback
from trl.experimental.async_grpo import AsyncGRPOTrainer
from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout
from trl.experimental.async_grpo.openenv_harness import (
    HarnessRolloutWorker,
    _HarnessRolloutLoop,
)


@dataclass
class RolloutBundle:
    rows: list
    rollout_id: str
    enqueued_at: float | None = None

    @property
    def model_version(self):
        return self.rows[0].model_version

    @property
    def group_id(self):
        return self.rows[0].group_id

    @property
    def prompt(self):
        return self.rows[0].prompt

    @property
    def completion(self):
        return self.rows[0].completion

    @property
    def advantage(self):
        return self.rows[0].advantage

    @property
    def metrics(self):
        return self.rows[0].metrics

    @property
    def forwarded_tokens(self):
        return sum(len(row.input_ids) for row in self.rows)


class CreditQueue:
    """Native queue plus a spawn-safe budget, passed through native worker IPC.

    Credits cover generation, scoring and queued work until consumption. Putting
    the counter beside the native queue avoids the worker's ordinary-pickle
    validation of loop kwargs (shared values must use the multiprocessing spawn path).
    """

    def __init__(self, queue, credits, capacity):
        self.queue, self.credits, self.capacity = queue, credits, capacity

    def reserve_group(self, count):
        # Reserve all generations together: individually competing waiters can
        # otherwise occupy every credit with several incomplete GRPO groups.
        with self.credits.get_lock():
            if self.credits.value < count:
                return False
            self.credits.value -= count
            return True

    def release(self, count):
        with self.credits.get_lock():
            if self.credits.value + count > self.capacity:
                raise RuntimeError("Rollout credit released more than once")
            self.credits.value += count

    def __getattr__(self, name):
        queue = self.__dict__.get("queue")
        if queue is None:
            raise AttributeError(name)
        return getattr(queue, name)


class AtomicHarnessLoop(_HarnessRolloutLoop):
    async def _reserve_group(self):
        while not self.rollout_buffer.reserve_group(self.num_generations):
            if self._stop_event.is_set():
                return False
            await asyncio.sleep(0.05)
        return True

    async def _generate_one(self, prompt, tool_dict, tools, group_id=0):
        credits = getattr(self.rollout_buffer, "credits", None)
        if credits is None:
            return await super()._generate_one(prompt, tool_dict, tools, group_id)
        if not hasattr(self, "_group_reservations"):
            self._group_reservations = {}
        if group_id not in self._group_reservations:
            self._group_reservations[group_id] = asyncio.create_task(
                self._reserve_group()
            )
        if not await self._group_reservations[group_id]:
            return self._EMPTY_ROLLOUT
        # Native groups can be created before credit becomes available. Record
        # actual dispatch, never relabel an older sampled policy as a newer one.
        version = self.model_version
        try:
            result = await super()._generate_one(prompt, tool_dict, tools, group_id)
        except BaseException:
            self.rollout_buffer.release(1)
            raise
        sequences = result[2]
        if sequences:
            if not hasattr(self, "_dispatch_versions"):
                self._dispatch_versions = {}
            self._dispatch_versions[sequences[0].rollout_id] = version
        else:
            self.rollout_buffer.release(1)
        return result

    async def _score_group(self, group):
        getattr(self, "_group_reservations", {}).pop(group.group_id, None)
        versions = getattr(self, "_dispatch_versions", {})
        actual = [
            versions.pop(sequences[0].rollout_id)
            for sequences in group.completions_sequences
            if sequences and sequences[0].rollout_id in versions
        ]
        if actual:
            if len(actual) != sum(
                bool(sequences) for sequences in group.completions_sequences
            ):
                raise RuntimeError("A scored rollout has no dispatch policy version")
            group = replace(group, model_version=min(actual))
        rows = await super()._score_group(group)
        bundles, offset = [], 0
        for sequences in group.completions_sequences:
            count = len(sequences)
            if count:
                selected = rows[offset : offset + count]
                assert len(selected) == count
                rollout_id = sequences[0].rollout_id
                assert all(seq.rollout_id == rollout_id for seq in sequences)
                bundles.append(RolloutBundle(selected, rollout_id))
            offset += count
        assert offset == len(rows)
        return bundles


class AtomicHarnessWorker(HarnessRolloutWorker):
    _loop_cls = AtomicHarnessLoop

    def __init__(self, *, max_outstanding_rollouts=0, **kwargs):
        super().__init__(**kwargs)
        self.max_outstanding_rollouts = max_outstanding_rollouts
        self.num_generations = kwargs["num_generations"]
        if max_outstanding_rollouts:
            if max_outstanding_rollouts < 2 * self.num_generations:
                raise ValueError("Outstanding budget must allow two complete groups")
            self.rollout_buffer = CreditQueue(
                self.rollout_buffer,
                self._mp_ctx.Value("i", max_outstanding_rollouts),
                max_outstanding_rollouts,
            )

    def release_rollouts(self, count):
        credits = getattr(self.rollout_buffer, "credits", None)
        if credits is not None:
            self.rollout_buffer.release(count)


class AtomicRolloutDataset(IterableDataset):
    def __init__(
        self,
        worker,
        metrics,
        target_tokens,
        max_row_tokens,
        max_staleness,
        heartbeat_seconds,
        max_rollouts_per_unit=None,
        rejection_path=None,
        group_offset=0,
    ):
        self.worker, self.metrics = worker, metrics
        self.target_tokens, self.max_row_tokens = target_tokens, max_row_tokens
        self.max_staleness, self.heartbeat_seconds = max_staleness, heartbeat_seconds
        self.wait_s = 0.0
        self.pending = None
        self.max_rollouts_per_unit = max_rollouts_per_unit
        self.rejection_path = rejection_path
        self.group_offset = group_offset

    def _next_bundle(self):
        if self.pending is not None:
            bundle, self.pending = self.pending, None
            return bundle
        started = time.monotonic()
        while True:
            try:
                bundle = self.worker.rollout_buffer.get(timeout=5)
                self.wait_s += time.monotonic() - started
                return bundle
            except queue.Empty:
                self.worker.check_health(self.heartbeat_seconds)

    def __iter__(self):
        while True:
            bundles, tokens = [], 0
            while tokens < self.target_tokens and (
                self.max_rollouts_per_unit is None
                or len(bundles) < self.max_rollouts_per_unit
            ):
                bundle = self._next_bundle()
                channel = self.worker.rollout_buffer
                if isinstance(channel, CreditQueue):
                    self.metrics["admission/outstanding_rollouts_max"].append(
                        float(channel.capacity - channel.credits.value)
                    )
                if not isinstance(bundle, RolloutBundle) or not bundle.rows:
                    raise RuntimeError(
                        "Atomic trainer requires nonempty rollout bundles"
                    )
                staleness = self.worker.model_version - bundle.model_version
                if staleness > self.max_staleness:
                    self.metrics["admission/stale_rollouts_dropped_total"].append(1.0)
                    self.metrics["sample/dropped_stale_total"].append(
                        float(len(bundle.rows))
                    )
                    if self.rejection_path:
                        self.rejection_path.parent.mkdir(parents=True, exist_ok=True)
                        with self.rejection_path.open("a") as stream:
                            stream.write(
                                json.dumps(
                                    {
                                        "rollout_id": bundle.rollout_id,
                                        "group_id": bundle.group_id + self.group_offset,
                                        "rows": len(bundle.rows),
                                        "model_version": bundle.model_version,
                                        "current_model_version": self.worker.model_version,
                                        "reason": "whole_rollout_staleness_limit",
                                    }
                                )
                                + "\n"
                            )
                    if hasattr(self.worker, "release_rollouts"):
                        self.worker.release_rollouts(1)
                    continue
                if any(len(row.input_ids) > self.max_row_tokens for row in bundle.rows):
                    raise RuntimeError(
                        "Captured row exceeds the tested context limit; refusing to discard it"
                    )
                if bundles and tokens + bundle.forwarded_tokens > self.target_tokens:
                    self.pending = bundle
                    break
                # Recheck pending bundles above when they are actually admitted next time.
                bundles.append(bundle)
                tokens += bundle.forwarded_tokens
                self.metrics["sample/staleness_mean"].append(float(staleness))
                self.metrics["sample/staleness_max"].append(float(staleness))
                self.metrics["sample/rollout_queue_size"].append(
                    float(self.worker.rollout_buffer.qsize())
                )
                if bundle.enqueued_at is not None:
                    self.metrics["sample/time_in_queue_s"].append(
                        time.time() - bundle.enqueued_at
                    )
            yield {"rollouts": bundles}


def pack_rows(bundles, target_tokens, max_row_tokens):
    """Keep exact sequences intact; a long sequence gets one dedicated forward."""
    packed, tokens = [], 0
    for bundle in bundles:
        for row in bundle.rows:
            size = len(row.input_ids)
            if size > max_row_tokens:
                raise ValueError("Row exceeds maximum context")
            if packed and tokens + size > target_tokens:
                yield packed
                packed, tokens = [], 0
            packed.append(
                {
                    key: getattr(row, key)
                    for key in (
                        "input_ids",
                        "completion_mask",
                        "old_log_probs",
                        "advantage",
                        "group_id",
                        "metrics",
                    )
                }
            )
            tokens += size
            if tokens >= target_tokens:
                yield packed
                packed, tokens = [], 0
    if packed:
        yield packed


def identity(value):
    return value


class AtomicRolloutTrainer(AsyncGRPOTrainer):
    def __init__(self, *args, max_row_tokens=131072, admission_dir=None, **kwargs):
        self.max_row_tokens = max_row_tokens
        self.admission_dir = Path(admission_dir) if admission_dir else None
        self._atomic_finished = []
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1 or self.aux_loss_enabled:
            raise ValueError(
                "Atomic recipe is validated only for a single-GPU dense trainer"
            )
        self.add_callback(AtomicAdmissionCallback(self))

    def get_train_dataloader(self):
        outstanding = getattr(self.rollout_worker, "max_outstanding_rollouts", 0)
        max_per_unit = None
        if outstanding:
            # Leave capacity for an entire new GRPO group while accumulating.
            # Otherwise a short partial batch can hold all credits and wait
            # forever for a group whose final generation cannot start.
            max_per_unit = (
                outstanding - self.rollout_worker.num_generations
            ) // self.args.gradient_accumulation_steps
            if max_per_unit < 1:
                raise ValueError("Outstanding budget cannot fill an optimizer update")
        dataset = AtomicRolloutDataset(
            self.rollout_worker,
            self._metrics["train"],
            self.args.token_budget,
            self.max_row_tokens,
            self.args.max_staleness,
            self.args.heartbeat_stale_after_s,
            max_rollouts_per_unit=max_per_unit,
            rejection_path=(
                self.admission_dir / "rejected_rollouts.jsonl"
                if self.admission_dir
                else None
            ),
            group_offset=self._groups_before_resume,
        )
        self._rollout_dataset = dataset
        self._atomic_collator = DataCollatorForRollout(
            self.processing_class.pad_token_id,
            groups_trained=self._trained_groups,
            metrics=self._metrics["train"],
            token_budget=self.max_row_tokens,
        )
        # There is exactly one rank. Dispatcher prefetch/slicing would split the
        # nested rollout container; inner tensors move to the GPU in training_step.
        return DataLoader(dataset, batch_size=None, num_workers=0, collate_fn=identity)

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches = [next(epoch_iterator) for _ in range(num_batches)]
        count = sum(
            sum(row.completion_mask[1:])
            for batch in batches
            for bundle in batch["rollouts"]
            for row in bundle.rows
        )
        if count <= 0:
            raise RuntimeError("Optimizer update has no supervised tokens")
        for batch in batches:
            batch["normalization_tokens"] = count
        return batches, None

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        # Native loss divides by this forward's token count and by GAS. Undo
        # those factors and normalize once across all exact tokens in the update.
        # Keep native counters untouched: they report actual forwarded/trained tokens.
        return loss * (
            self.current_gradient_accumulation_steps
            * inputs["global_n_tokens"][0]
            / self._atomic_normalization_tokens
        )

    def training_step(self, model, inputs, num_items_in_batch):
        self._atomic_normalization_tokens = inputs["normalization_tokens"]
        total_loss = torch.zeros((), device=self.args.device)
        rows_done = tokens_done = 0
        for rows in pack_rows(
            inputs["rollouts"], self.args.token_budget, self.max_row_tokens
        ):
            tensors = self._atomic_collator([[rows]])
            tokens_done += int(tensors["global_n_tokens"][0])
            rows_done += len(rows)
            total_loss += super().training_step(model, tensors, None)
        expected_rows = sum(len(bundle.rows) for bundle in inputs["rollouts"])
        expected_tokens = sum(
            sum(row.completion_mask[1:])
            for bundle in inputs["rollouts"]
            for row in bundle.rows
        )
        assert rows_done == expected_rows and tokens_done == expected_tokens
        self._atomic_finished.extend(
            {
                "rollout_id": bundle.rollout_id,
                "local_group_id": bundle.group_id,
                "group_id": bundle.group_id + self._groups_before_resume,
                "model_version": bundle.model_version,
                "rows": len(bundle.rows),
                "supervised_tokens": sum(
                    sum(row.completion_mask[1:]) for row in bundle.rows
                ),
            }
            for bundle in inputs["rollouts"]
        )
        return total_loss

    def floating_point_ops(self, inputs):
        # Native per-forward token/timing metrics cover the nested batches.
        return 0


class AtomicAdmissionCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        rows = self.trainer._atomic_finished
        if self.trainer.admission_dir:
            self.trainer.admission_dir.mkdir(parents=True, exist_ok=True)
            with (self.trainer.admission_dir / "optimizer_rollouts.jsonl").open(
                "a"
            ) as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": state.global_step,
                            "rollouts": rows,
                            "all_admitted_rows_consumed": True,
                            "normalization": "update_supervised_token_mean",
                        }
                    )
                    + "\n"
                )
        self.trainer._atomic_finished = []
        if hasattr(self.trainer.rollout_worker, "release_rollouts"):
            self.trainer.rollout_worker.release_rollouts(len(rows))
