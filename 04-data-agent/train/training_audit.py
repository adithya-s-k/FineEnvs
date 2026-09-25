"""Local capture artifacts and bounded coverage checks for the multiharness recipe."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from transformers import TrainerCallback


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class AuditedSession:
    def __init__(self, session, path, metadata):
        self.session = session
        self.path = path
        self.metadata = metadata

    def __getattr__(self, name):
        return getattr(self.session, name)

    def wait_for_completion(self, *args, **kwargs):
        started = time.time()
        try:
            return self.session.wait_for_completion(*args, **kwargs)
        finally:
            result = self.session.result
            write_json(self.path, {
                **self.metadata, "started_at": started, "finished_at": time.time(),
                "result": result.model_dump(mode="json") if result is not None else None,
            })


class AuditedFactory:
    def __init__(self, factory, directory):
        self.factory = factory
        self.directory = str(directory)

    def __getattr__(self, name):
        # During unpickling the wrapped factory has not been assigned yet.
        factory = self.__dict__.get("factory")
        if factory is None:
            raise AttributeError(name)
        return getattr(factory, name)

    def create(self, task, seed=None, episode_id=None):
        session = self.factory.create(task, seed=seed, episode_id=episode_id)
        return AuditedSession(
            session, Path(self.directory) / "rollouts" / f"{episode_id}.json",
            {"group_id": (seed or 0) + getattr(self.factory, 'group_offset', 0),
             "local_group_id": seed, "episode_id": episode_id,
             "harness": self.factory.harness_for(seed), "task_index": session._task_index},
        )


class PairCoverageCallback(TrainerCallback):
    """Use TRL's existing collated group counter; require coverage to persist a full update.

    The extra update avoids stopping on the dataloader's prefetched batch. Coverage counts a
    group once any of its rows enters training, and does not claim every forked row was consumed.
    """

    def __init__(self, trainer, n_tasks, harnesses, min_steps, directory, *, all_pairs=False, schedule=None,
                 group_offset=0):
        self.trainer = trainer
        self.n_rows = n_tasks
        self.all_pairs = all_pairs
        self.schedule = schedule
        self.group_offset = group_offset
        if all_pairs and n_tasks % len(harnesses):
            raise ValueError("Cartesian task/harness schedule has an incomplete task")
        self.n_tasks = n_tasks // len(harnesses) if all_pairs else n_tasks
        if schedule is not None:
            self.n_tasks = schedule['task_count']
        self.harnesses = harnesses
        self.min_steps = min_steps
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.previous_pairs = set()

    def on_step_end(self, args, state, control, **kwargs):
        group_ids = sorted(g + self.group_offset for g in self.trainer._trained_groups)
        pairs = {
            ((g % self.n_rows) // len(self.harnesses) if self.all_pairs else g % self.n_tasks,
             g % len(self.harnesses)) for g in group_ids
        }
        if self.schedule is not None:
            groups = self.schedule['groups']
            pairs = {(groups[g % len(groups)]['task_row'],
                      self.harnesses.index(groups[g % len(groups)]['harness'])) for g in group_ids}
        stable_pairs = pairs & self.previous_pairs
        complete = len(stable_pairs) == self.n_tasks * len(self.harnesses)
        report = {
            "optimizer_steps": state.global_step, "target_min_steps": self.min_steps,
            "resumed_group_offset": self.group_offset,
            "hard_max_steps": args.max_steps, "collated_group_ids": group_ids,
            "covered_pairs": [{"task_row": t, "harness": self.harnesses[h]}
                              for t, h in sorted(stable_pairs)],
            "pair_coverage_complete": complete,
            "unique_tasks_covered": len({t for t, _ in stable_pairs}),
            "target_unique_tasks": self.n_tasks,
            "harness_pair_counts": {name: sum(h == i for _, h in stable_pairs)
                                    for i, name in enumerate(self.harnesses)},
            "coverage_semantics": "at least one row per pair; stable over two optimizer boundaries",
        }
        write_json(self.directory / "coverage.json", report)
        print(f"PAIR_COVERAGE step={state.global_step} pairs={len(stable_pairs)}/"
              f"{self.n_tasks * len(self.harnesses)}", flush=True)
        self.previous_pairs = pairs
        if self.min_steps and state.global_step >= self.min_steps and complete:
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        with (self.directory / "metrics.jsonl").open("a") as output:
            output.write(json.dumps({"step": state.global_step, **(logs or {})}) + "\n")
        for key in ("loss", "grad_norm", "ratio"):
            value = (logs or {}).get(key)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise RuntimeError(f"Non-finite {key}: stopping the training check")


class CheckpointReadyCallback(TrainerCallback):
    """Publish a completion marker after Trainer has finished writing a checkpoint."""

    def __init__(self, base_model, base_revision):
        self.base_model = base_model
        self.base_revision = base_revision

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            from checkpoint_artifacts import mark_saved
            mark_saved(Path(args.output_dir) / f"checkpoint-{state.global_step}",
                       state.global_step, self.base_model, self.base_revision,
                       final=control.should_training_stop or state.global_step >= args.max_steps)


class PeriodicCheckpointCallback(TrainerCallback):
    """Bound recovery loss when optimizer updates are too slow for step-based saves."""

    def __init__(self, seconds):
        if seconds <= 0:
            raise ValueError("Checkpoint interval must be positive")
        self.seconds = seconds
        self.last_saved = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_saved = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self.last_saved is not None and time.monotonic() - self.last_saved >= self.seconds:
            control.should_save = True

    def on_save(self, args, state, control, **kwargs):
        # A regular step-based checkpoint also resets the recovery interval.
        self.last_saved = time.monotonic()


class WallTimeCallback(TrainerCallback):
    """Save and stop at an optimizer boundary before the allocation expires."""

    def __init__(self, seconds):
        if seconds <= 0:
            raise ValueError("Training wall-time budget must be positive")
        self.seconds = seconds
        self.started = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        requested = (Path(args.output_dir).parent / 'STOP_AFTER_STEP').exists()
        if requested or (self.started is not None and time.monotonic() - self.started >= self.seconds):
            control.should_save = True
            control.should_training_stop = True
