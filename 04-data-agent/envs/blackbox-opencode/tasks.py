# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task discovery for the data-agent suite, served through OpenEnv's Task API.

WHY EVERY CACHE HERE IS MODULE LEVEL
`HTTPEnvServer` constructs a FRESH environment for every task request and closes it in a `finally`
(`core/env_server/http_server.py`). Anything cached on `self` is therefore rebuilt on each `/task`
call, and a dataset download would happen per request. The caches below live at module scope, which
is the same discipline `openenv.harbor.tasks` follows for the same reason.

WHY DIFFICULTY IS A SPLIT AND NOT A FILTER ARGUMENT
A task's INDEX is its identity everywhere downstream: checkpoints, eval subsets and reproduction
manifests all record it. A filter applied at listing time shifts every index after it, so "task 12"
silently means different things to two callers. `train:medium` is therefore its own split with its
own index space, and no method takes a difficulty keyword.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .task import DataAgentTask, instruction_id, _tolerance


logger = logging.getLogger(__name__)

DATASET = "HuggingEnvs/data-agent"

# Unbounded listing would materialise thousands of task specs on a throwaway instance, and the Task
# API request times out before it answers. `num_tasks` still reports the true total, which the
# TaskProvider protocol explicitly allows.
PREVIEW_LIMIT = 200

TIERS = ("easy", "medium", "hard")

_LOCK = threading.Lock()
_ROWS: dict[
    str, list[dict[str, Any]]
] = {}  # split name -> usable rows, in dataset order
_TASKS: dict[tuple[str, int], DataAgentTask] = {}  # (split, index) -> parsed task


def _load_split(split: str) -> list[dict[str, Any]]:
    """Rows for one base split, dropping any that cannot produce a gradable rollout.

    A row without a bucket stages no data, and a row without an answer cannot be graded; both would
    score zero for reasons that have nothing to do with the policy. They are dropped at discovery so
    they never reach a sandbox, and the count is logged rather than silently absorbed.
    """
    import os
    frozen = os.environ.get("DATA_AGENT_FROZEN_TASKS_DIR")
    if frozen:
        return _frozen_rows(frozen, split)
    from datasets import load_dataset

    raw = list(load_dataset(DATASET, split=split))
    rows, skipped = [], {"no_bucket": 0, "no_answer": 0}
    for row in raw:
        if not row.get("hf_bucket") or not row.get("bucket_prefix"):
            skipped["no_bucket"] += 1
            continue
        if not str(row.get("answer") or "").strip():
            skipped["no_answer"] += 1
            continue
        rows.append(row)
    if any(skipped.values()):
        logger.warning(
            "data-agent %s: %d of %d rows unusable %s",
            split,
            sum(skipped.values()),
            len(raw),
            skipped,
        )
    return rows


def rows_for(split: str) -> list[dict[str, Any]]:
    """Rows for a split name, which is `<base>` or `<base>:<tier>`. Cached for the process."""
    with _LOCK:
        if split in _ROWS:
            return _ROWS[split]
    base, _, tier = split.partition(":")
    if tier and tier not in TIERS:
        raise ValueError(f"unknown difficulty tier {tier!r}; expected one of {TIERS}")
    rows = _load_split(base)
    if tier:
        rows = [r for r in rows if r.get("difficulty_tier") == tier]
    with _LOCK:
        _ROWS.setdefault(split, rows)
        return _ROWS[split]


def task_at(split: str, index: int) -> DataAgentTask:
    """The parsed task at `index` within `split`.

    Raises:
        IndexError: If out of range. `HTTPEnvServer` turns this into a 400 rather than a 500.
    """
    key = (split, index)
    with _LOCK:
        cached = _TASKS.get(key)
    if cached is not None:
        return cached
    rows = rows_for(split)
    if index < 0 or index >= len(rows):
        raise IndexError(
            f"task index {index} out of range for split {split!r} ({len(rows)} tasks)"
        )
    task = DataAgentTask.from_row(rows[index])
    with _LOCK:
        _TASKS.setdefault(key, task)
    return task


def index_of_instruction(split: str, instruction: str) -> int | None:
    """Recover a task index from its instruction text.

    The loop-owning trainer forwards only the prompt, so this is how a rollout is matched back to its
    task. Collisions are counted and warned about rather than resolved last-write-wins: these
    instructions are template-generated, so two tasks CAN share wording, and a silent overwrite would
    grade a rollout against the wrong gold answer.
    """
    wanted = instruction_id(instruction)
    hits = [
        i
        for i, row in enumerate(rows_for(split))
        if instruction_id(row["instruction"]) == wanted
    ]
    if not hits:
        return None
    if len(hits) > 1:
        logger.warning(
            "instruction maps to %d tasks in %s (indices %s); using the first. Gold answers may differ.",
            len(hits),
            split,
            hits[:5],
        )
    return hits[0]


def prefetch(splits: list[str]) -> None:
    """Warm the caches before the server accepts traffic, so no request pays for a dataset download."""
    for split in splits:
        try:
            rows_for(split)
        except Exception:
            logger.warning("prefetch failed for split %s", split, exc_info=True)


class DataAgentTaskProvider:
    """The five Task API methods, over `HuggingEnvs/data-agent`.

    Holds only the split names. Everything expensive lives in this module's caches, because the
    server rebuilds this object per request.
    """

    def __init__(self, splits: list[str]) -> None:
        self._splits = list(splits)

    def list_splits(self) -> list[dict[str, Any]]:
        """Every split, each with its own index space.

        Returned as dicts rather than bare strings: the server coerces an unrecognised string's
        `type` to `"validation"`, so anything that is not literally train/validation/test would be
        mislabelled.
        """
        out: list[dict[str, Any]] = []
        for configured in self._splits:
            # A configured split may ALREADY name a tier (`train:medium`). Appending tiers to it
            # produces `train:medium:easy`, which is not a split -- it parses as tier `medium:easy`
            # and every one of them comes back as an error entry, burying the real splits in noise.
            base = configured.split(":", 1)[0]
            tiers = () if ":" in configured else TIERS
            for name in (configured, *(f"{base}:{tier}" for tier in tiers)):
                entry: dict[str, Any] = {
                    "name": name,
                    "type": "train" if base == "train" else "validation",
                    "dataset": DATASET,
                }
                try:
                    entry["num_tasks"] = len(rows_for(name))
                except (
                    Exception
                ) as exc:  # a broken split must not hide the working ones
                    entry["error"] = str(exc)
                out.append(entry)
        return out

    def num_tasks(self, split: str) -> int:
        return len(rows_for(split))

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        task = task_at(split, index)
        return _public(task, split, index)

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        total = len(rows_for(split))
        lo = 0 if start is None else max(0, start)
        hi = total if stop is None else min(total, stop)
        return [self.get_task(split, i) for i in range(lo, hi)]

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        """A bounded preview. `num_tasks` remains the authority on how many there are."""
        return self.get_task_range(split, 0, PREVIEW_LIMIT)


def _public(task: DataAgentTask, split: str, index: int) -> dict[str, Any]:
    """What a caller may see. The gold answer is deliberately absent: it is grading material, and a
    task spec travels to whoever asks, including the agent's own side of the wire."""
    return {
        "split": split,
        "index": index,
        "task_id": task.task_id,
        "instruction": task.instruction,
        "question": task.question,
        "difficulty_tier": task.difficulty_tier,
        "difficulty_level": task.difficulty_level,
        "reward_mode": task.reward_mode,
        "files": task.files,
    }


def _frozen_rows(root: str, split: str) -> list[dict[str, Any]]:
    """Read the shared frozen data without invoking Harbor's execution path."""
    from pathlib import Path
    import hashlib
    import json
    import tomllib
    if split not in {"train", "test"}:
        raise ValueError("Frozen task split must be train or test")
    root = Path(root)
    manifest = json.loads((root.parent / f"{split}_manifest.json").read_text())
    rows = []
    for entry in sorted(manifest["tasks"], key=lambda t: t["name"]):
        path = root / split / "tasks" / entry["name"]
        instruction = (path / "instruction.md").read_text()
        expected = entry["file_hashes"][f"tasks/{entry['name']}/instruction.md"]
        if hashlib.sha256(instruction.encode()).hexdigest() != expected:
            raise ValueError(f"Frozen instruction hash mismatch: {entry['name']}")
        if hashlib.sha256((path / "task.toml").read_bytes()).hexdigest() != entry.get("effective_task_toml_sha256", entry["file_hashes"][f"tasks/{entry['name']}/task.toml"]):
            raise ValueError(f"Frozen task configuration mismatch: {entry['name']}")
        spec = tomllib.loads((path / "task.toml").read_text())
        meta, env, verifier = spec["metadata"], spec["environment"]["env"], spec["verifier"]["env"]
        rows.append(dict(task_id=entry["name"], instruction=instruction, answer=meta["gold_answer"],
             question=spec["task"]["description"], reward_mode=verifier["REWARD_MODE"],
             atol=_tolerance(verifier.get("ATOL")), rtol=_tolerance(verifier.get("RTOL")),
             hf_bucket=env["HF_BUCKET"], bucket_prefix=env["BUCKET_PREFIX"],
             difficulty_tier=meta["difficulty_tier"], difficulty_level=meta["difficulty_level"]))
    return rows
