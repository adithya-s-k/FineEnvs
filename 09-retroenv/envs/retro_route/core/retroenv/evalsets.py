"""Frozen evaluation subsets: a fixed list of release task ids, served as a split of its own.

An evaluation set names tasks that already exist in the release and never copies them, so it
scores exactly as those tasks do in their source split. Its ``evalset_id`` hashes the design
and the task list; a file whose id does not match its contents is refused, so a hand edit
cannot silently change what a board measures.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .chemistry import canonicalize_smiles
from .models import RetroTask

EVALSET_VERSION = 1
TIERS = ("easy", "medium", "hard")


def evalset_id(record: dict[str, Any]) -> str:
    body = {key: record[key] for key in ("evalset_version", "name", "seed", "design", "tasks")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# select_tiered rescans its candidates on every pick, so without this each draw costs an
# RDKit canonicalization per candidate per pick. Task ids are unique across the release.
_FIRST_REACTION: dict[str, str] = {}


def first_reaction(task: RetroTask) -> str:
    """The class of the known route's first disconnection, preferring the patent route."""
    if task.task_id in _FIRST_REACTION:
        return _FIRST_REACTION[task.task_id]
    target = canonicalize_smiles(task.target_smiles)
    routes = sorted(task.reference_routes, key=lambda route: route.kind != "patent")
    found = "other"
    for route in routes:
        step = next((s for s in route.steps if canonicalize_smiles(s.product) == target), None)
        if step is not None:
            found = step.reaction_class or "other"
            break
    _FIRST_REACTION[task.task_id] = found
    return found


def _order(seed: str, task_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def select_tiered(tasks: Iterable[RetroTask], quotas: dict[str, int], seed: str) -> list[RetroTask]:
    """Pick ``quotas[tier]`` tasks per difficulty tier, spread over source split and route length.

    Within a tier, candidates form strata by (source split, shortest-route depth), each ordered
    by a seeded hash. Strata are visited round-robin, shallow first, and each visit prefers a
    candidate whose first reaction class the tier has not used yet. The result depends only on
    the candidates and the seed.
    """
    pool = list(tasks)
    chosen: list[RetroTask] = []
    for tier in TIERS:
        quota = quotas.get(tier, 0)
        strata: dict[tuple[int, str], list[RetroTask]] = {}
        for task in pool:
            if task.difficulty.get("tier") == tier:
                key = (int(task.difficulty.get("min_depth") or task.max_depth), task.split)
                strata.setdefault(key, []).append(task)
        for members in strata.values():
            members.sort(key=lambda task: _order(seed, task.task_id))
        available = sum(len(members) for members in strata.values())
        if available < quota:
            raise ValueError(f"tier {tier!r} has {available} candidates, fewer than its quota of {quota}")
        picked: list[RetroTask] = []
        used: set[str] = set()
        while len(picked) < quota:
            for key in sorted(strata):
                if len(picked) == quota:
                    break
                members = strata[key]
                if not members:
                    continue
                fresh = next((task for task in members if first_reaction(task) not in used), members[0])
                members.remove(fresh)
                picked.append(fresh)
                used.add(first_reaction(fresh))
        chosen.extend(picked)
    return chosen


def describe_task(task: RetroTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "source_split": task.split,
        "tier": task.difficulty.get("tier"),
        "min_depth": task.difficulty.get("min_depth"),
        "max_depth": task.max_depth,
        "first_reaction": first_reaction(task),
        "heavy_atoms": task.difficulty.get("heavy_atoms"),
        "target_smiles": task.target_smiles,
    }


def make_evalset(name: str, tasks: Iterable[RetroTask], *, seed: str, design: dict[str, Any]) -> dict[str, Any]:
    record = {
        "evalset_version": EVALSET_VERSION,
        "name": name,
        "seed": seed,
        "design": design,
        "tasks": [describe_task(task) for task in tasks],
    }
    record["size"] = len(record["tasks"])
    record["evalset_id"] = evalset_id(record)
    return record


def load_evalset(path: str | Path) -> dict[str, Any]:
    record = json.loads(Path(path).read_text())
    if record.get("evalset_version") != EVALSET_VERSION:
        raise ValueError(f"{path}: evaluation set version {record.get('evalset_version')}, expected {EVALSET_VERSION}")
    if record.get("evalset_id") != evalset_id(record):
        raise ValueError(f"{path}: evalset_id does not match its contents; regenerate it instead of editing it")
    ids = [task["task_id"] for task in record["tasks"]]
    if len(ids) != len(set(ids)) or len(ids) != record.get("size"):
        raise ValueError(f"{path}: task ids must be unique and match size")
    return record


class EvalsetStore:
    """A task store plus evaluation sets, each served as a split of the same name."""

    def __init__(self, base: Any, evalsets: Iterable[dict[str, Any]]):
        self.base = base
        self._evalsets: dict[str, tuple[RetroTask, ...]] = {}
        for record in evalsets:
            name = record["name"]
            if name in base.splits() or name in self._evalsets:
                raise ValueError(f"evaluation set {name!r} collides with an existing split")
            self._evalsets[name] = tuple(self._resolve(record, entry) for entry in record["tasks"])

    def _resolve(self, record: dict[str, Any], entry: dict[str, Any]) -> RetroTask:
        task_id, split = entry["task_id"], entry.get("source_split")
        position = self.base.position(split, task_id) if split in self.base.splits() else None
        task = self.base.task(split, position) if position is not None else self.base.find(task_id)
        if task is None:
            raise ValueError(f"evaluation set {record['name']!r} names {task_id}, which this release lacks")
        return task

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def evalset_names(self) -> list[str]:
        return sorted(self._evalsets)

    def splits(self) -> list[str]:
        return sorted([*self.base.splits(), *self._evalsets])

    def tasks(self, split: str):
        if split in self._evalsets:
            return self._evalsets[split]
        return self.base.tasks(split)

    def task(self, split: str, index: int) -> RetroTask:
        return self.tasks(split)[index]

    def public_task(self, split: str, index: int) -> dict:
        return self.task(split, index).to_dict(include_hidden=False)

    def position(self, split: str, task_id: str) -> int | None:
        if split in self._evalsets:
            return next((i for i, task in enumerate(self._evalsets[split]) if task.task_id == task_id), None)
        return self.base.position(split, task_id)

    def iter_all(self) -> Iterable[RetroTask]:
        # Evaluation sets only re-list release tasks, so the release alone is every task once.
        return self.base.iter_all()
