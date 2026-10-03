"""Task and building-block storage with canonicalized, hidden references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .chemistry import canonicalize_smiles
from .models import RetroTask


def load_stock(path: Path) -> frozenset[str]:
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            # Conventional .smi files put an optional identifier after whitespace.
            smiles = text.split(maxsplit=1)[0]
            try:
                values.add(canonicalize_smiles(smiles))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid stock SMILES: {exc}") from exc
    if not values:
        raise ValueError(f"stock file contains no molecules: {path}")
    return frozenset(values)


class TaskStore:
    """Load split JSONL files and stocks without exposing private references."""

    def __init__(self, tasks_dir: str | Path, stocks_dir: str | Path):
        self.tasks_dir = Path(tasks_dir)
        self.stocks_dir = Path(stocks_dir)
        self._tasks: dict[str, tuple[RetroTask, ...]] = {}
        self._stocks: dict[str, frozenset[str]] = {}

    def splits(self) -> list[str]:
        return sorted(path.stem for path in self.tasks_dir.glob("*.jsonl"))

    def tasks(self, split: str) -> tuple[RetroTask, ...]:
        if split not in self._tasks:
            path = self.tasks_dir / f"{split}.jsonl"
            if not path.exists():
                raise KeyError(f"unknown split {split!r}; available: {self.splits()}")
            loaded: list[RetroTask] = []
            seen: set[str] = set()
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        task = RetroTask.from_dict(json.loads(line))
                    except Exception as exc:
                        raise ValueError(f"{path}:{line_number}: {exc}") from exc
                    if task.split != split:
                        raise ValueError(
                            f"{path}:{line_number}: task split {task.split!r} != {split!r}"
                        )
                    if task.task_id in seen:
                        raise ValueError(f"duplicate task_id in {path}: {task.task_id}")
                    seen.add(task.task_id)
                    loaded.append(task)
            self._tasks[split] = tuple(loaded)
        return self._tasks[split]

    def task(self, split: str, index: int) -> RetroTask:
        return self.tasks(split)[index]

    def stock(self, stock_id: str) -> frozenset[str]:
        if stock_id not in self._stocks:
            path = self.stocks_dir / f"{stock_id}.smi"
            if not path.exists():
                raise FileNotFoundError(f"stock {stock_id!r} not found at {path}")
            self._stocks[stock_id] = load_stock(path)
        return self._stocks[stock_id]

    def public_task(self, split: str, index: int) -> dict:
        return self.task(split, index).to_dict(include_references=False)

    def iter_all(self) -> Iterable[RetroTask]:
        for split in self.splits():
            yield from self.tasks(split)

