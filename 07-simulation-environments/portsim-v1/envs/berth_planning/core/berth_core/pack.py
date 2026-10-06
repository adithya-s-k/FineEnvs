"""Task packs: directories of tasks (tasks.jsonl or tasks.jsonl.gz, with their reference plans) plus a manifest.

The server serves several packs at once; splits come from each task's `split` field. By default that is the dock-v1
eval and train packs (the hard tasks); the first pack, berth-v1, stays available by setting BERTH_TASKS_DIR.

    BERTH_TASKS_DIR=/path/a:/path/b     colon-separated pack directories
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

from .model import Task

TASKS_ROOT = Path(__file__).resolve().parents[2] / "tasks"
DEFAULT_PACKS = ["dock-v1-eval", "dock-v1-train"]
LEGACY_PACK = "berth-v1"


def default_roots() -> list[Path]:
    env = os.environ.get("BERTH_TASKS_DIR", "").strip()
    if env:
        return [Path(p) for p in env.split(":") if p]
    roots = [TASKS_ROOT / p for p in DEFAULT_PACKS if _tasks_file(TASKS_ROOT / p)]
    return roots or [TASKS_ROOT / LEGACY_PACK]


def _tasks_file(root: Path) -> Path | None:
    for name in ("tasks.jsonl", "tasks.jsonl.gz"):
        if (root / name).is_file():
            return root / name
    return None


def _read(root: Path) -> tuple[list[Task], dict]:
    path = _tasks_file(root)
    if path is None:
        raise FileNotFoundError(f"no tasks.jsonl(.gz) in {root}")
    raw = path.read_bytes()
    body = gzip.decompress(raw) if path.suffix == ".gz" else raw
    manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").is_file() else {}
    digest = hashlib.sha256(body).hexdigest()
    if manifest.get("sha256") and manifest["sha256"] != digest:
        raise ValueError(f"{path} does not match its manifest.json (sha256 {digest[:12]} != {manifest['sha256'][:12]})")
    return [Task.from_dict(json.loads(line)) for line in body.decode().splitlines() if line.strip()], manifest


class TaskPack:
    def __init__(self, roots: Path | str | list[Path | str] | None = None):
        if roots is None:
            roots = default_roots()
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.roots = [Path(r) for r in roots]
        self.root = self.roots[0]
        self.tasks: list[Task] = []
        self.manifests: dict[str, dict] = {}
        for r in self.roots:
            tasks, manifest = _read(r)
            self.tasks += tasks
            self.manifests[r.name] = manifest
        self.manifest = self.manifests[self.root.name]
        self._by_id = {}
        for t in self.tasks:
            if t.task_id in self._by_id:
                raise ValueError(f"task id {t.task_id} appears in more than one pack")
            self._by_id[t.task_id] = t
        self._by_split: dict[str, list[Task]] = {}
        for t in self.tasks:
            self._by_split.setdefault(t.split, []).append(t)

    def splits(self) -> list[str]:
        return sorted(self._by_split)

    def count(self, split: str) -> int:
        return len(self._by_split.get(split, []))

    def at(self, split: str, index: int) -> Task:
        return self._by_split[split][index]

    def get(self, task_id: str) -> Task:
        return self._by_id[task_id]

    def public(self, task: Task) -> dict:
        """What anyone may see: everything but the reference plans and costs."""
        return task.to_dict(public=True)


_PACK: TaskPack | None = None


def load_pack(root: Path | str | list | None = None) -> TaskPack:
    global _PACK
    if root is not None:
        return TaskPack(root)
    if _PACK is None:
        _PACK = TaskPack()
    return _PACK
