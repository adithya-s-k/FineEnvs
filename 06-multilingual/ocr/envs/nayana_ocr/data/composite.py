"""One catalog over the Nayana corpus and evaluation-only sources beside it.

The environment, the HTTP routes and every client speak one catalog interface. This
keeps that interface and routes each call to whichever source owns the split or task,
so adding a benchmark changes what a deployment serves without changing how anything
asks for it. Everything a source does not claim - training samplers, block streams,
prefetch, the corpus manifest - goes to the primary catalog untouched.
"""

from pathlib import Path

from .catalog import SPLITS


class _Sized:
    """Stands in for a split's rows where only the length is wanted (the manifest)."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class CompositeCatalog:
    def __init__(self, primary, extras):
        self._primary = primary
        self._extras = list(extras)

    def __getattr__(self, name):
        # Only reached for attributes this class does not define.
        return getattr(self._primary, name)

    def _for_split(self, split):
        return next((e for e in self._extras if e.owns_split(split)), self._primary)

    def _for_task(self, task_id):
        return next((e for e in self._extras if e.owns_task(task_id)), self._primary)

    def splits(self):
        base = (
            self._primary.splits() if hasattr(self._primary, "splits") else list(SPLITS)
        )
        return base + [s for e in self._extras for s in e.splits()]

    @property
    def eval_splits(self):
        merged = dict(getattr(self._primary, "eval_splits", {}))
        for extra in self._extras:
            merged.update((s, _Sized(extra.count(s))) for s in extra.splits())
        return merged

    @property
    def eval_ids(self):
        merged = dict(getattr(self._primary, "eval_ids", {}))
        for extra in self._extras:
            merged.update(extra.eval_ids())
        return merged

    def count(self, split):
        return self._for_split(split).count(split)

    def at(self, split, index):
        return self._for_split(split).at(split, index)

    def task_range(self, split, start=None, stop=None):
        return self._for_split(split).task_range(split, start, stop)

    def group_count(self, split, language, family):
        return self._for_split(split).group_count(split, language, family)

    def group_at(self, split, language, family, index):
        return self._for_split(split).group_at(split, language, family, index)

    def group_position(self, task_id, split, language, family):
        return self._for_split(split).group_position(task_id, split, language, family)

    def get(self, task_id):
        return self._for_task(task_id).get(task_id)

    def public(self, task):
        return self._for_task(task["task_id"]).public(task)

    def materialize(self, task):
        return self._for_task(task["task_id"]).materialize(task)

    def image(self, task):
        return self._for_task(task["task_id"]).image(task)

    def prefetch(self, *, task_ids=(), block_ids=()):
        # The playground prefetches neighbouring task IDs. A benchmark ID handed to the
        # corpus loader would fail to parse, so each source warms only its own tasks.
        mine = [t for t in task_ids if self._for_task(t) is self._primary]
        for extra in self._extras:
            ids = [t for t in task_ids if extra.owns_task(t)]
            if ids and hasattr(extra, "prefetch"):
                extra.prefetch(task_ids=ids)
        if hasattr(self._primary, "prefetch"):
            return self._primary.prefetch(task_ids=mine, block_ids=block_ids)
        return None

    def asset_bytes(self, sha, task_id):
        owner = self._for_task(task_id)
        if owner is not self._primary or hasattr(self._primary, "asset_bytes"):
            return owner.asset_bytes(sha, task_id)
        # A prepared snapshot serves files by hash; answer in the bytes form the route
        # expects, since defining asset_bytes here is what selects that branch.
        path, mime = self._primary.asset(sha)
        return Path(path).read_bytes(), mime

    def close(self):
        for extra in self._extras:
            extra.close()
        if hasattr(self._primary, "close"):
            self._primary.close()
