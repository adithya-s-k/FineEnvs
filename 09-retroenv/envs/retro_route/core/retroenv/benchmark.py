"""Load one built release directory: tasks, stock, reaction library and precedent corpus.

Layout (the Hugging Face dataset repo root)::

    manifest.json                 split rules and build provenance
    tasks-private/<split>.jsonl   tasks with hidden known routes
    stocks/<stock_id>.smi         building blocks
    library/reactions.jsonl.gz    corpus reactions (stereo-free keys, class, patents, visibility)
    library/templates.json        retro-template counts from every corpus reaction
    library/templates-visible.json  retro-template counts from train-visible reactions only
    library/reagents.json         common reagents tolerated as extra precursors

A prepared directory may add two things beside the release files (see ``serving``):

    serving/serving.sqlite        the serving index; when present, tasks, stock and
                                  precedents load from it instead of being recomputed
    evalsets/<name>.json          frozen evaluation sets, each served as a split
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .environment import RetroRouteSession
from .evalsets import EvalsetStore, load_evalset
from .leakage import LeakageRules
from .reactions import DEFAULT_MIN_COUNT, ReactionLibrary, iter_reactions
from .retrieval import PrecedentIndex
from .serving import INDEX_FILE, ServingIndex
from .store import TaskStore


@dataclass
class Benchmark:
    root: Path
    store: Any  # TaskStore, SqliteTaskStore, or either wrapped in EvalsetStore
    manifest: dict[str, Any]
    rules: LeakageRules
    library: ReactionLibrary
    tool_library: ReactionLibrary
    precedents: PrecedentIndex

    @classmethod
    def load(cls, root: str | Path) -> "Benchmark":
        root = Path(root)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        rules = LeakageRules.from_manifest(manifest)
        library_dir = root / "library"
        min_count = int(manifest.get("library", {}).get("min_template_count", DEFAULT_MIN_COUNT))
        rows = list(iter_reactions(library_dir / "reactions.jsonl.gz"))
        reagents = json.loads((library_dir / "reagents.json").read_text())
        library = ReactionLibrary(
            ((row["product"], tuple(row["reactants"])) for row in rows),
            json.loads((library_dir / "templates.json").read_text()),
            reagents,
            min_count=min_count,
        )
        tool_library = ReactionLibrary(
            ((row["product"], tuple(row["reactants"])) for row in rows if row["visible"]),
            json.loads((library_dir / "templates-visible.json").read_text()),
            reagents,
            min_count=min_count,
        )
        index_path = root / "serving" / INDEX_FILE
        if index_path.exists():
            index = ServingIndex(index_path)
            store: Any = index.task_store()
            precedents = index.precedents(rules, library_dir / "reactions.jsonl.gz")
        else:
            store = TaskStore(root / "tasks-private", root / "stocks")
            precedents = PrecedentIndex(rows, rules)
        evalsets = [load_evalset(path) for path in sorted((root / "evalsets").glob("*.json"))]
        if evalsets:
            store = EvalsetStore(store, evalsets)
        return cls(
            root=root,
            store=store,
            manifest=manifest,
            rules=rules,
            library=library,
            tool_library=tool_library,
            precedents=precedents,
        )

    def session(self, **kwargs: Any) -> RetroRouteSession:
        return RetroRouteSession(
            library=self.library, tool_library=self.tool_library, precedent_index=self.precedents, **kwargs
        )


@lru_cache(maxsize=4)
def load_benchmark(root: str) -> Benchmark:
    """Process-wide cache; a release directory is immutable once built."""
    return Benchmark.load(root)
