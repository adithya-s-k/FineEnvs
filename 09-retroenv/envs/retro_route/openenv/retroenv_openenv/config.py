"""Server settings from environment variables, and the read-only data every session shares.

Everything is read from one release directory (``tasks-private/``, ``stocks/``,
``library/`` and ``manifest.json``; see ``retroenv.benchmark``). The hidden
routes stay inside the server process; nothing here is bundled into the Docker
image (``prepare.py`` fetches the release at startup).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from retroenv.benchmark import Benchmark, load_benchmark
from retroenv.environment import RetroRouteSession
from retroenv.retrieval import cached_stock_index
from retroenv.store import TaskStore
from retroenv.tools import tool_names
from retroenv.verifier import prepare_stock

from .client import ENV_NAME  # noqa: F401  (defined with the client, which must stay light)


def _resolve_root() -> Path:
    benchmark = os.getenv("RETROENV_BENCHMARK_DIR")
    if benchmark:
        return Path(benchmark)
    prepared = Path(os.getenv("RETROENV_PREPARED_DIR", "prepared"))
    if (prepared / "tasks-private").is_dir():
        return prepared
    raise RuntimeError(
        "No release configured. Set RETROENV_BENCHMARK_DIR to a release directory "
        "(tasks-private/, stocks/, library/), or run prepare.py first."
    )


@dataclass(frozen=True)
class Settings:
    benchmark_dir: Path
    default_split: str = "train"
    max_tool_calls: int = 32
    toolset: str = "full"
    pubchem_cache: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        cache = os.getenv("RETROENV_PUBCHEM_CACHE") or None
        settings = cls(
            benchmark_dir=_resolve_root(),
            default_split=os.getenv("RETROENV_DEFAULT_SPLIT", "train"),
            max_tool_calls=int(os.getenv("RETROENV_MAX_TOOL_CALLS", "32")),
            toolset=os.getenv("RETROENV_TOOLSET", "full"),
            pubchem_cache=Path(cache) if cache else None,
        )
        tool_names(settings.toolset)  # fail fast on an unknown toolset
        return settings


@dataclass
class Resources:
    settings: Settings
    benchmark: Benchmark
    pubchem_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def store(self) -> TaskStore:  # or SqliteTaskStore / EvalsetStore: the same interface
        return self.benchmark.store

    @classmethod
    def load(cls, settings: Settings) -> "Resources":
        benchmark = load_benchmark(str(settings.benchmark_dir.resolve()))
        splits = benchmark.store.splits()
        if not splits:
            raise RuntimeError(f"no task splits found in {settings.benchmark_dir}")
        if settings.default_split not in splits:
            raise RuntimeError(f"default split {settings.default_split!r} is not among {splits}")
        # Index every stock and canonicalize it for the verifier once, so no episode pays for it.
        for stock_id in benchmark.store.stock_ids():
            stock = benchmark.store.stock(stock_id)
            prepare_stock(cached_stock_index(stock).known)
        return cls(settings=settings, benchmark=benchmark, pubchem_cache=_load_pubchem_cache(settings.pubchem_cache))

    def session(self) -> RetroRouteSession:
        return self.benchmark.session(
            max_tool_calls=self.settings.max_tool_calls,
            pubchem_cache=self.pubchem_cache,
            toolset=self.settings.toolset,
        )


def _load_pubchem_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"PubChem cache must be a JSON object: {path}")
    return {str(key).casefold(): row for key, row in value.items() if isinstance(row, dict)}


@lru_cache(maxsize=1)
def shared_resources() -> Resources:
    """Loaded once per process, on the first environment instance."""
    return Resources.load(Settings.from_env())
