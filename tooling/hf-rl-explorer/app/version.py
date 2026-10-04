"""What produced a rollout, recorded on every run so results can be compared and reproduced.

  app       this explorer's version (pyproject) and a hash of the source that actually shipped (app/ + web/js),
            so two deploys of the same version with different code are told apart
  dataset   the Hub dataset and the revision the task was read from
"""

from __future__ import annotations

import hashlib
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def app_version() -> str:
    try:
        return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    except (OSError, KeyError, ValueError):
        return "0"


@lru_cache(maxsize=1)
def source_hash() -> str:
    h = hashlib.sha256()
    for base in (ROOT / "app", ROOT / "web"):
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".js", ".txt", ".json", ".css", ".html") and "__pycache__" not in p.parts:
                h.update(p.relative_to(ROOT).as_posix().encode())
                h.update(p.read_bytes())
    return h.hexdigest()[:12]


def provenance(dataset: str, revision: str | None) -> dict:
    return {"app": {"version": app_version(), "source": source_hash()},
            "dataset": {"repo": dataset, "revision": revision}}
