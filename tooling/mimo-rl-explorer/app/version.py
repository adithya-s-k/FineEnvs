"""What produced a rollout, recorded on every run so results can be compared and reproduced.

  app       this explorer's version (pyproject) and a hash of the source that actually shipped (app/ + web/js),
            so two deploys of the same version with different code are told apart
  harness   OpenCode, pinned
  reference the XiaomiMiMo/verl and mimoagent commits the graders and configuration were taken from
  dataset   the XiaomiMiMo/MiMo-V2.6-RL-oss revision the tasks were read from
"""

from __future__ import annotations

import hashlib
import tomllib
from functools import lru_cache
from pathlib import Path

from . import config

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {"XiaomiMiMo/verl": "a2ad9f6", "XiaomiMiMo/mimoagent": "467f0a1"}


@lru_cache(maxsize=1)
def app_version() -> str:
    try:
        return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    except (OSError, KeyError, ValueError):
        return "0"


@lru_cache(maxsize=1)
def source_hash() -> str:
    h = hashlib.sha256()
    for base in (ROOT / "app", ROOT / "web" / "js"):
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".js", ".txt", ".json") and "__pycache__" not in p.parts:
                h.update(p.relative_to(ROOT).as_posix().encode())
                h.update(p.read_bytes())
    return h.hexdigest()[:12]


@lru_cache(maxsize=1)
def dataset_revision() -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(config.DATASET).sha
    except Exception:  # offline: the rollout still records everything else
        return None


def provenance() -> dict:
    from .runner import opencode

    return {"app": {"version": app_version(), "source": source_hash()},
            "harness": {"name": "opencode", "version": opencode.VERSION},
            "reference": dict(REFERENCES),
            "dataset": {"repo": config.DATASET, "revision": dataset_revision()}}
