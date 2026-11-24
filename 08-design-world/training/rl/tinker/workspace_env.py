"""Load TINKER_API_KEY from a `.env` up the tree (never printed, never overrides a key already set)."""

from __future__ import annotations

import os
from pathlib import Path

KEYS = ("TINKER_API_KEY",)


def _candidates() -> list[Path]:
    """Any `.env` in the working directory, then in this directory or its parents."""
    here = Path(__file__).resolve().parent
    return [Path.cwd() / ".env", *(p / ".env" for p in (here, *here.parents))]


def load(keys: tuple[str, ...] = KEYS) -> bool:
    """Set the given keys from the first `.env` that has them. Returns whether TINKER_API_KEY is now set."""
    for path in _candidates():
        if all(os.environ.get(k) for k in keys):
            break
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            name, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and name in keys and not os.environ.get(name):
                os.environ[name] = value.strip().strip("'\"")
    return bool(os.environ.get("TINKER_API_KEY"))
