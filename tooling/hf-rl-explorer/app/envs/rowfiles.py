"""Which of a dataset's files hold one split, for the commands that run its rows with their own framework (verl and
SkyRL read local parquet or JSON Lines; NeMo Gym and Verifiers download by split). Read from the dataset's card and
file listing (direct.data_files, cached an hour), never waiting long: a listing still coming answers [] this time
and is there the next.
"""

from __future__ import annotations

import posixpath
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

_pool = ThreadPoolExecutor(2, thread_name_prefix="rowfiles")
_pending: dict[tuple[str, str], Any] = {}
WAIT = 3.0


def files_of(spec: str, meta: dict[str, Any], config: str, split: str) -> list[str]:
    """The repository files behind `config`/`split`, [] when they aren't known (yet): public datasets only."""
    if meta.get("restricted") or not meta.get("sha"):
        return []
    from . import direct

    key = (spec, str(meta.get("sha")))
    fut = _pending.get(key)
    if fut is None or (fut.done() and fut.exception() is not None):
        fut = _pending[key] = _pool.submit(direct.data_files, spec, meta, None)
        if len(_pending) > 500:
            for k in list(_pending)[:250]:
                _pending.pop(k, None)
    try:
        configs = fut.result(timeout=WAIT)
    except (FutureTimeout, Exception):  # noqa: BLE001 - slow or failed: the command falls back to the whole dataset
        return []
    return list((configs.get(config) or {}).get(split) or [])


def local_dir(spec: str) -> str:
    return "data/" + spec.split("/")[-1]


def download(spec: str, files: list[str]) -> str:
    """An `hf download` of these files (named one by one when there are a few, by their folder when there are many,
    the whole dataset when they aren't known)."""
    if files and len(files) <= 6:
        what = " " + " ".join(f'"{f}"' if " " in f else f for f in files)
    elif files and len({posixpath.dirname(f) for f in files}) == 1 and posixpath.dirname(files[0]):
        what = f' --include "{posixpath.dirname(files[0])}/*"'
    else:
        what = ""
    return f"hf download {spec}{what} --repo-type dataset --local-dir {local_dir(spec)}"


def local_paths(spec: str, files: list[str], fallback: str) -> list[str]:
    """Where those files land: each of them (up to 40), else their folder's files by extension, else `fallback` (a
    glob) when they aren't known."""
    if not files:
        return [f"{local_dir(spec)}/{fallback}"]
    if len(files) <= 40:
        return [f"{local_dir(spec)}/{f}" for f in files]
    d, ext = posixpath.dirname(files[0]), posixpath.splitext(files[0])[1]
    return [f"{local_dir(spec)}/{d + '/' if d else ''}*{ext}"]
