#!/usr/bin/env python3
"""Make the data a server reads available before it starts, and check every byte of it.

Three sources, tried in this order:

* ``RETROENV_BENCHMARK_DIR``: a local release or prepared directory, used as it is.
* ``RETROENV_CORPUS_MANIFEST``: a pinned serving snapshot in the FineEnvs bucket. This is the
  default (``corpus-manifest.json`` beside this file) when nothing else is set, and what the
  Docker image serves. Each file the manifest names comes from ``RETROENV_BUCKET_ROOT`` when
  that directory holds it (a Space mounts the bucket there), and otherwise over HTTP from the
  public bucket (``RETROENV_BUCKET_ID`` overrides its name). Every file is checked against its
  SHA-256, and files already in ``RETROENV_PREPARED_DIR`` with the right digest are kept, so a
  restart fetches nothing.
* ``RETROENV_TASKS_REPO=org/name[@revision]``: download a release dataset such as
  ``LiteFold/RetroEnv``, check it against its checksums.json, and serve it without an index
  (``RETROENV_TASKS_SUBDIR`` selects a subdirectory of the repo).

The last line printed is the directory to serve; ``start.sh`` passes it to the server.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "corpus-manifest.json"
RELEASE_FILES = ("checksums.json", "manifest.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> None:
    """Check a directory against the snapshot it was prepared from, or a release's checksums."""
    pinned = root / "corpus-manifest.json"
    if pinned.exists():
        manifest = json.loads(pinned.read_text())
        bad = [e["local"] for e in _entries(manifest) if not _matches(root / e["local"], _unpacked(e))]
        if bad:
            raise SystemExit(f"prepare: {root} does not match its corpus manifest: {bad[:5]}")
        print(f"prepare: verified {len(_entries(manifest))} file(s) in {root}", file=sys.stderr)
        return
    if not all((root / name).is_dir() for name in ("tasks-private", "stocks", "library")):
        raise SystemExit(f"{root} needs tasks-private/, stocks/ and library/")
    manifest = root / "checksums.json"
    if not manifest.exists():
        print(f"prepare: {root} has no checksums.json; skipping verification", file=sys.stderr)
        return
    expected = json.loads(manifest.read_text())["sha256"]
    # The tasks, the stock and the reaction library all decide rewards, so a
    # missing or altered file is a startup failure, not something to skip.
    required = {name for name in expected if name.startswith(("tasks-private/", "stocks/", "library/"))}
    for name in sorted(required):
        path = root / name
        if not path.exists():
            raise SystemExit(f"prepare: {path} is listed in checksums.json but missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise SystemExit(f"prepare: checksum mismatch for {path}")
    print(f"prepare: verified {len(required)} file(s) in {root}", file=sys.stderr)


def _entries(manifest: dict) -> list[dict]:
    return [*manifest["files"], *manifest.get("evalsets", {}).values()]


def _unpacked(entry: dict) -> dict:
    """What the prepared file must match: the entry itself, or what its gzip expands to."""
    return entry.get("unpacked") or entry


def _matches(path: Path, expected: dict) -> bool:
    return path.exists() and path.stat().st_size == expected["size"] and sha256_file(path) == expected["sha256"]


def _fetched(prepared: Path, entry: dict) -> Path:
    """Where a file lands before it is checked: still gzipped if the bucket stores it packed."""
    target = prepared / entry["local"]
    return target.with_name(target.name + (".gz.partial" if entry.get("unpacked") else ".partial"))


def _settle(prepared: Path, entry: dict) -> bool:
    """Check a fetched file, unpack it if needed, and move it into place; False if it is bad."""
    target, fetched = prepared / entry["local"], _fetched(prepared, entry)
    if not _matches(fetched, entry):
        fetched.unlink()
        return False
    if entry.get("unpacked"):
        unpacked = target.with_name(target.name + ".partial")
        with gzip.open(fetched, "rb") as source, unpacked.open("wb") as sink:
            shutil.copyfileobj(source, sink, 1 << 20)
        fetched.unlink()
        if not _matches(unpacked, entry["unpacked"]):
            unpacked.unlink()
            return False
        fetched = unpacked
    fetched.replace(target)
    return True


def from_bucket(manifest_path: Path, prepared: Path) -> Path:
    """Fetch and check a pinned serving snapshot into ``prepared``."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "ready" or manifest.get("storage") != "bucket-sqlite":
        raise SystemExit(f"prepare: {manifest_path} is not a finished serving snapshot")
    bucket = os.getenv("RETROENV_BUCKET_ID") or manifest["bucket_id"]
    mount = Path(os.environ["RETROENV_BUCKET_ROOT"]) if os.getenv("RETROENV_BUCKET_ROOT") else None
    prepared.mkdir(parents=True, exist_ok=True)
    pending, copied, kept = [], 0, 0
    for entry in _entries(manifest):
        target = prepared / entry["local"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if _matches(target, _unpacked(entry)):
            kept += 1
            continue
        mounted = mount / entry["path"] if mount else None
        if mounted is not None and mounted.exists() and mounted.stat().st_size == entry["size"]:
            shutil.copyfile(mounted, _fetched(prepared, entry))
            copied += 1
        else:
            pending.append(entry)
    if pending:
        from huggingface_hub import HfApi

        HfApi().download_bucket_files(
            bucket,
            [(e["path"], str(_fetched(prepared, e))) for e in pending],
            raise_on_missing_files=True,
            token=os.getenv("HF_TOKEN") or None,
        )
    # Settle every fetched file before failing: good ones are kept for the next attempt.
    bad = [e["path"] for e in _entries(manifest) if _fetched(prepared, e).exists() and not _settle(prepared, e)]
    if bad:
        raise SystemExit(f"prepare: {bad} from {bucket} does not match its manifest digest")
    # An evaluation set from an earlier snapshot would otherwise still be served.
    listed = {Path(e["local"]).name for e in manifest.get("evalsets", {}).values()}
    for stale in (prepared / "evalsets").glob("*.json"):
        if stale.name not in listed:
            stale.unlink()
    shutil.copyfile(manifest_path, prepared / "corpus-manifest.json")
    print(
        f"prepare: snapshot {manifest['snapshot_id'][:12]} from {bucket}: "
        f"{kept} kept, {copied} copied from {mount}, {len(pending)} downloaded",
        file=sys.stderr,
    )
    verify(prepared)
    return prepared


def from_repo(repo: str, prepared: Path) -> Path:
    from huggingface_hub import snapshot_download

    repo_id, _, revision = repo.partition("@")
    subdir = os.getenv("RETROENV_TASKS_SUBDIR", "").strip("/")
    prefix = f"{subdir}/" if subdir else ""
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision or None,
        local_dir=prepared,
        allow_patterns=[f"{prefix}{name}" for name in ("tasks-private/*", "stocks/*", "library/*", *RELEASE_FILES)],
        token=os.getenv("HF_TOKEN") or None,
    )
    target = prepared / subdir if subdir else prepared
    verify(target)
    return target


def main() -> int:
    local = os.getenv("RETROENV_BENCHMARK_DIR")
    if local:
        verify(Path(local))
        print(local)
        return 0
    prepared = Path(os.getenv("RETROENV_PREPARED_DIR", "prepared"))
    manifest = os.getenv("RETROENV_CORPUS_MANIFEST")
    repo = os.getenv("RETROENV_TASKS_REPO")
    if manifest or (not repo and DEFAULT_MANIFEST.exists()):
        target = from_bucket(Path(manifest) if manifest else DEFAULT_MANIFEST, prepared)
    elif repo:
        target = from_repo(repo, prepared)
    else:
        raise SystemExit(
            "Set RETROENV_BENCHMARK_DIR, RETROENV_CORPUS_MANIFEST (a serving snapshot), "
            "or RETROENV_TASKS_REPO to a released task dataset on the Hub."
        )
    # start.sh serves whatever this prints last.
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
