#!/usr/bin/env python3
"""Resumably fetch pinned, reviewed raw sources and write machine-readable receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "sources.json"
DEFAULT_OUTPUT = HERE.parent / "data" / "raw"
DEFAULT_STATUSES = {"approved", "approved_noncommercial"}


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "retro-source-manifest-v1":
        raise ValueError("unsupported source manifest")
    return value


def fetch_source(source: dict[str, Any], root: Path) -> dict[str, Any]:
    fetch = source.get("fetch")
    if not fetch:
        raise ValueError(f"source {source['id']} has no reviewed fetch configuration")
    destination = root / source["id"]
    kind = fetch["type"]
    if kind == "git":
        details = _fetch_git(fetch, destination)
    elif kind == "figshare":
        details = _fetch_figshare(fetch, destination)
    elif kind == "huggingface_dataset":
        details = _fetch_huggingface(fetch, destination)
    elif kind == "zenodo":
        details = _fetch_zenodo(fetch, destination)
    else:
        raise ValueError(f"unsupported fetch type: {kind}")
    receipt = {
        "source_id": source["id"],
        "title": source["title"],
        "status": source["status"],
        "license": source["license"],
        "evidence_kind": source.get("evidence_kind"),
        "homepage": source.get("homepage"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "fetch": fetch,
        "result": details,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "RETROENV_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _fetch_git(fetch: dict[str, Any], destination: Path) -> dict[str, Any]:
    revision = fetch["revision"]
    if not (destination / ".git").exists():
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f"refusing to replace non-git directory: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", fetch["url"], str(destination)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", "--force", revision],
        check=True,
    )
    if shutil.which("git-lfs"):
        subprocess.run(["git", "-C", str(destination), "lfs", "pull"], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise RuntimeError(f"git revision mismatch: expected {revision}, got {actual}")
    return {"revision": actual, "path": str(destination)}


def _fetch_figshare(fetch: dict[str, Any], destination: Path) -> dict[str, Any]:
    article_id = int(fetch["article_id"])
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        metadata = json.loads(response.read())
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for spec in metadata["files"]:
        target = destination / spec["name"]
        expected = spec.get("supplied_md5") or spec.get("computed_md5")
        if target.exists() and (not expected or _md5(target) == expected):
            status = "already_present"
        else:
            temporary = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(
                spec["download_url"], headers={"User-Agent": "RetroEnv/0.1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if expected and _md5(temporary) != expected:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch for {spec['name']}")
            os.replace(temporary, target)
            status = "downloaded"
        files.append(
            {
                "name": spec["name"],
                "size": spec["size"],
                "md5": expected,
                "status": status,
            }
        )
    metadata_path = destination / "figshare_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "article_id": article_id,
        "doi": metadata.get("doi"),
        "license": (metadata.get("license") or {}).get("name"),
        "files": files,
    }


def _fetch_huggingface(fetch: dict[str, Any], destination: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install huggingface_hub to fetch Hugging Face datasets") from exc
    destination.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        repo_id=fetch["repo_id"],
        repo_type="dataset",
        revision=fetch["revision"],
        local_dir=destination,
    )
    return {"revision": fetch["revision"], "path": str(result)}


def _fetch_zenodo(fetch: dict[str, Any], destination: Path) -> dict[str, Any]:
    record_id = int(fetch["record_id"])
    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        metadata = json.loads(response.read())
    available = {item["key"]: item for item in metadata["files"]}
    selected = list(fetch.get("files") or sorted(available))
    missing = sorted(set(selected) - set(available))
    if missing:
        raise RuntimeError(f"Zenodo record {record_id} is missing files: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for name in selected:
        spec = available[name]
        target = destination / name
        checksum = str(spec.get("checksum") or "")
        algorithm, expected = checksum.split(":", 1) if ":" in checksum else ("", "")
        if algorithm and algorithm != "md5":
            raise RuntimeError(f"unsupported Zenodo checksum {checksum!r} for {name}")
        if target.exists() and (not expected or _md5(target) == expected):
            status = "already_present"
        else:
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.unlink(missing_ok=True)
            request = urllib.request.Request(
                spec["links"]["self"], headers={"User-Agent": "RetroEnv/0.1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if expected and _md5(temporary) != expected:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch for {name}")
            os.replace(temporary, target)
            status = "downloaded"
        files.append(
            {
                "name": name,
                "size": spec["size"],
                "md5": expected or None,
                "status": status,
            }
        )
    license_value = (metadata.get("metadata") or {}).get("license") or {}
    return {
        "record_id": record_id,
        "doi": (metadata.get("metadata") or {}).get("doi"),
        "license": license_value.get("id") if isinstance(license_value, dict) else license_value,
        "files": files,
    }


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--include-research-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    by_id = {source["id"]: source for source in manifest["sources"]}
    if args.list:
        for source in manifest["sources"]:
            print(f"{source['id']:28s} {source['status']:24s} {source['license']}")
        return 0

    if args.source:
        missing = sorted(set(args.source) - set(by_id))
        if missing:
            parser.error(f"unknown source(s): {', '.join(missing)}")
        selected = [by_id[source_id] for source_id in args.source]
    else:
        statuses = set(DEFAULT_STATUSES)
        if args.include_research_only:
            statuses.add("research_only")
        selected = [source for source in manifest["sources"] if source["status"] in statuses]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in selected:
        if source["status"] == "pending_review":
            raise RuntimeError(f"refusing unreviewed source: {source['id']}")
        print(f"fetching {source['id']} ({source['license']})", flush=True)
        fetch_source(source, args.output_dir)
    # A filtered/resumed run must not erase receipts from earlier sources.
    receipts = []
    for receipt_path in sorted(args.output_dir.glob("*/RETROENV_RECEIPT.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipts.append(receipt)
    receipts.sort(key=lambda item: item["source_id"])
    summary = args.output_dir / "DOWNLOAD_RECEIPTS.json"
    summary.write_text(json.dumps(receipts, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
