#!/usr/bin/env python3
"""Hash downloaded raw artifacts into a reproducible, source-grouped inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = HERE.parent / "data" / "raw"
DEFAULT_OUTPUT = DEFAULT_RAW_DIR / "RAW_INVENTORY.json"
EXCLUDED_PARTS = {".git", "__pycache__"}
EXCLUDED_NAMES = {"RAW_INVENTORY.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(raw_dir: Path) -> dict[str, Any]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    artifacts: list[dict[str, Any]] = []
    source_totals: dict[str, dict[str, int]] = {}
    for root, directories, files in os.walk(raw_dir):
        directories[:] = sorted(directory for directory in directories if directory not in EXCLUDED_PARTS)
        for filename in sorted(files):
            path = Path(root) / filename
            relative = path.relative_to(raw_dir)
            if filename in EXCLUDED_NAMES or any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            size = path.stat().st_size
            source_id = relative.parts[0] if len(relative.parts) > 1 else "_root"
            artifacts.append(
                {
                    "path": relative.as_posix(),
                    "source_id": source_id,
                    "size": size,
                    "sha256": sha256(path),
                }
            )
            total = source_totals.setdefault(source_id, {"files": 0, "bytes": 0})
            total["files"] += 1
            total["bytes"] += size
    artifacts.sort(key=lambda item: item["path"])
    return {
        "schema_version": "retro-raw-inventory-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "sha256",
        "exclusions": [".git/**", "**/__pycache__/**", "RAW_INVENTORY.json"],
        "totals": {
            "files": len(artifacts),
            "bytes": sum(item["size"] for item in artifacts),
        },
        "sources": dict(sorted(source_totals.items())),
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = inventory(args.raw_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "files": value["totals"]["files"],
                "bytes": value["totals"]["bytes"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
