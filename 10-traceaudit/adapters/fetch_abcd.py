"""Restore pinned public ABCD source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1] / "data" / "sources" / "abcd"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE_DIR / "raw")
    args = parser.parse_args()
    manifest = json.loads((SOURCE_DIR / "source.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, record in manifest["files"].items():
        if Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("Manifest filename must be a single safe path component")
        target = args.output / filename
        if target.exists():
            payload = target.read_bytes()
            if (
                len(payload) == record["bytes"]
                and hashlib.sha256(payload).hexdigest() == record["sha256"]
            ):
                print(f"Verified existing {filename}")
                continue
            raise ValueError(f"Existing file differs from the pinned source: {target}")
        request = urllib.request.Request(
            record["url"], headers={"User-Agent": "FineEnvs-TraceAudit"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        if (
            len(payload) != record["bytes"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Downloaded source verification failed: {filename}")
        target.write_bytes(payload)
        print(f"Downloaded and verified {filename}")
    (args.output / "source.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
