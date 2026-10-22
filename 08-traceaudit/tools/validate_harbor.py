"""Validate every stored trajectory with a hash-checked, pinned official Harbor validator.

--fetch bootstraps only pure upstream Pydantic models/validator, not the full agent runtime.
Subsequent validation is offline. Python namespace stubs do not change validation logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "upstream"
PROVENANCE = json.loads((ROOT / "data/schema/provenance.json").read_text(encoding="utf-8"))
STUB = '"""Namespace for the pinned, unmodified upstream ATIF validation modules."""\n'


def prepare(fetch=False):
    for name, digest in PROVENANCE["upstream_files"].items():
        path = CACHE / name
        if not path.exists() and fetch:
            path.parent.mkdir(parents=True, exist_ok=True)
            if name in {
                "harbor/__init__.py",
                "harbor/models/__init__.py",
                "harbor/utils/__init__.py",
            }:
                payload = STUB.encode()
            else:
                url = (
                    "https://raw.githubusercontent.com/harbor-framework/harbor/"
                    f"{PROVENANCE['commit']}/src/{name}"
                )
                with urlopen(url, timeout=45) as response:
                    payload = response.read()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError(f"Upstream hash mismatch: {name}")
            path.write_bytes(payload)
        if not path.exists():
            raise ValueError(
                "Pinned Harbor files missing. "
                "Run tools/validate_harbor.py --fetch once during setup"
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Cached validator hash mismatch: {name}")
    schema = ROOT / "envs/trace_audit/core/atif-v1.7.schema.json"
    if hashlib.sha256(schema.read_bytes()).hexdigest() != PROVENANCE["schema_sha256"]:
        raise ValueError("Bundled schema hash mismatch")
    if str(CACHE) not in sys.path:
        sys.path.insert(0, str(CACHE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    prepare(args.fetch)
    from harbor.utils.trajectory_validator import TrajectoryValidator

    if str(ROOT / "envs") not in sys.path:
        sys.path.insert(0, str(ROOT / "envs"))
    from trace_audit.core import load_records

    count = 0
    for path in args.paths or [ROOT / "data/sample.jsonl"]:
        for record in load_records(path):
            validator = TrajectoryValidator()
            if not validator.validate(record["trajectory"], validate_images=False):
                raise ValueError(f"Harbor rejected {record['record_id']}: {validator.errors}")
            count += 1
    print(
        json.dumps(
            {
                "validated": count,
                "schema": "ATIF-v1.7",
                "official_harbor_commit": PROVENANCE["commit"],
            }
        )
    )


if __name__ == "__main__":
    main()
