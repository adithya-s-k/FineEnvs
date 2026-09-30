"""Install the locked runtime from an immutable HF dataset volume."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    bundle, root = Path("/bundle"), Path("/workspace/repro")
    metadata = json.loads((bundle / "bundle.json").read_text())
    digest = hashlib.sha256((bundle / "bundle.tar.gz").read_bytes()).hexdigest()
    if digest != metadata["sha256"] or digest != os.environ["BUNDLE_SHA256"]:
        raise ValueError("Frozen runtime hash mismatch")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle / "bundle.tar.gz") as archive:
        archive.extractall(root, filter="data")
    os.environ["REPRO_ROOT"] = str(root)
    role = sys.argv[sys.argv.index("--role") + 1]
    names = ["env"] if role == "coordinator" else ["env", "train"]
    for name in names:
        target = root / ("OpenEnv/.venv" if name == "env" else ".venv312")
        subprocess.run(["uv", "venv", "--python", "3.12", str(target)], check=True)
        subprocess.run(["uv", "pip", "sync", "--python", str(target / "bin/python"), "--require-hashes",
                        str(root / "hf/locks" / f"requirements-{name}.lock")], check=True)
    python = root / ("OpenEnv/.venv/bin/python" if role == "coordinator" else ".venv312/bin/python")
    if role != "coordinator":
        subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", "--no-build-isolation",
                        "--editable", str(root / "source/trl")], check=True)
    os.execv(str(python), [str(python), "-u", str(root / "hf/runtime/hard_curriculum_job.py"), *sys.argv[1:]])


if __name__ == "__main__":
    main()
