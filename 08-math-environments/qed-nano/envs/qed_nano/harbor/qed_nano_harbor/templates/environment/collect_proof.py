#!/usr/bin/env python3
"""Hand the proof to the verifier: copy /workspace/proof.md, if it is a regular file, to /logs/artifacts/proof.md.

Harbor runs this in the agent's sandbox after the agent finishes. Only this one file crosses to the separate
verifier sandbox; symlinks and anything else are ignored. The verifier re-checks size and encoding.
"""
import os
import shutil
import stat
from pathlib import Path

SRC, OUT = Path("/workspace/proof.md"), Path("/logs/artifacts")
LIMIT = 256_000

OUT.mkdir(parents=True, exist_ok=True)
for entry in OUT.iterdir():
    shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
try:
    st = os.lstat(SRC)
    if stat.S_ISREG(st.st_mode) and st.st_size <= LIMIT:
        (OUT / "proof.md").write_bytes(SRC.read_bytes())
        print(f"collected {st.st_size} bytes")
    else:
        print("proof.md is not a regular file within the size limit; nothing collected")
except FileNotFoundError:
    print("no /workspace/proof.md")
