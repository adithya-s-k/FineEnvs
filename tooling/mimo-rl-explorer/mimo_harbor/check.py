"""Static checks on a converted dataset, with Harbor's own task loader (run it with Harbor installed):

    python -m mimo_harbor.check /path/to/mimo-v2.6-rl-harbor

For every task: Harbor loads it; the image is pinned by digest and the Dockerfile uses the same one; the setup the
healthcheck unpacks is byte-identical to environment/setup/; the grading files are only under tests/; nothing a grader
needs leaks into instruction.md or the setup; the directory hashes to its MANIFEST.json entry.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

EXPECTED = {"code": 2698, "cyber": 1000, "general": 925, "terminal": 64, "webdev": 2093, "music": 1000}
SECRET = re.compile(r"\b(hf_|sk-)[A-Za-z0-9]{8,}")


def digest(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        h.update(p.relative_to(d).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def check_task(d: Path, kind: str, manifest: dict) -> list[str]:
    from harbor.models.task.task import Task

    errs = []
    try:
        t = Task(d)
    except Exception as e:  # noqa: BLE001
        return [f"harbor cannot load it: {type(e).__name__}: {e}"]
    c = t.config
    img = c.environment.docker_image or ""
    if "@sha256:" not in img:
        errs.append(f"image not pinned by digest: {img}")
    if (d / "environment" / "Dockerfile").read_text().strip().splitlines()[-1] != f"FROM {img}":
        errs.append("Dockerfile FROM differs from docker_image")
    hc = c.environment.healthcheck.command if c.environment.healthcheck else ""
    m = re.search(r"echo ([A-Za-z0-9+/=]+) \| base64 -d", hc)
    if not m:
        errs.append("no setup payload in the healthcheck")
    else:
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(m.group(1))))) as tf:
            payload = {i.name: tf.extractfile(i).read() for i in tf.getmembers() if i.isfile()}
        readable = {p.relative_to(d / "environment" / "setup").as_posix(): p.read_bytes()
                    for p in (d / "environment" / "setup").rglob("*") if p.is_file()}
        if payload != readable:
            errs.append("healthcheck payload differs from environment/setup/")
        blob = b"".join(payload.values())
        if SECRET.search(blob.decode(errors="replace")):
            errs.append("something token-shaped in the setup")
    instr = (d / "instruction.md").read_text()
    if not instr.strip():
        errs.append("empty instruction")
    if SECRET.search(instr):
        errs.append("something token-shaped in the instruction")
    if not (d / "tests" / "test.sh").exists():
        errs.append("no tests/test.sh")
    if kind == "general":   # the rubric (verify.py, verifier_meta.json) must only be under tests/
        if any((d / "environment").rglob("verifier_meta.json")) or b"verifier_meta" in (d / "environment" / "setup" / "files" / "fetch.json").read_bytes():
            errs.append("rubric reachable before grading")
        if not list((d / "tests" / "verifier").rglob("verify.py")):
            errs.append("no verify.py under tests/verifier")
        if not c.environment.mcp_servers:
            errs.append("no MCP servers")
    if kind in ("cyber", "general") and c.agent.user != "agent":
        errs.append("agent should run as the unprivileged `agent` user")
    if manifest.get(d.name) != digest(d):
        errs.append("differs from MANIFEST.json")
    return errs


def main(root: Path) -> int:
    bad, counts = 0, Counter()
    for kind, n in EXPECTED.items():
        manifest = json.loads((root / kind / "MANIFEST.json").read_text())["tasks"]
        dirs = sorted(p for p in (root / kind).iterdir() if p.is_dir())
        counts[kind] = len(dirs)
        if len(dirs) != n or len(manifest) != n:
            print(f"{kind}: {len(dirs)} task dirs, {len(manifest)} in the manifest, expected {n}")
            bad += 1
        for d in dirs:
            for e in check_task(d, kind, manifest):
                bad += 1
                print(f"{kind}/{d.name}: {e}")
    print(f"checked {sum(counts.values())} tasks: {dict(counts)}; {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
