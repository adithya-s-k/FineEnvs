"""Snapshot the repository inside each Code task's image, for the repository browser on the task page.

Each Code image is one flattened multi-GB layer, so the repository can't be read from the registry cheaply.
Instead this starts the task's own image as an HF Sandbox (the same image a rollout uses, untouched), and records:
the file list with sizes (`git ls-files`), the base commit, the upstream remote, whether history is truncated at the
base, and the content of small text files. The result is one gzipped JSON per task in <store>/repo-snapshots/.

    uv run python -m app.snapshot_repos --workers 8                    # every Code task not yet snapshotted
    uv run python -m app.snapshot_repos --ids format-code-task-000158  # specific tasks
    STORAGE_DIR=... to write where the Space reads (its bucket mount is /data)

Runs on your HF account: about a minute of cpu-basic sandbox per task.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import catalog, config

OUT = config.STORAGE_DIR / "repo-snapshots"
TEXT_MAX = 200_000        # per file
TEXT_BUDGET = 25_000_000  # per repository, before compression

PROBE = r'''
import json, os, subprocess, sys, gzip
cwd = sys.argv[1]; text_max = int(sys.argv[2]); budget = int(sys.argv[3])
def git(*a):
    r = subprocess.run(["git", "-c", "safe.directory=*", *a], cwd=cwd, capture_output=True)
    return r.stdout.decode(errors="replace").strip() if r.returncode == 0 else ""
base = git("rev-parse", "HEAD")
remote = git("remote", "get-url", "origin")
extra = [x for x in git("rev-list", "--all", "--not", base).split() if len(x) == 40] if base else []
names = [n for n in (git("ls-files", "-z").split("\0")) if n] if base else []
if not names:   # not a git work tree: walk it
    for root, ds, fs in os.walk(cwd):
        ds[:] = [d for d in ds if d not in (".git", "node_modules", ".venv", "target", "__pycache__")]
        names += [os.path.relpath(os.path.join(root, f), cwd) for f in fs]
files, texts, used = [], {}, 0
for n in sorted(names):
    p = os.path.join(cwd, n)
    try:
        size = os.path.getsize(p)
    except OSError:
        continue
    entry = {"path": n, "size": size, "preview": False}
    if size <= text_max and used + size <= budget:
        try:
            b = open(p, "rb").read()
            if b"\0" not in b[:8192]:
                texts[n] = b.decode("utf-8"); used += size; entry["preview"] = True
        except (OSError, UnicodeDecodeError):
            pass
    files.append(entry)
doc = {"base": base, "remote": remote, "history_truncated": not extra, "files": files, "texts": texts}
with gzip.open("/tmp/.snapshot.json.gz", "wt", encoding="utf-8") as f:
    json.dump(doc, f)
print("SNAPSHOT_OK", len(files), len(texts), used)
'''


def snapshot(task_id: str, token: str) -> dict:
    from huggingface_hub import Sandbox

    inst = catalog.rows()[task_id]["instance"]
    sb = Sandbox.create(image=catalog.image_for(task_id), flavor="cpu-basic", idle_timeout=600, forward_hf_token=False,
                        token=token, start_timeout=900, labels={"app": "mimo-explorer", "job": "repo-snapshot"})
    try:
        sb.files.write("/tmp/.snapshot.py", PROBE)
        res = sb.run(f"python3 /tmp/.snapshot.py {inst['cwd']} {TEXT_MAX} {TEXT_BUDGET}", shell=True, check=False, timeout=600)
        if "SNAPSHOT_OK" not in (res.stdout or ""):
            raise RuntimeError(((res.stderr or "") + (res.stdout or ""))[-400:])
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "s.json.gz")
            sb.files.download("/tmp/.snapshot.json.gz", local)
            doc = json.loads(gzip.decompress(open(local, "rb").read()))
    finally:
        try:
            Sandbox.kill(sb.id, token=token)
        except Exception:
            pass
    doc.update(task_id=task_id, image=catalog.image_for(task_id), cwd=inst["cwd"], taken_at=time.time())
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f".{task_id}.json.gz.tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(doc, f)
    tmp.replace(OUT / f"{task_id}.json.gz")
    return {"task_id": task_id, "files": len(doc["files"]), "texts": len(doc["texts"]), "truncated": doc["history_truncated"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from huggingface_hub import get_token

    token = get_token()
    ids = a.ids or sorted(t for t, r in catalog.rows().items() if r["instance"].get("dataset_type") == "opensource-code"
                          and not (OUT / f"{t}.json.gz").exists())
    if a.limit:
        ids = ids[:a.limit]
    print(f"snapshotting {len(ids)} repositories into {OUT}", flush=True)
    done = bad = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(snapshot, t, token): t for t in ids}
        for f in as_completed(futs):
            try:
                r = f.result(); done += 1
                print(f"[{done + bad}/{len(ids)}] ok   {r['task_id']}  {r['files']} files, {r['texts']} previewable, history {'truncated' if r['truncated'] else 'NOT truncated'}", flush=True)
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"[{done + bad}/{len(ids)}] FAIL {futs[f]}  {type(e).__name__}: {str(e)[:200]}", flush=True)
    print(f"done: {done} ok, {bad} failed")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
