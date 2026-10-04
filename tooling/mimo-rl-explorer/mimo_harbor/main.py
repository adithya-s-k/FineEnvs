"""Convert MiMo-V2.6-RL-oss to Harbor tasks.

    uv run python -m mimo_harbor.main --output-dir ../harbor-datasets/mimo-v2.6-rl
    uv run python -m mimo_harbor.main --output-dir OUT --split parity        # the parity subset only
    uv run python -m mimo_harbor.main --output-dir OUT --task-ids arvo_42480818 --overwrite

Writes OUT/<dataset>/<task>/ for the six datasets (code, cyber, general, terminal, webdev, music) and
OUT/<dataset>/MANIFEST.json, the sha256 of every task directory, so a re-run can be checked byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import adapter, jobs, source

PARITY_PER_DATASET = 6


def task_digest(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in d.rglob("*") if x.is_file()):
        h.update(p.relative_to(d).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def parity_ids(rows: dict[str, dict]) -> list[str]:
    """A fixed, representative subset: per dataset, the tasks whose sha256(id) sort first."""
    out = []
    for kind in sorted({source.kind(r) for r in rows.values()}):
        ids = [t for t, r in rows.items() if source.kind(r) == kind]
        out += sorted(ids, key=lambda t: hashlib.sha256(t.encode()).hexdigest())[:PARITY_PER_DATASET]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--task-ids", nargs="*")
    ap.add_argument("--datasets", nargs="*", help="code cyber general terminal webdev music")
    ap.add_argument("--split", choices=["full", "parity"], default="full")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    rows = source.load_rows()
    ids = list(rows) if a.split == "full" else parity_ids(rows)
    if a.task_ids:
        ids = [t for t in ids if t in set(a.task_ids)]
    if a.datasets:
        ids = [t for t in ids if source.kind(rows[t]) in a.datasets]
    if a.limit:
        ids = ids[: a.limit]
    out = a.output_dir.resolve()

    def one(tid: str) -> tuple[str, str, str]:
        kind = source.kind(rows[tid])
        d = out / kind / adapter.slug(tid)
        if d.exists() and not a.overwrite:
            return kind, d.name, task_digest(d)
        d = adapter.write(adapter.Task(tid, rows[tid]), out)
        return kind, d.name, task_digest(d)

    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, ids))
    by_kind: dict[str, dict[str, str]] = {}
    for kind, name, digest in results:
        by_kind.setdefault(kind, {})[name] = digest
    for kind, tasks in sorted(by_kind.items()):
        mf = out / kind / "MANIFEST.json"
        old = json.loads(mf.read_text())["tasks"] if mf.exists() else {}
        old.update(tasks)
        mf.write_text(json.dumps({"adapter": f"mimo_harbor {adapter.ADAPTER_VERSION}", "source": source.DATASET,
                                  "revision": source.REVISION, "tasks": dict(sorted(old.items()))}, indent=1) + "\n")
        print(f"{kind}: {len(tasks)} tasks written ({len(old)} in the manifest)")
    jobs.write(out, sorted(by_kind))


if __name__ == "__main__":
    main()
