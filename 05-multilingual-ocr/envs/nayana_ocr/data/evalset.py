"""Frozen evaluation set: fixed task IDs, spread across documents, no silent substitution.

A source block is ~100 consecutive pages and in practice one document. Drawing a whole
language from one block would be cheap and would also score that language on a single
document, so the selection caps how many tasks of a family may come from one block and
walks on to further blocks. Every fetched block still serves one task per family, so the
cold cost stays a fraction of uniform sampling while the set covers many documents.
"""

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .schema import FAMILIES, SCHEMA_VERSION, canonical_json

EVALSET_VERSION = 1
DEFAULT_SIZE = 500
DEFAULT_SPLIT = "test"
# The published 22-language index set is 4.30 GB; a 4 GB budget cannot hold it, and an
# all-language evaluation would re-download ~200 MB databases as it rotates languages.
INDEX_CACHE_BYTES = 6_000_000_000
# One task per family per source block: a language spans as many documents as its
# largest family quota, instead of collapsing onto the single densest block.
PER_BLOCK_PER_FAMILY = 1


def allocate(languages, families, size):
    """Per-(language, family) counts totalling exactly ``size``.

    Families are held equal so per-family means carry the same sample size. The
    within-family remainder rotates across families, keeping language totals within
    one task of each other.
    """
    if not languages or not families or size < len(languages) * len(families):
        raise ValueError("Allocate at least one task per language/task group")
    if len(set(languages)) != len(languages) or len(set(families)) != len(families):
        raise ValueError("Duplicate language or task family")
    counts = {}
    base_family, extra_families = divmod(size, len(families))
    for index, family in enumerate(families):
        target = base_family + (index < extra_families)
        base, remainder = divmod(target, len(languages))
        for offset, language in enumerate(languages):
            # Rotate which languages absorb the remainder so no language is always short.
            rotated = (offset - index * remainder) % len(languages)
            counts[language, family] = base + (rotated < remainder)
    if sum(counts.values()) != size:
        raise ValueError("Allocation does not total the requested size")
    return counts


def _reference_digest(task):
    reference = task.get("reference")
    if not isinstance(reference, str):
        raise ValueError(f"Task {task.get('task_id')!r} carries no text reference")
    return hashlib.sha256(reference.encode()).hexdigest()


def _order(seed, snapshot_id, *parts):
    return hashlib.sha256(
        canonical_json(["nayana-evalset-v1", seed, snapshot_id, *parts]).encode()
    ).digest()


def _block_candidates(catalog, split, language, families, seed):
    """Hash order, deliberately not density order.

    Taking only one task per family from a block means a block just has to contain the
    families still needed; its density buys nothing. Preferring dense blocks actively
    hurts, because the corpus is parallel translations of the same sources: the densest
    blocks are the *same documents* in every language, so all 22 languages converge on
    them. Measured over the 500-task set, density order covered 28 unique documents and
    hash order covered 181, while reading less data.
    """
    blocks = catalog.blocks(split, [language], list(families))
    if not blocks:
        raise ValueError(f"No {split} blocks indexed for {language!r}")
    return sorted(
        blocks,
        key=lambda block: _order(seed, catalog.snapshot_id, block["block_id"]),
    )


def _block_tasks(catalog, block_id, split, families):
    rows, start = [], 0
    while True:
        page = catalog.block_tasks(block_id, split, list(families), start, 1000)
        rows.extend(page)
        if len(page) < 1000:
            return rows
        start += len(page)


def select(
    catalog,
    *,
    split=DEFAULT_SPLIT,
    languages=None,
    families=FAMILIES,
    size=DEFAULT_SIZE,
    seed=42,
    per_block_per_family=PER_BLOCK_PER_FAMILY,
):
    """Choose the fixed task IDs, spread across source documents.

    One source block is ~100 consecutive pages and in practice a single document, so
    taking a language's whole allocation from the densest block would score that language
    on one document. ``per_block_per_family`` caps how many tasks of a family may come
    from one block, which forces the walk onward to further blocks and therefore further
    documents. At the default of 1 a language draws from as many blocks as its largest
    family quota, and each fetched block still serves one task per family rather than one
    task overall.
    """
    if per_block_per_family < 1:
        raise ValueError("Take at least one task per family from a chosen block")
    languages = list(languages or catalog.languages)
    families = list(families)
    counts = allocate(languages, families, size)
    selection, blocks_used = [], []
    for language in languages:
        needed = {family: counts[language, family] for family in families}
        for block in _block_candidates(catalog, split, language, families, seed):
            if not any(needed.values()):
                break
            rows = _block_tasks(catalog, block["block_id"], split, families)
            taken = 0
            for family in families:
                if not needed[family]:
                    continue
                available = sorted(
                    (row for row in rows if row["family"] == family),
                    key=lambda row: _order(seed, catalog.snapshot_id, row["task_id"]),
                )
                for row in available[: min(needed[family], per_block_per_family)]:
                    selection.append(
                        {
                            "task_id": row["task_id"],
                            "language": language,
                            "family": family,
                            "block_id": row["block_id"],
                        }
                    )
                    needed[family] -= 1
                    taken += 1
            if taken:
                blocks_used.append(block)
        if any(needed.values()):
            short = {family: count for family, count in needed.items() if count}
            raise ValueError(f"{language}: {split} split cannot supply {short}")
    if len(selection) != size:
        raise ValueError(f"Selected {len(selection)} tasks; expected {size}")
    return selection, blocks_used


def build(
    catalog,
    *,
    split=DEFAULT_SPLIT,
    languages=None,
    families=FAMILIES,
    size=DEFAULT_SIZE,
    seed=42,
    validate=True,
    per_block_per_family=PER_BLOCK_PER_FAMILY,
):
    """Select, optionally load every task once, and return the frozen record.

    Loading is what proves a task is actually usable: image bounds and annotation
    validity are checked at load time, not at index time. A task that fails is
    recorded, never silently swapped for another.
    """
    selection, blocks_used = select(
        catalog,
        split=split,
        languages=languages,
        families=families,
        size=size,
        seed=seed,
        per_block_per_family=per_block_per_family,
    )
    failures = []
    for entry in selection:
        task = catalog.get(entry["task_id"])
        entry["page_id"] = task["page_id"]
        entry["document_id"] = task["document_id"]
        entry["unit"] = task["unit"]
        if not validate:
            entry["reference_sha256"] = _reference_digest(task)
            continue
        try:
            # Pin what the server actually serves, not what the index predicted.
            served = catalog.materialize(task)
        except Exception as error:  # Record, do not substitute.
            entry["reference_sha256"] = _reference_digest(task)
            failures.append(
                {
                    "task_id": entry["task_id"],
                    "language": entry["language"],
                    "family": entry["family"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        entry["reference_sha256"] = _reference_digest(served)
        entry["asset_sha256"] = served["asset_sha256"]
        entry["width"] = served["width"]
        entry["height"] = served["height"]
    selection.sort(
        key=lambda entry: (entry["language"], entry["family"], entry["task_id"])
    )
    unique_blocks = {block["block_id"]: block for block in blocks_used}
    record = {
        "evalset_version": EVALSET_VERSION,
        "snapshot_id": catalog.snapshot_id,
        "split": split,
        "seed": seed,
        "size": len(selection),
        "validated": bool(validate),
        "languages": list(languages or catalog.languages),
        "families": list(families),
        "per_block_per_family": per_block_per_family,
        "blocks": sorted(unique_blocks),
        "documents": sorted({entry["document_id"] for entry in selection}),
        "block_bytes": sum(block["image_bytes"] for block in unique_blocks.values()),
        "failures": failures,
        "tasks": selection,
    }
    record["evalset_id"] = hashlib.sha256(
        canonical_json(
            [
                EVALSET_VERSION,
                catalog.snapshot_id,
                split,
                seed,
                [entry["task_id"] for entry in selection],
            ]
        ).encode()
    ).hexdigest()
    return record


def save(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return path


def eval_split_name(record):
    """The split name a frozen set is served under, derived from the set itself.

    Named for how many languages it covers and which source split it draws from —
    `eval_21_validation` — so the name cannot claim something the tasks do not. The
    ASR environment derives its split names the same way, so `eval_21` means the same
    21 languages in both.
    """
    return f"eval_{len(record['languages'])}_{record['split']}"


def load(path, snapshot_id=None):
    """Read a frozen set and refuse one that does not match the served corpus."""
    record = json.loads(Path(path).read_text())
    if record.get("evalset_version") != EVALSET_VERSION:
        raise ValueError("Rebuild this evaluation set for the current evalset version")
    if snapshot_id is not None and record.get("snapshot_id") != snapshot_id:
        raise ValueError(
            "Evaluation set was built for a different corpus snapshot; rebuild it"
        )
    if record.get("failures"):
        raise ValueError(
            f"{len(record['failures'])} recorded task(s) do not load; "
            "rebuild the set or fix the source before reporting scores"
        )
    tasks = record["tasks"]
    expected = hashlib.sha256(
        canonical_json(
            [
                EVALSET_VERSION,
                record["snapshot_id"],
                record["split"],
                record["seed"],
                [entry["task_id"] for entry in tasks],
            ]
        ).encode()
    ).hexdigest()
    if expected != record.get("evalset_id"):
        raise ValueError("Evaluation set identity does not match its task IDs")
    if len(tasks) != record["size"] or len({e["task_id"] for e in tasks}) != len(tasks):
        raise ValueError("Evaluation set size or task uniqueness is inconsistent")
    return record


PACK_DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pages(language TEXT, page_id TEXT, PRIMARY KEY(language,page_id));
CREATE TABLE IF NOT EXISTS assets(sha TEXT PRIMARY KEY, mime TEXT NOT NULL, bytes INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, split TEXT NOT NULL, position INTEGER,
    payload TEXT NOT NULL, UNIQUE(split,position));
CREATE INDEX IF NOT EXISTS task_split ON tasks(split,position);
"""


def export_pack(catalog, record, directory):
    """Write a self-contained snapshot the ordinary local server can serve offline.

    The result is the same on-disk shape ``nayana-prepare`` produces, so it needs no
    bucket access, no network, and no new serving path: point ``NAYANA_SNAPSHOT`` at it.
    """
    if record["failures"]:
        raise ValueError("Refusing to export a set with recorded load failures")
    directory = Path(directory)
    (directory / "assets").mkdir(parents=True, exist_ok=True)
    counts, pages, media_bytes = {}, set(), 0
    with closing(sqlite3.connect(directory / "catalog.sqlite")) as db:
        db.executescript(PACK_DDL)
        with db:
            db.execute("DELETE FROM tasks")
            db.execute("DELETE FROM assets")
            db.execute("DELETE FROM pages")
            for position, entry in enumerate(record["tasks"]):
                task = catalog.materialize(catalog.get(entry["task_id"]))
                if task["asset_sha256"] != entry.get("asset_sha256"):
                    raise ValueError(
                        f"{entry['task_id']}: served asset no longer matches the frozen set"
                    )
                raw, mime = catalog.asset_bytes(task["asset_sha256"], task["task_id"])
                (directory / "assets" / task["asset_sha256"]).write_bytes(raw)
                payload = {
                    key: value
                    for key, value in task.items()
                    if not key.startswith("_") and key != "media"
                }
                db.execute(
                    "INSERT INTO tasks(id,split,position,payload) VALUES(?,?,?,?)",
                    (task["task_id"], task["split"], position, canonical_json(payload)),
                )
                db.execute(
                    "INSERT OR REPLACE INTO assets VALUES(?,?,?)",
                    (task["asset_sha256"], mime, len(raw)),
                )
                db.execute(
                    "INSERT OR IGNORE INTO pages VALUES(?,?)",
                    (task["language"], task["page_id"]),
                )
                media_bytes += len(raw)
                pages.add((task["language"], task["page_id"]))
                key = (task["split"], task["language"], task["family"])
                counts[key] = counts.get(key, 0) + 1
        identity = hashlib.sha256(
            canonical_json(["nayana-evalpack-v1", record["evalset_id"]]).encode()
        )
        for (payload,) in db.execute("SELECT payload FROM tasks ORDER BY id"):
            identity.update(payload.encode())
    from collections import Counter

    manifest = {
        "status": "ready",
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": identity.hexdigest(),
        "corpus_snapshot_id": record["snapshot_id"],
        "evalset_id": record["evalset_id"],
        "source_license": "cc-by-nc-4.0",
        "media_bytes": media_bytes,
        "pages": dict(Counter(language for language, _ in pages)),
        "counts": [
            {"split": split, "language": language, "family": family, "tasks": tasks}
            for (split, language, family), tasks in sorted(counts.items())
        ],
        "config": {
            "source": record.get("source", "Cognitive-Lab/NayanaOCR_Corpus_2025"),
            "languages": record["languages"],
            "families": record["families"],
            "split_seed": record["seed"],
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    # Re-key the frozen set to this pack so the same --evalset flag works against it.
    packed = dict(record, snapshot_id=manifest["snapshot_id"])
    packed["evalset_id"] = hashlib.sha256(
        canonical_json(
            [
                EVALSET_VERSION,
                packed["snapshot_id"],
                packed["split"],
                packed["seed"],
                [entry["task_id"] for entry in packed["tasks"]],
            ]
        ).encode()
    ).hexdigest()
    save(packed, directory / "evalset.json")
    return manifest


def subsample(record, limit, seed=42):
    """A deterministic, family-balanced slice of a frozen set, for smoke runs only.

    A score from a slice is not comparable with a score from the whole set, so the caller
    must record `eval_limit` alongside the `evalset_id` and never report the two as the
    same measurement.
    """
    tasks = record["tasks"]
    if limit < 1:
        raise ValueError("Use a positive evaluation limit")
    if limit >= len(tasks):
        return list(tasks)
    families = sorted({entry["family"] for entry in tasks})
    per_family, extra = divmod(limit, len(families))
    chosen = []
    for index, family in enumerate(families):
        want = per_family + (index < extra)
        # Order within a family by a seeded hash so the slice is stable but not a
        # prefix of one language.
        pool = sorted(
            (entry for entry in tasks if entry["family"] == family),
            key=lambda entry: _order(seed, record["snapshot_id"], entry["task_id"]),
        )
        chosen.extend(pool[:want])
    chosen.sort(
        key=lambda entry: (entry["language"], entry["family"], entry["task_id"])
    )
    return chosen


def summarize(record):
    from collections import Counter

    languages = Counter(entry["language"] for entry in record["tasks"])
    families = Counter(entry["family"] for entry in record["tasks"])
    return {
        "evalset_id": record["evalset_id"],
        "snapshot_id": record["snapshot_id"],
        "split": record["split"],
        "size": record["size"],
        "validated": record["validated"],
        "blocks": len(record["blocks"]),
        "block_bytes": record["block_bytes"],
        "failures": len(record["failures"]),
        "documents": len(record["documents"]),
        "tasks_per_document": round(
            record["size"] / max(1, len(record["documents"])), 2
        ),
        "per_family": dict(sorted(families.items())),
        "per_language": dict(sorted(languages.items())),
    }


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser(
        "build", help="Select, validate, and freeze the evaluation set"
    )
    make.add_argument("--manifest", default=os.environ.get("NAYANA_CORPUS_MANIFEST"))
    make.add_argument(
        "--cache-dir", default=os.environ.get("NAYANA_CACHE_DIR"), required=False
    )
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--split", default=DEFAULT_SPLIT)
    make.add_argument("--size", type=int, default=DEFAULT_SIZE)
    make.add_argument("--seed", type=int, default=42)
    make.add_argument("--languages", nargs="*", default=None)
    make.add_argument("--families", nargs="*", default=list(FAMILIES))
    make.add_argument("--source-root", default=os.environ.get("NAYANA_SOURCE_ROOT"))
    make.add_argument("--local-source", action="store_true")
    make.add_argument(
        "--index-cache-bytes",
        type=int,
        default=int(os.environ.get("NAYANA_INDEX_CACHE_BYTES", INDEX_CACHE_BYTES)),
        help="Must hold every selected language index at once, or the build re-downloads them",
    )
    make.add_argument(
        "--group-cache-bytes",
        type=int,
        default=int(os.environ.get("NAYANA_GROUP_CACHE_BYTES", "4000000000")),
    )
    make.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip loading each task; the set is then not proven usable",
    )

    check = sub.add_parser(
        "verify", help="Validate a frozen set against a served corpus"
    )
    check.add_argument("--evalset", type=Path, required=True)
    check.add_argument("--snapshot-id", default=None)

    pack = sub.add_parser(
        "export",
        help="Write a self-contained offline pack that the local server can serve",
    )
    pack.add_argument("--evalset", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--manifest", default=os.environ.get("NAYANA_CORPUS_MANIFEST"))
    pack.add_argument("--cache-dir", default=os.environ.get("NAYANA_CACHE_DIR"))
    pack.add_argument("--source-root", default=os.environ.get("NAYANA_SOURCE_ROOT"))
    pack.add_argument("--local-source", action="store_true")
    pack.add_argument(
        "--index-cache-bytes",
        type=int,
        default=int(os.environ.get("NAYANA_INDEX_CACHE_BYTES", INDEX_CACHE_BYTES)),
    )

    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(summarize(load(args.evalset, args.snapshot_id)), indent=2))
        return

    if args.command == "export":
        if not args.manifest or not args.cache_dir:
            parser.error(
                "Provide --manifest and --cache-dir (or the matching NAYANA_* variables)"
            )
        from .corpus import CorpusCatalog

        catalog = CorpusCatalog(
            args.manifest,
            args.cache_dir,
            source_root=args.source_root or None,
            local_source=args.local_source,
            index_cache_bytes=args.index_cache_bytes,
        )
        try:
            record = load(args.evalset, catalog.snapshot_id)
            manifest = export_pack(catalog, record, args.output)
        finally:
            catalog.close()
        print(json.dumps(manifest, indent=2)[:2000])
        return

    if not args.manifest or not args.cache_dir:
        parser.error(
            "Provide --manifest and --cache-dir (or the matching NAYANA_* variables)"
        )
    from .corpus import CorpusCatalog

    catalog = CorpusCatalog(
        args.manifest,
        args.cache_dir,
        source_root=args.source_root or None,
        local_source=args.local_source,
        index_cache_bytes=args.index_cache_bytes,
        group_cache_bytes=args.group_cache_bytes,
    )
    required = sum(
        catalog.manifest["indexes"][language]["size"]
        for language in (args.languages or catalog.languages)
    )
    if required > args.index_cache_bytes:
        parser.error(
            f"Selected language indexes need {required:,} bytes but the index cache holds "
            f"{args.index_cache_bytes:,}; raise --index-cache-bytes to avoid re-downloading them"
        )
    try:
        record = build(
            catalog,
            split=args.split,
            languages=args.languages,
            families=args.families,
            size=args.size,
            seed=args.seed,
            validate=not args.no_validate,
        )
    finally:
        catalog.close()
    save(record, args.output)
    print(json.dumps(summarize(record), indent=2))
    if record["failures"]:
        raise SystemExit(
            f"{len(record['failures'])} task(s) failed to load; "
            "resolve them before using this set for scores"
        )


if __name__ == "__main__":
    main()
