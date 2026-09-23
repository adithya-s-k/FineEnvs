"""Frozen evaluation sets: fixed task IDs, balanced across languages and families.

Selection reads metadata columns only. Measured on one FLEURS shard that is 0.3 MB and
5.9s against 309.6 MB and 34.3s for the same rows with audio, because each published file
is a single row group and the audio column dominates it. Every eligibility rule this
environment applies - duration policy from `num_samples`, and whether a reference survives
normalization - is decidable from metadata, so the cheap pass is also the correct one.
Decoding every selected utterance is a stronger guarantee available through `--verify`,
and it costs the full ~310 MB per language.

Both published sets come out of one pass: the overlap set is a strict filter of the full
set, so it shares task IDs, costs no extra reads, and cannot disagree with it.
"""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

from .schema import (
    DEFAULT_REVISION,
    FAMILIES,
    REPO_ID,
    canonical_json,
    digest,
    task_id,
)
from .tasks import MAX_SECONDS, MIN_SECONDS, SAMPLING_RATE

EVALSET_VERSION = 1
DEFAULT_SPLIT = "test"
METADATA_COLUMNS = [
    "id",
    "num_samples",
    "transcription",
    "raw_transcription",
    "language",
]

# The OCR corpus names languages by ISO 639-1; FLEURS names them language_region. Only
# the pairs that are genuinely the same language are listed, so an overlap set can be
# compared with 05-multilingual-ocr language for language.
OCR_LANGUAGE_MAP = {
    "ar": "ar_eg",
    "bn": "bn_in",
    "de": "de_de",
    "en": "en_us",
    "es": "es_419",
    "fr": "fr_fr",
    "gu": "gu_in",
    "hi": "hi_in",
    "it": "it_it",
    "ja": "ja_jp",
    "kn": "kn_in",
    "ko": "ko_kr",
    "ml": "ml_in",
    "mr": "mr_in",
    "or": "or_in",
    "pa": "pa_in",
    "ru": "ru_ru",
    "ta": "ta_in",
    "te": "te_in",
    "th": "th_th",
    "zh": "cmn_hans_cn",
}
# Sanskrit is in the OCR corpus and not in FLEURS, so the overlap is 21, not 22.
OCR_ONLY = ("sa",)


def overlap_languages(available):
    """FLEURS codes that correspond to an OCR corpus language, in OCR order."""
    missing = [
        iso for iso, code in OCR_LANGUAGE_MAP.items() if code not in set(available)
    ]
    if missing:
        raise ValueError(f"FLEURS lacks mapped languages: {missing}")
    return [OCR_LANGUAGE_MAP[iso] for iso in sorted(OCR_LANGUAGE_MAP)]


# A shard is ~310 MB and fsspec's default HTTP timeout assumes small objects, so a dozen
# concurrent whole-shard reads time out rather than finish. Verification is bandwidth
# bound anyway: more streams do not make it faster, they make it fail.
VERIFY_TIMEOUT_SECONDS = 1800
VERIFY_WORKERS = 4


def http_filesystem(token=None):
    import aiohttp
    import fsspec
    from huggingface_hub import get_token

    token = token or get_token()
    return fsspec.filesystem(
        "http",
        client_kwargs={
            "headers": {"Authorization": f"Bearer {token}"} if token else {},
            "timeout": aiohttp.ClientTimeout(total=VERIFY_TIMEOUT_SECONDS),
        },
    )


def allocate(languages, families, size):
    """Per-(language, family) counts totalling exactly ``size``.

    Families are held equal so per-family means carry the same sample size, and the
    within-family remainder rotates across families so no language is always short.
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
            rotated = (offset - index * remainder) % len(languages)
            counts[language, family] = base + (rotated < remainder)
    if sum(counts.values()) != size:
        raise ValueError("Allocation does not total the requested size")
    return counts


def _order(seed, language, sample_id):
    return digest(["fleurs-evalset-v1", seed, language, int(sample_id)])


def eligible_rows(rows, language):
    """Rows this environment would actually serve, decided from metadata alone."""
    keep = []
    for row in rows:
        duration = (row.get("num_samples") or 0) / SAMPLING_RATE
        if not MIN_SECONDS <= duration <= MAX_SECONDS:
            continue
        if row.get("id") is None:
            continue
        keep.append(row)
    return keep


def read_metadata(language, split, source_root=None, token=None):
    """Read one language/split's metadata columns, never its audio."""
    import pyarrow.parquet as pq

    from .prepare import bucket_shards, mounted_shards

    if source_root:
        paths = mounted_shards(source_root, language, split)
        opener = lambda path: path  # noqa: E731 - pyarrow takes the path directly
    else:
        fs = http_filesystem(token)
        paths = [
            f"https://huggingface.co/buckets/FineEnvs/fleurs-bucket/resolve/{p}"
            for p in bucket_shards(language, split)
        ]
        opener = fs.open

    rows = []
    for path in paths:
        handle = opener(path)
        try:
            table = pq.ParquetFile(handle).read(columns=METADATA_COLUMNS)
        finally:
            if hasattr(handle, "close"):
                handle.close()
        rows.extend({**row, "split": split} for row in table.to_pylist())
    return rows


def select(metadata, languages, families, size, seed=42, revision=DEFAULT_REVISION):
    """Choose fixed task IDs, balanced by family and even across languages."""
    counts = allocate(list(languages), list(families), size)
    selection = []
    for language in languages:
        rows = sorted(
            eligible_rows(metadata[language], language),
            key=lambda row: _order(seed, language, row["id"]),
        )
        if not rows:
            raise ValueError(f"{language}: no eligible utterances")
        for index, family in enumerate(families):
            want = counts[language, family]
            # Families take disjoint bands of the hash order, so one bad recording cannot
            # dominate a language across every task it appears in. The band is fixed by
            # family position rather than by how many tasks are wanted, which makes a
            # larger set a superset of a smaller one: the 102-language and overlap sets
            # therefore agree on every task they share instead of sampling independently.
            band = [
                row
                for position, row in enumerate(rows)
                if position % len(families) == index
            ]
            if len(band) < want:
                raise ValueError(
                    f"{language}/{family}: {len(band)} eligible utterances, need {want}"
                )
            chosen = band[:want]
            for row in chosen:
                reference = (
                    language
                    if family == "language_id"
                    else row.get(
                        "transcription"
                        if family == "transcription"
                        else "raw_transcription"
                    )
                )
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError(
                        f"{language}/{family}: utterance {row['id']} has no reference"
                    )
                selection.append(
                    {
                        "task_id": task_id(
                            revision, language, row["split"], row["id"], family
                        ),
                        "language": language,
                        "language_name": row.get("language") or language,
                        "family": family,
                        "split": row["split"],
                        "sample_id": int(row["id"]),
                        "num_samples": int(row["num_samples"]),
                        "duration_seconds": round(
                            row["num_samples"] / SAMPLING_RATE, 3
                        ),
                        "reference_sha256": hashlib.sha256(
                            reference.encode()
                        ).hexdigest(),
                    }
                )
    selection.sort(key=lambda e: (e["language"], e["family"], e["sample_id"]))
    return selection


def record(
    selection,
    *,
    name,
    languages,
    families,
    split,
    size,
    seed,
    validated,
    revision=DEFAULT_REVISION,
    failures=(),
):
    body = {
        "evalset_version": EVALSET_VERSION,
        "name": name,
        "source": REPO_ID,
        "revision": revision,
        "split": split,
        "seed": seed,
        "size": len(selection),
        "requested_size": size,
        "validated": validated,
        "languages": list(languages),
        "families": list(families),
        "failures": list(failures),
        "tasks": selection,
    }
    body["evalset_id"] = hashlib.sha256(
        canonical_json(
            [
                EVALSET_VERSION,
                REPO_ID,
                revision,
                split,
                seed,
                [e["task_id"] for e in selection],
            ]
        ).encode()
    ).hexdigest()
    return body


def save(body, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    return path


def load(path):
    body = json.loads(Path(path).read_text())
    if body.get("evalset_version") != EVALSET_VERSION:
        raise ValueError("Rebuild this evaluation set for the current version")
    if body.get("failures"):
        raise ValueError(
            f"{len(body['failures'])} recorded task(s) did not verify; rebuild the set"
        )
    tasks = body["tasks"]
    expected = hashlib.sha256(
        canonical_json(
            [
                EVALSET_VERSION,
                body["source"],
                body["revision"],
                body["split"],
                body["seed"],
                [e["task_id"] for e in tasks],
            ]
        ).encode()
    ).hexdigest()
    if expected != body.get("evalset_id"):
        raise ValueError("Evaluation set identity does not match its task IDs")
    if len(tasks) != body["size"] or len({e["task_id"] for e in tasks}) != len(tasks):
        raise ValueError("Evaluation set size or task uniqueness is inconsistent")
    return body


def summarize(body):
    from collections import Counter

    return {
        "name": body["name"],
        "evalset_id": body["evalset_id"],
        "size": body["size"],
        "validated": body["validated"],
        "languages": len(body["languages"]),
        "failures": len(body["failures"]),
        "per_family": dict(sorted(Counter(e["family"] for e in body["tasks"]).items())),
        "per_language_range": [
            min(Counter(e["language"] for e in body["tasks"]).values()),
            max(Counter(e["language"] for e in body["tasks"]).values()),
        ],
    }


def verify_audio(languages, selection, split, source_root=None, workers=VERIFY_WORKERS):
    """Decode every selected utterance. Costs the full shard per language."""
    import pyarrow.parquet as pq

    from ..training import decode_audio
    from .prepare import bucket_shards, mounted_shards

    wanted = {}
    for entry in selection:
        wanted.setdefault(entry["language"], set()).add(entry["sample_id"])
    failures = []

    def check(language):
        found = {}
        if source_root:
            paths = mounted_shards(source_root, language, split)
            tables = (pq.ParquetFile(p).read(columns=["id", "audio"]) for p in paths)
        else:
            fs = http_filesystem()

            def tables_from_bucket():
                for path in bucket_shards(language, split):
                    url = f"https://huggingface.co/buckets/FineEnvs/fleurs-bucket/resolve/{path}"
                    with fs.open(url) as handle:
                        yield pq.ParquetFile(handle).read(columns=["id", "audio"])

            tables = tables_from_bucket()
        for table in tables:
            for row in table.to_pylist():
                if row["id"] in wanted[language]:
                    found[row["id"]] = (row.get("audio") or {}).get("bytes")
        results = []
        for sample_id in sorted(wanted[language]):
            raw = found.get(sample_id)
            try:
                if not isinstance(raw, bytes):
                    raise ValueError("missing audio bytes")
                decode_audio(raw)
                results.append((sample_id, hashlib.sha256(raw).hexdigest(), None))
            except Exception as error:
                results.append((sample_id, None, f"{type(error).__name__}: {error}"))
        return language, results

    hashes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for language, results in pool.map(check, sorted(wanted)):
            for sample_id, sha, error in results:
                if error:
                    failures.append(
                        {"language": language, "sample_id": sample_id, "error": error}
                    )
                else:
                    hashes[language, sample_id] = sha
            print(f"  verified {language}: {len(results)} utterances", flush=True)
    for entry in selection:
        sha = hashes.get((entry["language"], entry["sample_id"]))
        if sha:
            entry["audio_sha256"] = sha
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("build", help="Select and freeze both evaluation sets")
    make.add_argument("--output-dir", type=Path, required=True)
    make.add_argument("--split", default=DEFAULT_SPLIT)
    make.add_argument("--size", type=int, default=510)
    make.add_argument("--overlap-size", type=int, default=504)
    make.add_argument("--families", nargs="+", default=list(FAMILIES))
    make.add_argument("--seed", type=int, default=42)
    make.add_argument("--source-root")
    make.add_argument("--workers", type=int, default=8)
    make.add_argument(
        "--verify-workers",
        type=int,
        default=VERIFY_WORKERS,
        help="Whole-shard reads are bandwidth bound; more streams time out, not finish",
    )
    make.add_argument(
        "--verify",
        action="store_true",
        help="Also decode every selected utterance; reads the full shard per language",
    )

    check = sub.add_parser("verify", help="Validate a frozen set")
    check.add_argument("--evalset", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(summarize(load(args.evalset)), indent=2))
        return

    from .prepare import BUCKET_ID

    if args.source_root:
        languages = sorted(
            p.name
            for p in (Path(args.source_root) / "parquet-data").iterdir()
            if p.is_dir() and p.name != "all"
        )
    else:
        from huggingface_hub import HfApi

        languages = sorted(
            {
                entry.path.split("/")[1]
                for entry in HfApi().list_bucket_tree(
                    BUCKET_ID, "parquet-data/", recursive=False
                )
                if entry.path.count("/") >= 1 and entry.path.split("/")[1] != "all"
            }
        )
    languages = [lang for lang in languages if lang]
    print(f"{len(languages)} FLEURS languages", flush=True)

    # One metadata pass serves both sets; the reads are independent, so overlap them.
    metadata = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(read_metadata, lang, args.split, args.source_root): lang
            for lang in languages
        }
        for future in concurrent.futures.as_completed(futures):
            language = futures[future]
            metadata[language] = future.result()
            print(f"  {language}: {len(metadata[language])} utterances", flush=True)

    families = list(args.families)
    overlap = overlap_languages(languages)
    sets = {
        "fleurs-all": (
            languages,
            args.size,
            select(metadata, languages, families, args.size, args.seed),
        ),
        "fleurs-ocr-overlap": (
            overlap,
            args.overlap_size,
            select(metadata, overlap, families, args.overlap_size, args.seed),
        ),
    }

    # The sets nest, so verify the union in one pass: reading a language's shard twice
    # would double the expensive part of the build for no additional guarantee.
    shared_failures = []
    if args.verify:
        union, seen = [], set()
        for _, _, selection in sets.values():
            for entry in selection:
                key = (entry["language"], entry["sample_id"])
                if key not in seen:
                    seen.add(key)
                    union.append(entry)
        print(
            f"verifying {len(union)} utterances across {len({e['language'] for e in union})} "
            "languages (reads the full shard per language)",
            flush=True,
        )
        shared_failures = verify_audio(
            languages, union, args.split, args.source_root, args.verify_workers
        )
        verified = {
            (e["language"], e["sample_id"]): e["audio_sha256"]
            for e in union
            if e.get("audio_sha256")
        }
        for _, _, selection in sets.values():
            for entry in selection:
                sha = verified.get((entry["language"], entry["sample_id"]))
                if sha:
                    entry["audio_sha256"] = sha

    outputs = {}
    for name, (langs, size, selection) in sets.items():
        chosen = {(e["language"], e["sample_id"]) for e in selection}
        failures = [
            f for f in shared_failures if (f["language"], f["sample_id"]) in chosen
        ]
        body = record(
            selection,
            name=name,
            languages=langs,
            families=families,
            split=args.split,
            size=size,
            seed=args.seed,
            validated=bool(args.verify) and not failures,
            failures=failures,
        )
        path = save(body, args.output_dir / f"eval-{name}.json")
        outputs[name] = summarize(body)
        print(f"wrote {path}", flush=True)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
