"""Sarvam Indic OCR Bench, served as evaluation splits beside the Nayana corpus.

6,909 human-reviewed text-block crops in 23 languages, from
https://huggingface.co/datasets/sarvamai/indic-ocr-bench (Apache-2.0, Sarvam AI).

Each crop is served as an ordinary section-OCR task: the same prompt, the same single
text answer, the same reward and the same observation as a corpus region, so a model is
measured on the benchmark exactly as it is trained and scored everywhere else. What
keeps the benchmark out of training is the split, not the task: its tasks exist only
in the indic_ocr_bench_* splits, and no sampler ever draws from those.

Storage mirrors the corpus. The crops and index are built once and published to a
bucket; a deployment mounts it read-only and reads in place, and anything else fetches
from it on first use. Building from the pinned dataset revision is only a fallback.
"""

import hashlib
import io
import json
import os
import re
import threading
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from .catalog import PUBLIC_FIELDS

REPO = "sarvamai/indic-ocr-bench"
REVISION = "84ce7ce447456a92bcbf25f3c0a55d6a5a44a24b"
VERSION = REVISION[:12]
SNAPSHOT_ID = f"{REPO}@{VERSION}"
SOURCE = "indic-ocr-bench"
BUCKET = "FineEnvs/indic-ocr-bench-bucket"
FAMILY = "section_ocr"  # deliberately the corpus family: same task, reward and serving
LICENSE = "apache-2.0"
CITATION = """@misc{sarvam-indic-ocr-bench,
  title={Sarvam Indic OCR Bench},
  author={Sarvam AI},
  year={2026},
  url={https://huggingface.co/datasets/sarvamai/indic-ocr-bench}
}"""

# Served split -> (directory in the dataset repo, total rows at the pinned revision).
# Counts are known in advance so a manifest can list the splits without fetching
# anything; a build or load whose count disagrees is refused rather than served.
SPLITS = {
    "indic_ocr_bench_test": ("test", 6909),
    "indic_ocr_bench_small": ("small_representative", 1173),
}

# Rows the pinned revision ships with no image at all - the crop is absent from the
# source data, so the row cannot be answered and is not served. A served model would
# only score zero on it. Pinned by name, so a row gaining or losing its image upstream
# is caught at build time rather than silently absorbed into a changed benchmark.
EXCLUDED = {
    "indic_ocr_bench_test": (),
    "indic_ocr_bench_small": (),
}


def served_count(split):
    return SPLITS[split][1] - len(EXCLUDED[split])


def read_index(path):
    """Tasks from an index file; the excluded rows travel alongside for provenance."""
    data = json.loads(Path(path).read_text())
    return data["tasks"] if isinstance(data, dict) else data

# The dataset names languages in English; these are the codes its own card lists -
# ISO 639-1 where one exists, ISO 639-3 otherwise. An unknown name is an error rather
# than a guess, because a mislabelled language silently corrupts a per-language score.
LANGUAGES = {
    "Assamese": "as",
    "Bengali": "bn",
    "Bodo": "brx",
    "Dogri": "doi",
    "English": "en",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Kashmiri": "ks",
    "Konkani": "kok",
    "Maithili": "mai",
    "Malayalam": "ml",
    "Manipuri": "mni",
    "Marathi": "mr",
    "Nepali": "ne",
    "Odia": "or",
    "Punjabi": "pa",
    "Sanskrit": "sa",
    "Santhali": "sat",
    "Sindhi": "sd",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
}
LANGUAGE_NAMES = {code: name for name, code in LANGUAGES.items()}

MIMES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def is_bench_task(task_id):
    return isinstance(task_id, str) and task_id.startswith(f"{SOURCE}.")


def prompt(language):
    # Word for word the section-OCR prompt: the point is to measure the same skill.
    return (
        f"Transcribe the text in this cropped document region (language: {language}). "
        "Return only the text, preserving its language and punctuation."
    )


def index_path(root, split):
    return Path(root) / VERSION / f"{split}.json"


def asset_path(root, sha):
    return Path(root) / VERSION / "assets" / sha


def download_split(directory):
    """Fetch one split's parquet files at the pinned dataset revision."""
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        f
        for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
        if f.startswith(f"{directory}/") and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"{SNAPSHOT_ID} has no {directory} parquet")
    return [
        hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION) for f in files
    ]


def _write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def build_split(split, root, fetch=download_split, discover=False):
    """Extract one split from the dataset into root: content-addressed crops + index.

    This is what publishing runs, once, to fill the bucket. Serving never calls it
    unless the bucket is unavailable. discover=True reports imageless rows instead of
    checking them against EXCLUDED - for pinning them, never for serving.
    """
    import pyarrow.parquet as pq

    directory, total = SPLITS[split]
    rows, excluded, seen, read = [], [], set(), 0
    for path in fetch(directory):
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=64, columns=["image", "image_name", "gt", "language"]
        ):
            for record in batch.to_pylist():
                read += 1
                if record["image_name"] in seen:
                    raise ValueError(f"Duplicate image_name {record['image_name']!r}")
                seen.add(record["image_name"])
                if not (record["image"] or {}).get("bytes"):
                    excluded.append({
                        "image_name": record["image_name"],
                        "language": LANGUAGES.get(record["language"], record["language"]),
                        "reason": "no image in the source row",
                    })
                    continue
                rows.append(_extract(split, record, root))
    if read != total:
        raise ValueError(
            f"{SNAPSHOT_ID} {directory} has {read} rows, expected {total}; "
            "refusing to serve a split that is not the pinned one"
        )
    found = sorted(e["image_name"] for e in excluded)
    if not discover and found != sorted(EXCLUDED[split]):
        raise ValueError(
            f"{split}: rows without an image are {found}, pinned "
            f"{sorted(EXCLUDED[split])}; the source changed - re-pin before serving"
        )
    body = {"total_rows": read, "tasks": rows, "excluded": excluded}
    _write_atomic(index_path(root, split), json.dumps(body, ensure_ascii=False).encode())
    return rows


def _extract(split, record, root):
    image = record["image"] or {}
    raw = image.get("bytes")
    if not raw:
        raise ValueError(f"{record['image_name']}: image has no embedded bytes")
    name = record["language"]
    if name not in LANGUAGES:
        raise ValueError(f"{record['image_name']}: unknown language {name!r}")
    with Image.open(io.BytesIO(raw)) as decoded:
        width, height = decoded.size
        # The card says PNG; at the pinned revision some crops are JPEG. Read the bytes.
        mime = MIMES.get(decoded.format)
    if mime is None:
        raise ValueError(f"{record['image_name']}: unsupported image format")
    sha = hashlib.sha256(raw).hexdigest()
    if not asset_path(root, sha).exists():
        _write_atomic(asset_path(root, sha), raw)
    language = LANGUAGES[name]
    image_name = record["image_name"]
    return {
        "task_id": f"{SOURCE}.{VERSION}.{split}.{image_name}",
        "split": split,
        "language": language,
        "family": FAMILY,
        "unit": image_name,
        "page_id": image_name,
        "document_id": image_name,
        "prompt": prompt(language),
        "reference": record["gt"],
        "asset_sha256": sha,
        "mime": mime,
        "width": width,
        "height": height,
        "bbox": None,
        "page_width": width,
        "page_height": height,
    }


class BenchCatalog:
    """The Nayana catalog interface, for a source that is one fixed set of crops.

    root: a mounted copy of the bucket, read in place (a Space or a job).
    Otherwise each split is fetched from `bucket` into cache_dir on first use - index and
    every crop in one batch, since an evaluation visits them all - and, failing that,
    built from the pinned dataset revision when `fallback` allows it.
    """

    def __init__(self, cache_dir, *, root=None, bucket=BUCKET, fetch=download_split,
                 fallback=True):
        self.root = Path(root) if root else None
        self.cache = Path(cache_dir) / SOURCE
        self.bucket = bucket
        self.fetch = fetch
        self.fallback = fallback
        self._lock = threading.Lock()
        self._rows = {}
        self._by_id = {}

    # -- ownership, so a composite catalog can route requests -------------------------

    def splits(self):
        return list(SPLITS)

    def owns_split(self, split):
        return split in SPLITS

    def owns_task(self, task_id):
        return is_bench_task(task_id)

    @property
    def languages(self):
        return sorted(LANGUAGE_NAMES)

    def eval_ids(self):
        return {name: f"{SNAPSHOT_ID}:{directory}" for name, (directory, _) in SPLITS.items()}

    @property
    def storage(self):
        return str(self.root) if self.root else f"hf://buckets/{self.bucket} -> {self.cache}"

    # -- loading ----------------------------------------------------------------------

    def _load(self, split):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        if split not in self._rows:
            with self._lock:
                if split not in self._rows:
                    rows = self._resolve(split)
                    expected = served_count(split)
                    if len(rows) != expected:
                        raise ValueError(
                            f"{split} index has {len(rows)} tasks, expected {expected}"
                        )
                    self._rows[split] = rows
                    self._by_id.update((t["task_id"], t) for t in rows)
        return self._rows[split]

    def _resolve(self, split):
        if self.root:
            path = index_path(self.root, split)
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} is missing: publish {SNAPSHOT_ID} to the mounted bucket"
                )
            return read_index(path)
        local = index_path(self.cache, split)
        if local.exists():
            rows = read_index(local)
            if all(asset_path(self.cache, t["asset_sha256"]).exists() for t in rows):
                return rows
        try:
            return self._pull(split)
        except Exception as error:  # noqa: BLE001 - any bucket failure may fall back
            if not self.fallback:
                raise
            print(f"{SOURCE}: bucket unavailable ({error}); building {split} from "
                  f"{SNAPSHOT_ID}", flush=True)
            return build_split(split, self.cache, self.fetch)

    def _pull(self, split):
        from huggingface_hub import HfApi

        api = HfApi()
        remote = f"{VERSION}/{split}.json"
        local = index_path(self.cache, split)
        local.parent.mkdir(parents=True, exist_ok=True)
        # The client skips missing files silently by default; a missing index must be
        # an error here, so the fallback runs instead of a partial split being served.
        api.download_bucket_files(
            self.bucket, [(remote, str(local))], raise_on_missing_files=True
        )
        rows = read_index(local)
        assets = self.cache / VERSION / "assets"
        have = {p.name for p in assets.glob("*")} if assets.exists() else set()
        missing = sorted({t["asset_sha256"] for t in rows} - have)
        if missing:
            assets.mkdir(parents=True, exist_ok=True)
            api.download_bucket_files(
                self.bucket,
                [(f"{VERSION}/assets/{sha}", str(asset_path(self.cache, sha))) for sha in missing],
                raise_on_missing_files=True,
            )
        return rows

    def _asset_file(self, sha):
        return asset_path(self.root or self.cache, sha)

    # -- the catalog interface --------------------------------------------------------

    def count(self, split):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        return len(self._rows[split]) if split in self._rows else served_count(split)

    def at(self, split, index):
        rows = self._load(split)
        if not 0 <= index < len(rows):
            raise IndexError("Task index outside split")
        return rows[index]

    def get(self, task_id):
        if not self.owns_task(task_id):
            raise KeyError("Not an Indic OCR Bench task")
        parts = task_id.split(".", 3)
        if len(parts) != 4 or parts[1] != VERSION or parts[2] not in SPLITS:
            raise KeyError("Task belongs to another benchmark revision or split")
        self._load(parts[2])
        try:
            return self._by_id[task_id]
        except KeyError:
            raise KeyError("No such benchmark task") from None

    def task_range(self, split, start=0, stop=None):
        rows = self._load(split)
        start, stop = 0 if start is None else start, len(rows) if stop is None else stop
        if not 0 <= start <= stop <= len(rows) or stop - start > 1000:
            raise IndexError("Use an in-bounds range of at most 1000 tasks")
        return [self.public(t) for t in rows[start:stop]]

    def _group(self, split, language, family):
        return [
            t for t in self._load(split) if t["language"] == language and t["family"] == family
        ]

    def group_count(self, split, language, family):
        return len(self._group(split, language, family))

    def group_at(self, split, language, family, index):
        group = self._group(split, language, family)
        if not 0 <= index < len(group):
            raise IndexError("Task index outside language/task group")
        return group[index]

    def group_position(self, task_id, split, language, family):
        ids = [t["task_id"] for t in self._group(split, language, family)]
        return ids.index(task_id) if task_id in ids else None

    def public(self, task):
        # Exactly the corpus's public fields; the reference never leaves the server.
        result = {key: task.get(key) for key in PUBLIC_FIELDS}
        result.update(
            snapshot_id=SNAPSHOT_ID,
            media_ready=True,
            block_id=SOURCE,
            asset_path=f"/assets/{task['asset_sha256']}?task_id={quote(task['task_id'])}",
        )
        return result

    def materialize(self, task):
        return task

    def asset_bytes(self, sha, task_id):
        if not re.fullmatch(r"[0-9a-f]{64}", sha or "") or not task_id:
            raise KeyError("A benchmark task ID and valid asset hash are required")
        task = self.get(task_id)
        if task["asset_sha256"] != sha:
            raise KeyError("Asset does not match this task")
        raw = self._asset_file(sha).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Benchmark image checksum mismatch")
        return raw, task["mime"]

    def image(self, task):
        raw, _ = self.asset_bytes(task["asset_sha256"], task["task_id"])
        with Image.open(io.BytesIO(raw)) as image:
            return image.copy()

    def prefetch(self, *, task_ids=(), block_ids=()):
        # A split's crops arrive together on first use, so there is nothing to warm.
        return {"task_ids": 0, "block_ids": 0}

    def close(self):
        pass
