"""Sarvam Indic OCR Bench, served as evaluation splits beside the Nayana corpus.

6,909 human-reviewed text-block crops in 23 languages, from
https://huggingface.co/datasets/sarvamai/indic-ocr-bench (Apache-2.0, Sarvam AI).

The benchmark is evaluation-only by construction: its family is not in FAMILIES, so no
training sampler can draw it, and its tasks look to an agent exactly like section OCR -
the same observation fields, the same prompt, one text answer - so a model trained on
the corpus is measured here without learning a new interface.
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
SNAPSHOT_ID = f"{REPO}@{REVISION[:12]}"
FAMILY = "indic_ocr_bench"
PREFIX = "indic-ocr-bench"

# Served split -> (directory in the dataset repo, row count at the pinned revision).
# Counts are known in advance so a manifest can list the splits without downloading
# 730 MB; the first real load checks them and refuses a revision that disagrees.
SPLITS = {
    "indic_ocr_bench_test": ("test", 6909),
    "indic_ocr_bench_small": ("small_representative", 1173),
}

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

MIMES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def prompt(language):
    # Word for word the section-OCR prompt: the point is to measure the same skill.
    return (
        f"Transcribe the text in this cropped document region (language: {language}). "
        "Return only the text, preserving its language and punctuation."
    )


def download_split(directory):
    """Fetch one split's parquet files at the pinned revision; returns local paths."""
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        f
        for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
        if f.startswith(f"{directory}/") and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"{REPO}@{REVISION[:12]} has no {directory} parquet")
    return [
        hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION) for f in files
    ]


class BenchCatalog:
    """The Nayana catalog interface, for a source that is one fixed set of crops."""

    def __init__(self, cache_dir, fetch=download_split):
        self.root = Path(cache_dir) / PREFIX / REVISION[:12]
        self.fetch = fetch
        self._lock = threading.Lock()
        self._rows = {}  # served split -> [task, ...]
        self._by_id = {}  # task_id -> task

    # -- ownership, so a composite catalog can route requests -------------------------

    def splits(self):
        return list(SPLITS)

    def owns_split(self, split):
        return split in SPLITS

    def owns_task(self, task_id):
        return isinstance(task_id, str) and task_id.startswith(f"{PREFIX}.")

    @property
    def languages(self):
        return sorted(set(LANGUAGES.values()))

    def eval_ids(self):
        return {name: f"{SNAPSHOT_ID}:{directory}" for name, (directory, _) in SPLITS.items()}

    # -- loading ----------------------------------------------------------------------

    def _task_id(self, split, image_name):
        return f"{PREFIX}.{REVISION[:12]}.{split}.{image_name}"

    def _load(self, split):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        if split in self._rows:
            return self._rows[split]
        with self._lock:
            if split not in self._rows:
                rows = self._read_index(split) or self._build(split)
                self._rows[split] = rows
                self._by_id.update((t["task_id"], t) for t in rows)
        return self._rows[split]

    def _index_path(self, split):
        return self.root / f"{split}.json"

    def _asset_path(self, sha):
        return self.root / "assets" / f"{sha}"

    def _read_index(self, split):
        path = self._index_path(split)
        if not path.exists():
            return None
        rows = json.loads(path.read_text())
        _, expected = SPLITS[split]
        # A partial or stale index is rebuilt, never served.
        if len(rows) != expected or not all(
            self._asset_path(t["asset_sha256"]).exists() for t in rows
        ):
            return None
        return rows

    def _build(self, split):
        import pyarrow.parquet as pq

        directory, expected = SPLITS[split]
        (self.root / "assets").mkdir(parents=True, exist_ok=True)
        rows, seen = [], set()
        for path in self.fetch(directory):
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=64, columns=["image", "image_name", "gt", "language"]
            ):
                for record in batch.to_pylist():
                    rows.append(self._extract(split, record))
                    if rows[-1]["task_id"] in seen:
                        raise ValueError(f"Duplicate image_name {record['image_name']!r}")
                    seen.add(rows[-1]["task_id"])
        if len(rows) != expected:
            raise ValueError(
                f"{REPO}@{REVISION[:12]} {directory} has {len(rows)} rows, expected "
                f"{expected}; refusing to serve a split that is not the pinned one"
            )
        tmp = self._index_path(split).with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False))
        os.replace(tmp, self._index_path(split))
        return rows

    def _extract(self, split, record):
        image = record["image"] or {}
        raw = image.get("bytes")
        if not raw:
            raise ValueError(f"{record['image_name']}: image has no embedded bytes")
        name = record["language"]
        if name not in LANGUAGES:
            raise ValueError(f"{record['image_name']}: unknown language {name!r}")
        with Image.open(io.BytesIO(raw)) as decoded:
            width, height = decoded.size
            mime = MIMES.get(decoded.format)
        if mime is None:
            raise ValueError(f"{record['image_name']}: unsupported image format")
        sha = hashlib.sha256(raw).hexdigest()
        asset = self._asset_path(sha)
        if not asset.exists():
            tmp = asset.with_suffix(".tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, asset)
        language = LANGUAGES[name]
        image_name = record["image_name"]
        return {
            "task_id": self._task_id(split, image_name),
            "split": split,
            "language": language,
            "language_name": name,
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

    # -- the catalog interface --------------------------------------------------------

    def count(self, split):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        return len(self._rows[split]) if split in self._rows else SPLITS[split][1]

    def at(self, split, index):
        rows = self._load(split)
        if not 0 <= index < len(rows):
            raise IndexError("Task index outside split")
        return rows[index]

    def get(self, task_id):
        if not self.owns_task(task_id):
            raise KeyError("Not an Indic OCR Bench task")
        parts = task_id.split(".", 3)
        if len(parts) != 4 or parts[1] != REVISION[:12] or parts[2] not in SPLITS:
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
        # Only the shared public fields: the reference never leaves the server.
        result = {key: task.get(key) for key in PUBLIC_FIELDS}
        result.update(
            snapshot_id=SNAPSHOT_ID,
            media_ready=True,
            block_id=PREFIX,
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
        raw = self._asset_path(sha).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Cached image checksum mismatch")
        return raw, task["mime"]

    def image(self, task):
        raw, _ = self.asset_bytes(task["asset_sha256"], task["task_id"])
        with Image.open(io.BytesIO(raw)) as image:
            return image.copy()

    def close(self):
        pass
