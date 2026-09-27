"""Text-rendering tasks derived from leffff/Diffusion-Reward-Modeling-for-Text-Rendering-Dataset.

Derivation (`build_tasks`, run by train/publish_dataset.py): the source CSVs are read at a
pinned revision; prompts screened as sexual, vulgar, hateful, violent, self-harm or drug
content (exclusions.json, train/screen_prompts.py) are dropped from every split; the source
test split is served as `test`; the source train split is de-duplicated by prompt, stripped
of prompts that also occur in test, and hashed 96/4 into `train`/`validation`. Task IDs are
content hashes, so a replayed ID always means the same prompt and target.

Serving (`configured_catalog`) reads the published result, PUBLISHED_DATASET at
PUBLISHED_REVISION, or a local copy of it (IMAGE_TEXT_GEN_DATA_DIR, the Space's mount).
IMAGE_TEXT_GEN_SOURCE_DIR instead derives tasks from local leffff-format CSVs (tests, smoke).
"""

import csv
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

DATASET = "leffff/Diffusion-Reward-Modeling-for-Text-Rendering-Dataset"
REVISION = "0ec4bce48b13570f95609a32fe72c38c40b39c33"
SOURCE_LICENSE = "MIT"
SOURCE_FILES = {
    "train": "data_with_ocr_reward_train.csv",
    "test": "data_with_ocr_reward_test.csv",
}
PUBLISHED_DATASET = "AdithyaSK/image-text-gen-rl-prompts"
PUBLISHED_REVISION = "b8d641390af9827237fc273ca75bad09c5b00d27"
EXCLUSIONS = Path(__file__).with_name("exclusions.json")
SPLITS = ("train", "validation", "test")
FAMILY = "text_render"
SCHEMA_VERSION = "image-text-gen-tasks-v1"
VALIDATION_PERCENT = 4
BASELINE_COLUMNS = tuple(f"v{i}_qwen_ocr_levenstein_score" for i in range(1, 6))
PUBLIC_FIELDS = (
    "task_id",
    "split",
    "family",
    "prompt",
    "target_text",
    "text_len",
    "baseline_ocr",
)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _bucket(source_id):
    return int(digest([SCHEMA_VERSION, REVISION, "split", source_id])[:8], 16) % 100


def _task(row, split):
    source_id = str(row["id"])
    target = row["text"]
    scores = [float(row[c]) for c in BASELINE_COLUMNS if row.get(c) not in (None, "")]
    return {
        "task_id": "itg-" + digest([SCHEMA_VERSION, REVISION, source_id, row["prompt"], target])[:20],
        "split": split,
        "family": FAMILY,
        "prompt": row["prompt"],
        "target_text": target,
        "text_len": len(target),
        "baseline_ocr": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "source_id": source_id,
    }


def load_exclusions(path=EXCLUSIONS):
    if not Path(path).is_file():
        return frozenset()
    return frozenset(json.loads(Path(path).read_text(encoding="utf-8"))["excluded"])


def build_tasks(train_rows, test_rows, excluded=frozenset()):
    """Derive the three served splits from the two source splits."""
    excluded = {str(source_id) for source_id in excluded}
    train_rows = [row for row in train_rows if str(row["id"]) not in excluded]
    test_rows = [row for row in test_rows if str(row["id"]) not in excluded]
    test = [_task(row, "test") for row in test_rows if row["text"].strip()]
    test_prompts = {task["prompt"] for task in test}
    seen, train, validation = set(), [], []
    for row in train_rows:
        if not row["text"].strip() or row["prompt"] in test_prompts or row["prompt"] in seen:
            continue
        seen.add(row["prompt"])
        if _bucket(str(row["id"])) < VALIDATION_PERCENT:
            validation.append(_task(row, "validation"))
        else:
            train.append(_task(row, "train"))
    return {"train": train, "validation": validation, "test": test}


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_paths(source_dir=None):
    """Local CSVs from IMAGE_TEXT_GEN_SOURCE_DIR, else the pinned Hub revision."""
    source_dir = source_dir or os.environ.get("IMAGE_TEXT_GEN_SOURCE_DIR")
    if source_dir:
        return {split: Path(source_dir) / name for split, name in SOURCE_FILES.items()}
    from huggingface_hub import hf_hub_download

    return {
        split: Path(
            hf_hub_download(DATASET, name, repo_type="dataset", revision=REVISION)
        )
        for split, name in SOURCE_FILES.items()
    }


class Catalog:
    def __init__(self, splits, source=DATASET, revision=REVISION):
        self._splits = {name: list(splits.get(name, [])) for name in SPLITS}
        self._by_id = {task["task_id"]: task for rows in self._splits.values() for task in rows}
        if len(self._by_id) != sum(len(rows) for rows in self._splits.values()):
            raise ValueError("Duplicate task IDs across splits")
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "source": source,
            "revision": revision,
            "source_license": SOURCE_LICENSE,
            "family": FAMILY,
            "counts": {name: len(rows) for name, rows in self._splits.items()},
        }
        # Content identity: the same tasks in the same splits give the same ID wherever
        # the source files were read from.
        self.manifest["catalog_id"] = digest(
            [SCHEMA_VERSION, revision, {n: [t["task_id"] for t in r] for n, r in self._splits.items()}]
        )[:16]

    @classmethod
    def from_source(cls, source_dir=None, excluded=None):
        """Derive tasks from leffff-format CSVs (the pinned Hub revision by default).

        The shipped exclusions are keyed by leffff row IDs, so by default they apply only
        to the pinned source, not to arbitrary local CSVs such as test fixtures.
        """
        paths = source_paths(source_dir)
        if excluded is None:
            excluded = frozenset() if source_dir else load_exclusions()
        splits = build_tasks(_read_csv(paths["train"]), _read_csv(paths["test"]), excluded)
        return cls(splits, source=DATASET)

    @classmethod
    def from_published(cls, data_dir=None, repo=PUBLISHED_DATASET, revision=PUBLISHED_REVISION):
        """Load the published JSONL splits from a local directory or the pinned Hub revision."""
        splits = {}
        for split in SPLITS:
            if data_dir:
                path = Path(data_dir) / f"{split}.jsonl"
            else:
                from huggingface_hub import hf_hub_download

                path = Path(hf_hub_download(repo, f"data/{split}.jsonl", repo_type="dataset",
                                            revision=revision))
            with open(path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            splits[split] = [{**{k: row[k] for k in PUBLIC_FIELDS}, "source_id": row["source_id"]}
                             for row in rows]
        return cls(splits, source=repo)

    def rows(self, split):
        """Full task records for publishing (public fields plus the source id)."""
        return list(self._splits[split])

    def splits(self):
        return [name for name in SPLITS if self._splits[name]]

    def count(self, split):
        if split not in self._splits:
            raise ValueError(f"Unknown split {split!r}; choose one of {', '.join(SPLITS)}")
        return len(self._splits[split])

    def at(self, split, index):
        self.count(split)
        if not 0 <= index < len(self._splits[split]):
            raise IndexError(index)
        return self._splits[split][index]

    def get(self, task_id):
        try:
            return self._by_id[task_id]
        except KeyError as error:
            raise KeyError(f"Unknown task_id {task_id!r}") from error

    def public(self, task):
        return {key: task[key] for key in PUBLIC_FIELDS}

    def task_range(self, split, start=None, stop=None):
        start = 0 if start is None else start
        stop = self.count(split) if stop is None else stop
        if stop - start > 1000:
            raise ValueError("Request at most 1000 tasks per range")
        stop = min(stop, self.count(split))
        return [
            {**self.public(self._splits[split][index]), "index": index}
            for index in range(start, stop)
        ]


@lru_cache(maxsize=4)
def load_catalog(source_dir=None, data_dir=None):
    if source_dir:
        return Catalog.from_source(source_dir)
    return Catalog.from_published(data_dir)


def configured_catalog():
    return load_catalog(
        os.environ.get("IMAGE_TEXT_GEN_SOURCE_DIR") or None,
        os.environ.get("IMAGE_TEXT_GEN_DATA_DIR") or None,
    )
