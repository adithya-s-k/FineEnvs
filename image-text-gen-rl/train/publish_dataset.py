"""Publish the screened task splits as a Hub dataset and pin the env to that commit.

Derives train/validation/test from the pinned leffff CSVs minus exclusions.json, writes
data/<split>.jsonl plus a dataset card crediting the original, uploads to
AdithyaSK/image-text-gen-rl-prompts, and records the upload commit as PUBLISHED_REVISION
in envs/image_text_gen/data/catalog.py so the environment serves exactly that version.

  ../../launch image-text-gen-rl --exec python image-text-gen-rl/train/publish_dataset.py
"""

import argparse
import collections
import json
import re
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from image_text_gen.data.catalog import (
    DATASET,
    EXCLUSIONS,
    PUBLISHED_DATASET,
    REVISION,
    SPLITS,
    Catalog,
    load_exclusions,
)

CATALOG_FILE = Path(__file__).resolve().parents[1] / "envs/image_text_gen/data/catalog.py"
ENV_SPACE = "AdithyaSK/image-text-gen-rl-env"
CODE = "https://github.com/adithya-s-k/FineEnvs/tree/codex/image-text-gen-rl/image-text-gen-rl"


def card(catalog, categories, private):
    counts = {split: catalog.count(split) for split in SPLITS}
    lines = "\n".join(f"| `{k}` | {v} |" for k, v in sorted(categories.items(), key=lambda kv: -kv[1]))
    return f"""---
license: mit
language:
  - en
task_categories:
  - text-to-image
tags:
  - text-rendering
  - reinforcement-learning
  - openenv
pretty_name: Image Text Gen RL Prompts
size_categories:
  - 10K<n<100K
source_datasets:
  - {DATASET}
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
      - split: test
        path: data/test.jsonl
---

# Image Text Gen RL Prompts

Text-rendering prompts for reinforcement learning of text-to-image models: each prompt describes
an image and quotes exactly one string the image must show (`target_text`). These are the tasks
served by the [image text generation RL environment]({"https://huggingface.co/spaces/" + ENV_SPACE}),
which scores a generated image by having vision models transcribe it blind and comparing the
reading with the target.

## Credit

This dataset is a **screened, re-split derivative** of
[**{DATASET}**](https://huggingface.co/datasets/{DATASET})
by **leffff** ([code](https://github.com/leffff/Diffusion-Reward-Modeling-for-Text-Rendering)),
released under the MIT license and supported by [SMILES / Skoltech](https://smiles.skoltech.ru/).
All prompts, target texts and the SD3 OCR scores used for `baseline_ocr` are theirs; please credit
and cite the original work. This derivative is also MIT-licensed.

## What changed

Derived from the source at revision `{REVISION}`:

1. **Screening.** {sum(categories.values())} of {catalog.manifest["screened"]:,} source prompts were removed
   for sexual, vulgar, hateful, violent, self-harm or drug content: a strict word list plus a
   rubric classifier (Gemma 4 31B via HF Inference Providers). The screen errs toward removal.
   Only the removed row IDs and categories are listed in `excluded_ids.json`; their text is not
   republished.
2. **Splits.** The source test split is kept as `test`. The source train split is de-duplicated by
   prompt, stripped of prompts that also appear in test, and hashed 96/4 into `train`/`validation`.

| Split | Rows |
|---|---:|
| train | {counts["train"]:,} |
| validation | {counts["validation"]:,} |
| test | {counts["test"]:,} |

Removed by category:

| Category | Rows |
|---|---:|
{lines}

## Fields

| Field | Description |
|---|---|
| `task_id` | Stable content hash; the environment's replay ID |
| `split` | `train`, `validation` or `test` |
| `family` | `text_render` |
| `prompt` | Image description quoting the target text |
| `target_text` | The exact text to render |
| `text_len` | Characters in `target_text` |
| `baseline_ocr` | Mean Qwen OCR Levenshtein score of the source's five SD3 samples (difficulty prior) |
| `source_id` | Row `id` in the source dataset |

Source code, screening and reproduction: [{CODE}]({CODE}).
{"" if not private else chr(10) + "_This repository is private while the experiment is in progress._" + chr(10)}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=PUBLISHED_DATASET)
    parser.add_argument("--public", action="store_true", help="publish publicly (default: private)")
    args = parser.parse_args()

    screen = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    catalog = Catalog.from_source(excluded=load_exclusions())
    catalog.manifest["screened"] = screen["screened"]
    categories = collections.Counter(e["category"] for e in screen["excluded"].values())

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        root = Path(staging)
        (root / "data").mkdir()
        for split in SPLITS:
            with open(root / "data" / f"{split}.jsonl", "w", encoding="utf-8") as handle:
                for row in catalog.rows(split):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (root / "excluded_ids.json").write_text(json.dumps(
            {"source": DATASET, "revision": REVISION, "screened": screen["screened"],
             "excluded": screen["excluded"]}, indent=1) + "\n")
        (root / "README.md").write_text(card(catalog, categories, not args.public), encoding="utf-8")
        commit = api.upload_folder(
            repo_id=args.repo, repo_type="dataset", folder_path=root,
            commit_message=f"Publish screened splits (catalog {catalog.manifest['catalog_id']})",
        )
    revision = commit.oid
    source = CATALOG_FILE.read_text(encoding="utf-8")
    source = re.sub(r'PUBLISHED_REVISION = .*', f'PUBLISHED_REVISION = "{revision}"', source, count=1)
    CATALOG_FILE.write_text(source, encoding="utf-8")
    print(json.dumps({"repo": f"https://huggingface.co/datasets/{args.repo}", "revision": revision,
                      "catalog_id": catalog.manifest["catalog_id"],
                      "counts": catalog.manifest["counts"], "excluded": dict(categories)}, indent=2))


if __name__ == "__main__":
    main()
