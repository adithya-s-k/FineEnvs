---
title: Nayana Multilingual OCR
emoji: 📖
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8000
license: cc-by-nc-4.0
tags:
  - openenv
  - reinforcement-learning
  - multilingual
  - ocr
datasets:
  - Cognitive-Lab/NayanaOCR_Corpus_2025
---

# Nayana multilingual OCR environment

One OpenEnv environment serves the **complete 22-language Nayana corpus** for **full-page OCR,
section OCR, and multiple-choice document VQA**. Open `/web` for language/task selection,
indexed navigation, a full-size image viewer, and reward/CER feedback. References are revealed
after scoring in the playground. OpenEnv task discovery and observations exclude references.

The source lives in [HuggingEnvs/NayanaOCR_Corpus_2025_bucket](https://huggingface.co/buckets/HuggingEnvs/NayanaOCR_Corpus_2025_bucket),
attached read-only at `/corpus`. The Docker image bundles code and a small provenance manifest.
Per-language indexes and source image groups are fetched lazily into bounded local caches.
`/manifest` identifies the served corpus; `/docs` describes the API; `/data/cache` reports cache
counters. Training uses shuffled physical blocks with prefetch and task-ID replay.

Choose any task by its index. The first image from a source block may take longer to load:
Parquet groups contain roughly 100 pages. Subsequent tasks in that group reuse its cached
images. Defaults are 4 GB of index files, 4 GB of image groups, and 512 MB of rendered tasks.
The same package serves locally using HTTP ranges or a mounted/local source directory.

Full-page OCR preserves the canvas and original pixels inside valid text regions, masking
unannotated areas. References use the geometric `whitespace-columns-v1` reading order, with
RTL columns for Arabic. Overlapping/incomplete regions are excluded from full-page indexing;
actual image bounds are checked at load time. VQA receives original JPEGs; sections use PNG
crops. Descriptive VQA and table reconstruction are not scored.

Visual inspection found missing/distorted glyphs in an original Arabic source page
(`document_10026_page_106`), before masking. The playground flags this source caveat.
The corpus index is not an image-quality audit or evidence of model performance.

Source: [CognitiveLab's NayanaOCR_Corpus_2025](https://huggingface.co/datasets/Cognitive-Lab/NayanaOCR_Corpus_2025),
**CC BY-NC 4.0**. Copied annotations, page images, and OCR crops retain this license and
attribution. Environment source is Apache-2.0; see `LICENSE` and `NOTICE`.

[Source, reproduction guide, notebook, training scripts and checks](https://github.com/adithya-s-k/HuggingEnvs/tree/codex/multilingual-ocr/05-multilingual-ocr).
