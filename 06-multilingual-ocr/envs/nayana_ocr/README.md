---
title: Nayana Multilingual OCR
emoji: 📖
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8000
license: cc-by-nc-4.0
short_description: OpenEnv document OCR, layout and VQA over 22 languages
tags:
  - openenv
  - reinforcement-learning
  - multilingual
  - ocr
datasets:
  - Cognitive-Lab/NayanaOCR_Corpus_2025
  - sarvamai/indic-ocr-bench
---

# Nayana multilingual OCR environment

One OpenEnv environment serves the **complete 22-language Nayana corpus** for **full-page OCR,
section OCR, layout detection, multiple-choice and descriptive document VQA**. Open `/web` for language/task selection,
indexed navigation, a full-size image viewer, and reward/CER feedback. References are revealed
after scoring in the playground. OpenEnv task discovery and observations exclude references.

**Trained against it:** [gemma-4-E4B-it-kannada-ocr-grpo](https://huggingface.co/FineEnvs/gemma-4-E4B-it-kannada-ocr-grpo),
GRPO on 4,000 Kannada section crops, cuts Sarvam Indic OCR Bench character error on Kannada from
0.428 to 0.360. Part of the [Multilingual Multimodal Envs](https://huggingface.co/collections/FineEnvs/multilingual-multimodal-envs-6ac0c27c137f93e0799603e4) collection; source in
[`06-multilingual-ocr/`](https://github.com/adithya-s-k/FineEnvs/tree/codex/multilingual-ocr/06-multilingual-ocr).

The source lives in [FineEnvs/NayanaOCR_Corpus_2025_bucket](https://huggingface.co/buckets/FineEnvs/NayanaOCR_Corpus_2025_bucket),
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
crops. Layout returns JSON boxes labeled text/title/caption/table/image/formula and is scored
with mean class-aware region F1 at IoU .50:.05:.95. Descriptive VQA uses Gemma 4 31B as a strict
reference-based judge: all six rubric checks must pass. The playground shows layout overlays
after scoring. Table reconstruction is not scored.

The judge uses **HF Inference Providers**, explicitly routing Gemma 4 31B through DeepInfra.
No dedicated GPU endpoint is required. Provider failures return no reward; model/provider
and rubric identity are available in `/manifest`. Provider serving revisions cannot be
commit-pinned. This compares against source answers, not independent image verification.

Visual inspection found missing/distorted glyphs in an original Arabic source page
(`document_10026_page_106`), before masking. The playground flags this source caveat.
The corpus index is not an image-quality audit or evidence of model performance.

**Evaluation benchmark: [Sarvam Indic OCR Bench](https://huggingface.co/datasets/sarvamai/indic-ocr-bench)**
by **Sarvam AI** (Apache-2.0) - 6,909 text-block crops in 23 languages (all 22 languages of the
Eighth Schedule plus English), ground truth reviewed twice by human language experts. Eleven
of those languages are not in the Nayana corpus. It is served as evaluation splits
`indic_ocr_bench_test` (6,908 of 6,909) and `indic_ocr_bench_small` (1,173, ~51 per language) at revision
`84ce7ce`, from a serving copy in [FineEnvs/indic-ocr-bench-bucket](https://huggingface.co/buckets/FineEnvs/indic-ocr-bench-bucket)
mounted read-only at `/indic-ocr-bench`. Each crop is an ordinary **Section OCR** task - same
prompt, same answer, same reward and same observation as a corpus region - so results are
directly comparable. The benchmark's own CER and WER, from its official `metrics.py` (vendored
unmodified), are reported alongside as `official_cer` / `official_wer`; they are never rewarded.
These splits are never offered to training samplers. One test row, `indic_ocr_bench_test_eng_5`, has ground truth but no image in the source data, so it is not served; scores on `indic_ocr_bench_test` are over 6,908. All credit for the benchmark belongs to
Sarvam AI; please cite:

```bibtex
@misc{sarvam-indic-ocr-bench,
  title={Sarvam Indic OCR Bench},
  author={Sarvam AI},
  year={2026},
  url={https://huggingface.co/datasets/sarvamai/indic-ocr-bench}
}
```

Source: [CognitiveLab's NayanaOCR_Corpus_2025](https://huggingface.co/datasets/Cognitive-Lab/NayanaOCR_Corpus_2025),
**CC BY-NC 4.0**. Copied annotations, page images, and OCR crops retain this license and
attribution. Environment source is Apache-2.0; see `LICENSE` and `NOTICE`.

[Source, reproduction guide, notebook, training scripts and checks](https://github.com/adithya-s-k/FineEnvs/tree/codex/multilingual-ocr/06-multilingual-ocr).
