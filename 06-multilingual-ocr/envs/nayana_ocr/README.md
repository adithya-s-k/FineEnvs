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

# Nayana multilingual OCR

Read a page, transcribe a region, mark its layout, or answer a question about it, and see how the
answer scores. This Space serves the whole
[Nayana corpus](https://huggingface.co/datasets/Cognitive-Lab/NayanaOCR_Corpus_2025), 1,006,170
pages in 22 languages, as an [OpenEnv](https://github.com/huggingface/OpenEnv) environment you can
use in the browser or train a model against. Sarvam Indic OCR Bench is served beside it for evaluation.

Open [the playground](https://fineenvs-nayana-ocr-env.hf.space/web/), pick a task and a language,
and score an answer. The reference is revealed after scoring; the API never sends it.

**A model trained here:**
[gemma-4-E4B-it-kannada-ocr-grpo](https://huggingface.co/FineEnvs/gemma-4-E4B-it-kannada-ocr-grpo).
GRPO on 4,000 Kannada crops cut its character error on the Kannada part of Sarvam Indic OCR Bench
from 0.428 to 0.360. It is part of the [Multilingual Multimodal Envs](https://huggingface.co/collections/FineEnvs/multilingual-multimodal-envs-6ac0c27c137f93e0799603e4) collection, and the
source is in [`06-multilingual-ocr/`](https://github.com/adithya-s-k/FineEnvs/tree/codex/multilingual-ocr/06-multilingual-ocr).

## Tasks

| Task | You see | You answer | Reward |
|---|---|---|---|
| `section_ocr` | a crop of one text region | its text | `0.8 × (1 − character error) + 0.2 × exact match` |
| `page_ocr` | a whole page, with unannotated areas masked | its annotated text, in reading order | the same |
| `layout_detection` | a whole page | boxes labelled text, title, caption, table, image or formula | class-aware box F1, averaged over IoU 0.50 to 0.95 |
| `mcq_vqa` | a page, a question and options | a letter | exact match |
| `descriptive_vqa` | a page and an open question | a short answer | a strict Gemma 4 judge: all six checks must pass |

Full-page reading order follows the columns of the page, right to left for Arabic. Pages with
overlapping or incomplete regions are left out of full-page OCR. Table structure is not scored.

## How pages are served

The corpus stays in [a bucket](https://huggingface.co/buckets/FineEnvs/NayanaOCR_Corpus_2025_bucket),
mounted read-only at `/corpus`. Each language has an index of every task and where its page lives.
Indexes and pages are fetched when first needed and kept in bounded caches: 4 GB of indexes, 4 GB of
image groups and 512 MB of rendered tasks by default.

Pages are stored about 100 to a group, so the first page from a group takes a few seconds and the
rest are instant. `/manifest` identifies the served corpus, `/docs` describes the API, and
`/data/cache` reports cache counters.

## How free-text answers are graded

Descriptive VQA is graded by Gemma 4 31B through HF Inference Providers, routed to DeepInfra, so no
dedicated GPU is needed. It compares your answer with the corpus reference. It does not look at the
image. A provider failure returns no reward rather than a wrong one. The model, provider and rubric
are listed in `/manifest`. Provider-side serving versions cannot be pinned.

## Sarvam Indic OCR Bench

[Sarvam Indic OCR Bench](https://huggingface.co/datasets/sarvamai/indic-ocr-bench), by **Sarvam AI**
(Apache-2.0): 6,909 text-block crops in 23 languages, the 22 of India's Eighth Schedule plus
English. Its ground truth was reviewed twice by human language experts. Eleven of its languages
are not in Nayana.

It is served as two evaluation splits at revision `84ce7ce`: `indic_ocr_bench_test` (6,908 crops)
and `indic_ocr_bench_small` (1,173, about 51 per language). They come from a serving copy in
[FineEnvs/indic-ocr-bench-bucket](https://huggingface.co/buckets/FineEnvs/indic-ocr-bench-bucket),
mounted at `/indic-ocr-bench`. A benchmark crop is an ordinary section-OCR task, with the same
prompt and reward, so results compare directly with the corpus.

The benchmark's own CER and WER, from its official `metrics.py`, run unmodified, are reported next
to the reward as `official_cer` and `official_wer`. They are never rewarded, and the benchmark is
never offered to a training sampler. One test row, `indic_ocr_bench_test_eng_5`, ships without an
image, so the test split has 6,908 crops rather than 6,909. All credit for the benchmark belongs to
Sarvam AI.

## Caveats

One original Arabic page (`document_10026_page_106`) has missing and distorted glyphs in the source
scan, and the playground flags it. The index checks annotations, not image quality, and it is not
evidence of how any model performs.

## Licence and citation

The corpus is CognitiveLab's [NayanaOCR_Corpus_2025](https://huggingface.co/datasets/Cognitive-Lab/NayanaOCR_Corpus_2025),
**CC BY-NC 4.0**, and pages, annotations and crops served here keep that licence and attribution.
The environment code is Apache-2.0; see `LICENSE` and `NOTICE`.

```bibtex
@misc{fineenvs,
  author = {Kolavi, Adithya S},
  title  = {FineEnvs: Open Source RL Environments for LLM Agents},
  year   = {2026},
  url    = {https://github.com/adithya-s-k/FineEnvs}
}

@misc{sarvam-indic-ocr-bench,
  title  = {Sarvam Indic OCR Bench},
  author = {Sarvam AI},
  year   = {2026},
  url    = {https://huggingface.co/datasets/sarvamai/indic-ocr-bench}
}
```
