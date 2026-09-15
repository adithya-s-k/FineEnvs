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

One OpenEnv environment serves **full-page OCR**, **section OCR**, and
**multiple-choice document VQA**. The playground offers language/task selection,
previous/next/shuffle navigation, a full-size image viewer, and reward/CER feedback.
It reveals the reference after scoring, like the LaTeX OCR playground.

Open `/web` to try a task. `/manifest` describes the prepared data window; `/docs` describes
the API. OpenEnv observations and task discovery contain image paths and instructions,
never reference answers, including after grading. Episodes accept one answer.

Full-page OCR preserves the page canvas and layout, masking areas outside valid text-region
annotations. Some source pages contain unannotated headers; masking keeps visible content
aligned with the scored transcription. References join regions using the versioned
`whitespace-columns-v1` geometric reading order, with right-to-left columns for Arabic.
Pages with missing/invalid annotations or overlapping text boxes are excluded from this task.
This is annotated full-page transcription; table-format reconstruction and descriptive VQA
are not scored. VQA retains the original JPEG, and section OCR uses lossless PNG crops.

The Docker image expects a finalized snapshot at `/app/snapshot`, or at `NAYANA_SNAPSHOT`.
The deployment script bundles an explicitly selected window. The environment does not fetch
the entire corpus or use the Dataset Viewer as a training backend.
The public preview demonstrates serving and replay. It is not a held-out model benchmark.
Visual inspection found missing/distorted glyphs in an original Arabic source page
(`document_10026_page_106`). Those artifacts precede masking and are retained here. The
playground flags this caveat for Arabic; review source rendering before using it for training.

Source data: [CognitiveLab's NayanaOCR_Corpus_2025](https://huggingface.co/datasets/Cognitive-Lab/NayanaOCR_Corpus_2025),
licensed **CC BY-NC 4.0**. Prepared page images, annotations, and OCR crops retain that license
and attribution. The environment source code is Apache-2.0; see `LICENSE` and `NOTICE`.

Full source, training recipe, notebook, tests, and reproduction guide:
[HuggingEnvs / 05-multilingual-ocr](https://github.com/adithya-s-k/HuggingEnvs/tree/codex/multilingual-ocr/05-multilingual-ocr).
