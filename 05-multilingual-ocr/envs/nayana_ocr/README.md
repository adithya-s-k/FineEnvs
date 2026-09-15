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

One OpenEnv environment serves **section OCR** and **multiple-choice document VQA**.
Open `/web` to try a task. `/manifest` describes the prepared data window; `/docs` describes
the API. Observations contain an image path and instructions; reference answers remain
server-side. Episodes accept one answer.

The Docker image expects a finalized snapshot at `/app/snapshot`, or at `NAYANA_SNAPSHOT`.
The deployment script bundles an explicitly selected window. The environment does not fetch
the entire corpus or use the Dataset Viewer as a training backend.

Source data: [CognitiveLab's NayanaOCR_Corpus_2025](https://huggingface.co/datasets/Cognitive-Lab/NayanaOCR_Corpus_2025),
licensed **CC BY-NC 4.0**. Prepared page images, annotations, and OCR crops retain that license
and attribution. The environment source code is Apache-2.0; see `LICENSE` and `NOTICE`.

Full source, training recipe, notebook, tests, and reproduction guide:
[HuggingEnvs / 05-multilingual-ocr](https://github.com/adithya-s-k/HuggingEnvs/tree/codex/multilingual-ocr/05-multilingual-ocr).
