# Verification results

These are pipeline checks, not model evaluations or evidence of reward improvement.

## Current preview: schema 2 / package 0.2.0

| Artifact | What it records |
|---|---|
| `preparation-space-preview.json` | Source revision, 64-page configuration, counts, exclusions, and snapshot identity |
| `smoke-page-ocr-local.json` | Real local HTTP, WebSocket, binary media, and TRL adapter checks for all 12 language/task groups |
| `gradio-page-ocr.json` | Empty/exact scoring, navigation, reference clearing, RTL fields, and two independent Gradio sessions |
| `notebook-preview.json` | Six notebook source code cells executed on CPU with a smaller schema-2 window |
| `deployment.json` | Final HF Space commit, source commit, runtime and hosted checks |

The four-language preview uses 16 pages each of English, Kannada, Hindi, and Arabic. It has
**414 tasks: 303 section OCR, 58 full-page OCR, and 53 MCQ VQA**, with 185,369,938 media bytes.
Its snapshot ID is `5f4986cf4c484bdfef97413edbb84b09071c545c2eb32d1e6897c1e57b31c1a7`.
Six pages have overlapping regions and are excluded only from full-page OCR. Eleven MCQs
have ambiguous option mappings; 256 descriptive VQA questions are deferred. All sampled pages
are in train; this preview does not support a held-out model comparison.

All 26 tests passed with the optional TRL dependencies. They cover the original data/session
contracts plus geometric LTR/RTL reading order, overlap exclusion, full-page annotation masks,
and playground navigation. Local browser inspection covered English and Arabic full pages,
score/reference reveal, and selection clearing. OCR PNGs preserve dimensions and source pixels
inside annotated boxes. Geometric order and supplied annotation accuracy still need dataset
quality audits before using this as a model benchmark.

Each service probe scores an empty answer and a known reference in separate episodes, expecting
0 and 1. Two training adapter instances reset to the same ID and share one fetched image.
Trusted probes read references from a local catalog; OpenEnv observations and discovery do not
return them. The separate public Gradio playground intentionally reveals a reference after
scoring, matching the LaTeX OCR interaction.

## Historical milestone: schema 1 / package 0.1.0

The following artifacts describe the initial section OCR + MCQ implementation. They remain as
historical evidence and do **not** verify the current page OCR build:

| Artifact | What it recorded |
|---|---|
| `preparation-real-data.json` | Eight-page window, 37 tasks, 5,945,947 media bytes |
| `smoke-real-data.json` | Local HTTP/WebSocket/adapter checks across 4 languages × 2 families |
| `smoke-fixture.json` | Synthetic transport fixture |
| `smoke-docker.json` | Initial Linux ARM64 container smoke |
| `gradio-check.json` | Initial eight-group playground check |
| `verification.json` | Initial 21 tests, notebook cells, container, processor, and local HF launcher check |

Schema-1 snapshots are rejected by package 0.2.0; prepare a new directory. In particular,
`verification.json` describes deployment status at that earlier milestone, not current status.

The smoke elapsed times exclude preparation/startup and are individual observations, not p95
latency, memory, bandwidth, or throughput benchmarks. No GPU optimizer step, held-out model
comparison, or HF Jobs execution has been recorded for experiment 05. Training checkpoint
replay is not yet verified.
