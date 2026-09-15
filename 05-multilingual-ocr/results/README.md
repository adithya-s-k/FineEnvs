# Verification results

These are pipeline checks, not model evaluations or evidence of reward improvement.

## Complete corpus: schema 2 / index 1 / package 0.3.0

The current server uses the complete bucket copy and an immutable full-corpus index. Reports:

**38 tests passed**, including the installed TRL CPU loader. Both local and hosted servers
passed all **66 language/task combinations** through OpenEnv and all 66 through Gradio.
All six notebook CPU cells ran against the hosted corpus. The final deployment's source
inventory verification was followed by another 12-group service check.

The source audit covers 1,807 files (813,665,107,295 bytes). All 22 index databases passed
SHA-256 and SQLite integrity checks: 1,006,170 pages and 5,989,341 candidate tasks.

| Measured operation | Local HTTP ranges | Space bucket mount |
|---|---:|---:|
| First/middle/last metadata queries | 198 passed, no image loads | 198 passed, no image loads |
| Prefetch one 65-page source block | 9.161 s | 1.405 s |
| Median reset for 16 tasks after prefetch | 18.6 ms | 276.0 ms, including client network |
| Additional source group loads for those tasks | 0 | 0 |
| Additional renders for four repeated resets | 0 | 0 |
| Received HTTP payload for the prefetched block | 30,206,156 bytes | Not measured inside mount |

| Artifact | Scope |
|---|---|
| `corpus-cross-runtime.json` | Same pinned task on macOS/Linux: matching decoded pixels, different PNG encodings |
| `corpus-layout.json` | Exact index size, physical block count, source image-column size distribution, task totals and exclusions |
| `tests-corpus.json` | Full package tests, including HTTP, cache concurrency/eviction, iterator replay, and installed TRL group repetition |
| `corpus-index.json` | Full source audit, all page sets, all SQLite checksums/integrity checks, derived task totals and exclusions |
| `corpus-local.json` | First/middle/last metadata lookup across every language/split; prefetched block timings; all 66 language/task transport groups |
| `corpus-hosted.json` | The same addressability, block reuse, and oracle checks on the Space with its bucket mount |
| `corpus-hosted-final.json` | Twelve service groups checked again on the final Space revision after source-inventory verification |
| `corpus-local-restart.json` | Three service groups after restarting the local server |
| `deployment-corpus.json` | Space commit, exact snapshot, and read-only bucket volume |
| `notebook-corpus.json` | Execution of the notebook's actual CPU source cells against the complete corpus |
| `gradio-corpus.json` | Local and hosted UI navigation, indexed jump, full-page rendering, and score/reference behavior |

The index reads only annotations/page IDs. Its task counts precede lazy image-bound validation;
they are not an image-quality audit. The source has known annotation omissions and Arabic font
rendering defects. The 66-group smoke loads representative tasks that share source blocks where
possible. Metadata checks cover first, middle, and last positions in all 22 languages and three
splits; they do not claim to render every source page.

The one-block timing measurement excludes index construction and model generation. It records
actual HTTP payload bytes locally; mounted network I/O on the Space is not measured by that
counter. It demonstrates cache reuse for the measured block, not sustained training throughput.
All earlier reports below remain historical evidence and do not identify the currently served
full-corpus snapshot. GPU optimizer steps and held-out model improvement remain unverified.

## Historical preview: schema 2 / package 0.2.0

| Artifact | What it records |
|---|---|
| `preparation-space-preview.json` | Source revision, 64-page configuration, counts, exclusions, and snapshot identity |
| `smoke-page-ocr-local.json` | Real local HTTP, WebSocket, binary media, and TRL adapter checks for all 12 language/task groups |
| `smoke-page-ocr-hosted.json` | The same 12-group oracle and adapter checks against the deployed Docker Space |
| `gradio-hosted.json` | Hosted playground checks across all 12 groups and independent sessions |
| `hosted-manifest.json` | Public manifest checked field-for-field against the local snapshot |
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
The original Arabic JPEG for `document_10026_page_106` visibly contains missing/distorted
glyphs and replacement boxes, independently of our masking. Its exact asset hash and the
inspection limit are recorded in [REPRODUCE.md](../REPRODUCE.md). No automatic font-quality
filter is claimed; the UI flags this known source-rendering caveat for Arabic.

Each service probe scores an empty answer and a known reference in separate episodes, expecting
0 and 1. Two training adapter instances reset to the same ID and share one fetched image.
Trusted probes read references from a local catalog; OpenEnv observations and discovery do not
return them. The separate public Gradio playground intentionally reveals a reference after
scoring, matching the LaTeX OCR interaction.

The [live Space](https://huggingenvs-nayana-ocr-env.hf.space/web/) passed all 12 service groups
and all 12 Gradio groups, including two independent UI sessions. One Gradio sweep hit a client
read timeout after ten groups; the remaining cases passed with a 60-second client timeout.
Later Hub control-plane refreshes hit local DNS errors, while the live app remained reachable.
`deployment.json` records the last observed Hub stage separately from successful endpoint checks.

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
