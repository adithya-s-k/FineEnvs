---
title: LaTeX OCR Training
short_description: Cross-model RL training, evaluation, and run history
emoji: 🎯
sdk: gradio
app_file: app.py
tags:
 - trackio
hf_oauth: true
hf_oauth_scopes:
 - write-repos
---

# LaTeX OCR: combined model comparison

This is the single [HuggingEnvs LaTeX OCR results dashboard](https://huggingface.co/spaces/HuggingEnvs/trackio-latex-ocr).
It opens on **latex-ocr-comparison**, with four models overlaid on shared training and evaluation charts.
All curves use **optimizer step** on the x-axis. `eval/test_reward` shows the recorded evaluation score;
`eval/reward_gain` shows the change from each model's own baseline. Training reward, loss, entropy,
clipping, KL (where logged), and system metrics are also available.

## Main comparison

The latest common logged evaluation is at **step 3,500**. Each training snapshot contains **3,999**
optimizer steps against a configured 5,000-step budget; these are the available snapshots, not proof of
completed 5,000-step jobs.

| Model / configuration | Baseline reward | Reward at step 3,500 | Change |
|---|---:|---:|---:|
| Qwen3-VL-2B / colocated | 0.6154 | 0.7231 | +0.1077 |
| Qwen3.5-2B / colocated | 0.6877 | 0.7101 | +0.0224 |
| GLM-OCR / colocated | 0.4494 | 0.6644 | +0.2150 |
| Gemma4-E2B / stabilized | 0.3917 | 0.5697 | +0.1780 |

Gemma's stabilized run uses learning rate **5e-6** and KL coefficient **0.03**. The other three main runs use
**1e-5** and **0**. The earlier Gemma configuration fell from **0.3917** to **0.1322** at its last evaluation
(step 1,500); that run is retained in the history view. This makes the configuration change visible.

These are historical logged rewards. Evaluation sample counts and the exact scorer revision are not stored
in these comparison logs, so this is not a controlled model leaderboard or a fresh benchmark. The final
50 training records have a clipped-completion ratio of 1.0 for GLM and stabilized Gemma; use that diagnostic
alongside their reward curves when interpreting the results.

## Other views in the same dashboard

Use **Project** to choose:

- **latex-ocr-history**: 13 substantive earlier run segments, including scaled experiments, the original
  Gemma configuration, the Red Hat Qwen3-VL run, and the 200-step Qwen3.5 demo. Negative results are retained.
- **latex-ocr-2b-overnight**: the original Qwen3.5 run through 3,000 optimizer steps, including sample images.
- **latex-ocr-eval**: that overnight run's baseline/checkpoint evaluations on 1,000 examples and sample
  predictions. Its reward moves from 0.7446 to 0.8298 and exact match from 20.0% to 42.7%. Its evaluation
  protocol is separate from the four-model comparison; the published model is not independently matched
  to the local checkpoint by these logs.

Short 10–30-step preflights and restart fragments under 100 training records remain in the personal archives.
Several scale runs reused the same Trackio run ID across restarts. The history view splits those attempts
at optimizer-step resets; it does not connect unrelated curves or guess which evaluation belongs to an
ambiguous attempt. Run configs include source bucket, project, run ID, attempt, source hash, and training
parameters. The [summary and source inventory](https://github.com/adithya-s-k/HuggingEnvs/pull/8) document
the selections, exclusions, and recorded values.

## Project assets

- [Final collection](https://huggingface.co/collections/HuggingEnvs/latex-ocr)
- [OpenEnv environment and playground](https://huggingface.co/spaces/HuggingEnvs/latex-ocr-env)
- [Published Qwen3.5 GRPO model](https://huggingface.co/HuggingEnvs/qwen3.5-latex-ocr-grpo)
- [Environment, notebook, HF Jobs, and dashboard source](https://github.com/adithya-s-k/HuggingEnvs/pull/8)

One bucket, `HuggingEnvs/latex-ocr-results`, backs this Space at `/data` with `TRACKIO_DIR=/data/trackio`.
The experiment Spaces and original buckets remain under `AdithyaSK`. The current Qwen3-VL-2B notebook and
HF Jobs smoke are separate from these historical runs. Publish new exploratory runs to a personal Space.

## Rebuild and deploy

`archive-sources.json` records the verified source buckets, paths, and hashes. Download each database into
`<source-dir>/<bucket-name>/<project>.db`, verify its SHA-256 against the manifest, then run:

```bash
python 01-latex-ocr/dashboard/build_comparison.py \
  --source-dir /path/to/downloaded-archives --output-dir /tmp/latex-combined
uv run --with-requirements 01-latex-ocr/dashboard/requirements.txt --with pytest \
  python -m pytest 01-latex-ocr/dashboard/ -q
```

The builder writes two derived databases and `comparison-summary.json`. It preserves the original scalar
values, aligns training/evaluation by optimizer step, deduplicates identical repeated baselines, and rejects
conflicting values. It does not modify the archives. Only the main comparison and unambiguous single-attempt
history runs include mapped evaluations. The original overnight databases and restored images remain intact.

Upload the two derived databases under the bucket's `trackio/` directory. Deploy `app.py`, `requirements.txt`,
and this README to the canonical Space. Keep `TRACKIO_BUCKET_ID=HuggingEnvs/latex-ocr-results` and its `/data`
mount together. The dashboard pins Trackio **0.37.1**, the latest release verified on 2026-09-14.
Its native metric-aware sampling preserves interleaved reward/loss and sparse evaluation entries, so the
older scalar sampling workaround has been removed. CI uses the same requirements as the deployed Space
and checks that these metrics survive sampling without changing the archived values.
The app puts the combined project first for visitors without a selected project, and orders evaluation reward
and baseline-relative gain before training diagnostics. All other projects remain selectable.
