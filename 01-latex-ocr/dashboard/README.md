---
emoji: 🎯
sdk: gradio
app_file: app.py
tags:
 - trackio
hf_oauth: true
hf_oauth_scopes:
 - write-repos
---

# LaTeX OCR training and evaluation

The single results dashboard for [HuggingEnvs LaTeX OCR](https://huggingface.co/collections/HuggingEnvs/latex-ocr-6aa7ed8498b3ffd222ad1c8c).
Use the **Project** selector to switch between:

- **latex-ocr-2b-overnight**: Qwen3.5 training history through 3,000 optimizer steps, with rewards, losses, system metrics, and sample predictions.
- **latex-ocr-eval**: baseline and checkpoint comparisons on 1,000 test examples, including sample predictions.

The saved baseline records reward **0.744633157** and **20.0%** exact match. The step-3,000 checkpoint records reward **0.829776173** and **42.7%** exact match. These are historical logged results. The published model is not independently verified against that local checkpoint by this dashboard.

## Project assets

- [OpenEnv environment and playground](https://huggingface.co/spaces/HuggingEnvs/latex-ocr-env)
- [Published Qwen3.5 GRPO model](https://huggingface.co/HuggingEnvs/qwen3.5-latex-ocr-grpo)
- [Environment, notebook, and HF Jobs source](https://github.com/adithya-s-k/HuggingEnvs/pull/8)

Persistent data lives in `HuggingEnvs/latex-ocr-results`, mounted at `/data`, with `TRACKIO_DIR=/data/trackio`. This bucket contains the retained training and evaluation history. The original archives remain intact under `AdithyaSK`. Training images were recovered from the original run archive; evaluation images were restored from dataset test rows 0–14 after exact gold-text matching. Trial, demo, and ablation assets remain under `AdithyaSK` and are excluded from the final collection.

The repository's Qwen3-VL-2B notebook and HF Jobs smoke are separate from these historical Qwen3.5 results. Use a personal dashboard for exploratory or smoke runs.

## Deployment

Deploy `app.py`, `requirements.txt`, and this README to the canonical Space. Keep `TRACKIO_BUCKET_ID=HuggingEnvs/latex-ocr-results` and the `/data` bucket mount together. The app uses a compatibility wrapper for pinned Trackio 0.32.2: it reads all scalar rows from this bounded archive, avoiding periodic downsampling that otherwise hides every reward/loss entry. Media reads keep their normal limits. The database and recorded values are not rewritten.

Run the regression check from the repository root:

```bash
uv run --with trackio==0.32.2 --with pytest python -m pytest 01-latex-ocr/dashboard/test_app.py -q
```
