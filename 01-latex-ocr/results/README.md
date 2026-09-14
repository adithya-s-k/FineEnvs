# Verification

Smoke checks verify the migrated pipeline. They are not a benchmark or a reproduction of a long training run.

Recorded on 2026-09-14:

- **37 tests pass**, including migrated rubric/Task API/client tests and regressions for bounded loading,
  stale reset targets, stream task identity, raw-completion scoring, and zero-length rewards.
- [`smoke-fixture.json`](./smoke-fixture.json): real HTTP/WebSocket server with two offline image fixtures,
  tested in materialize and stream modes.
- [`smoke-real-data.json`](./smoke-real-data.json): the same server tests using two actual
  `unsloth/LaTeX_OCR` rows per split. Exact score 1, empty/padded score 0, partial score 0.592405.
- [`smoke-hosted.json`](./smoke-hosted.json): redeployed `HuggingEnvs/latex-ocr-env` Space, reporting 3,000 test
  tasks. Checks task discovery, PNG observations, independent sessions, and the same reward cases.
- Docker build and fixture smoke pass on Linux ARM64; `/healthz` and the `/web/` playground both return 200.
- [CPU HF Job](https://huggingface.co/jobs/AdithyaSK/6aa7e10321047bf1b0387444): completed with 37 tests
  and the materialize/stream fixture smoke passing, using the committed Jobs entry point.
- [A10G HF Job](https://huggingface.co/jobs/AdithyaSK/6aa7e1145527934177eddb1f): **completed** two
  Qwen3-VL-2B optimizer steps with finite loss (-0.01740147) and a verified adapter weight update.
  The two held-out samples scored 0.4859205 both before and after; this is a pipeline check, not a quality gain.
  Raw predictions and references are in [`smoke-gpu.json`](./smoke-gpu.json).
- Notebook schema and all code cells validate. Its training cell calls the shared Python runner.
- Dashboard regression passes on Trackio 0.37.1: native metric-aware sampling retains interleaved reward/loss
  curves and sparse evaluations without changing stored values; see [`dashboard/`](../dashboard/).

[`verification.json`](./verification.json) records source/deployment revisions and runtime versions.
The org Space runs the migrated environment source. The original deployment check is preserved in
[`smoke-hosted-before-transfer.json`](./smoke-hosted-before-transfer.json).
An oracle reference recovered after a terminal action is used only by the smoke tests, never for model training.

Re-run from the repository root:

```bash
uv run --frozen --project 01-latex-ocr/envs/latex_ocr --extra dev pytest 01-latex-ocr/envs/latex_ocr/tests -q
uv run --frozen --project 01-latex-ocr/envs/latex_ocr latex-ocr-smoke
uv run --frozen --project 01-latex-ocr/envs/latex_ocr latex-ocr-smoke --real-data
```

For CPU/GPU HF Jobs commands and persistent run output, see [`../train/README.md`](../train/README.md).

## Published assets

The [collection](https://huggingface.co/collections/HuggingEnvs/latex-ocr-6aa7ed8498b3ffd222ad1c8c)
contains four entries: environment, published model, source dataset, and one Trackio dashboard. The dashboard
uses a single bucket with five comparison runs across four models, 12 other earlier run segments, and the historical Qwen3.5 run. Six experimental Spaces and
nine trial/archive buckets were returned to `AdithyaSK`, preserving their data and Space bucket mounts.
[`hub-assets.json`](./hub-assets.json) records the curated inventory and transfer checks.

The retained `latex-ocr-eval` project records 1,000 test examples for each checkpoint. Its saved baseline
reward is 0.744633157 with 20.0% exact match; `ckpt-3000` records 0.829776173 with 42.7% exact match.
These are historical logged results, not a fresh evaluation or a reproduction by the two-step HF Job.
The logs identify a local checkpoint path; they do not independently establish that the published model's
weights are identical to `ckpt-3000`. Intermediate checkpoint comparisons and sample predictions remain
available in that same dashboard.

The final `HuggingEnvs/latex-ocr-results` bucket preserves the two historical databases unchanged. Training
image files were recovered from the original run archive. The 105 evaluation image references were restored
from the first 15 `unsloth/LaTeX_OCR` test images, after matching each sample's stored gold text exactly.
The original bucket remains intact under `AdithyaSK/trackio-latex-ocr-bucket`.

## Combined model results

[Model comparison](./model-comparison.md) includes unstable and stabilized Gemma in the main comparison,
the measured gains/regression, and snapshot limits. [REPRODUCE.md](../REPRODUCE.md) gives recorded configurations
and reconstruction/training commands. [comparison-summary.json](./comparison-summary.json)
records every selected segment and its original scalar values, evaluation points, provenance, and exclusions.
The builder and dashboard have five passing regression tests covering missing charts, optimizer-step
alignment, repeated baselines, asynchronous timestamps, restart separation, source preservation, and the hosted landing view.
