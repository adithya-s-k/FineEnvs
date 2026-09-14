# Verification

Smoke checks verify the migrated pipeline. They are not a benchmark or a reproduction of a long training run.

Recorded on 2026-09-14:

- **37 tests pass**, including migrated rubric/Task API/client tests and regressions for bounded loading,
  stale reset targets, stream task identity, raw-completion scoring, and zero-length rewards.
- [`smoke-fixture.json`](./smoke-fixture.json): real HTTP/WebSocket server with two offline image fixtures,
  tested in materialize and stream modes.
- [`smoke-real-data.json`](./smoke-real-data.json): the same server tests using two actual
  `unsloth/LaTeX_OCR` rows per split. Exact score 1, empty/padded score 0, partial score 0.592405.
- [`smoke-hosted.json`](./smoke-hosted.json): existing `AdithyaSK/latex-ocr-env` Space, reporting 3,000 test
  tasks. Checks task discovery, PNG observations, independent sessions, and the same reward cases.
- Docker build and fixture smoke pass on Linux ARM64; `/healthz` and the `/web/` playground both return 200.
- [CPU HF Job](https://huggingface.co/jobs/AdithyaSK/6aa7e10321047bf1b0387444): completed with 37 tests
  and the materialize/stream fixture smoke passing, using the committed Jobs entry point.
- [A10G HF Job](https://huggingface.co/jobs/AdithyaSK/6aa7e1145527934177eddb1f): **completed** two
  Qwen3-VL-2B optimizer steps with finite loss (-0.01740147) and a verified adapter weight update.
  The two held-out samples scored 0.4859205 both before and after; this is a pipeline check, not a quality gain.
  Raw predictions and references are in [`smoke-gpu.json`](./smoke-gpu.json).
- Notebook schema and all code cells validate. Its training cell calls the shared Python runner.

[`verification.json`](./verification.json) records source/deployment revisions and runtime versions.
The existing Space was checked, not redeployed; its observations validate compatibility with the migrated client.
An oracle reference recovered after a terminal action is used only by the smoke tests, never for model training.

Re-run from the repository root:

```bash
uv run --frozen --project 01-latex-ocr/envs/latex_ocr --extra dev pytest 01-latex-ocr/envs/latex_ocr/tests -q
uv run --frozen --project 01-latex-ocr/envs/latex_ocr latex-ocr-smoke
uv run --frozen --project 01-latex-ocr/envs/latex_ocr latex-ocr-smoke --real-data
```

For CPU/GPU HF Jobs commands and persistent run output, see [`../train/README.md`](../train/README.md).
