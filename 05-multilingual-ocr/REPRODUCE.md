# Reproduce experiment 05

Run commands from `05-multilingual-ocr/`. Python 3.12 and `uv` are the tested local setup.
The environment lockfile pins OpenEnv 0.4.2, Datasets 5.0.1, and optional TRL 1.13.0.
Nayana is pinned to `b220b074a8c82bb90427051e856e4c4edc79885b`; a full dataset download is not required.

## 1. Check the implementation on CPU

```bash
uv run --frozen --project envs/nayana_ocr --extra dev pytest envs/nayana_ocr/tests -q
uv run --frozen --project envs/nayana_ocr nayana-smoke

# Optional training dependencies exercise the actual TRL sampler, still on CPU.
uv run --frozen --project envs/nayana_ocr --extra dev --extra train \
  pytest envs/nayana_ocr/tests -q
```

The ordinary test run skips the two optional TRL sampler cases if TRL is not installed.
The smoke command starts a real server, uses separate WebSocket sessions, discovers tasks,
fetches image bytes, checks empty and exact predictions, and drives the training adapter.
Synthetic fixtures have ASCII transport labels across language configs; they measure no
multilingual recognition quality.

## 2. Prepare and check real data

```bash
uv run --frozen --project envs/nayana_ocr nayana-prepare \
  --output data/snapshots/real-smoke-v2 --languages en kn hi ar --pages-per-language 2
uv run --frozen --project envs/nayana_ocr nayana-smoke \
  --snapshot data/snapshots/real-smoke-v2 --output results/local-real-smoke-v2/smoke.json
```

The schema-2 public preview uses 16 pages per language instead of two: **64 pages,
414 tasks (303 section OCR, 58 full-page OCR, 53 MCQ VQA), 185,369,938 stored media bytes**.
Preparation took 53.294 seconds on the recorded run. This is one startup observation, not a
throughput or network-transfer benchmark. Arrow decoding, HTTP caches, and Python allocations
are additional to stored media. The first read may fetch more than one page's compressed bytes.

To reproduce the deployed window exactly:

```bash
uv run --frozen --project envs/nayana_ocr nayana-prepare \
  --output data/snapshots/space-preview-v2 --languages en kn hi ar --pages-per-language 16
```

Expected snapshot ID: `5f4986cf4c484bdfef97413edbb84b09071c545c2eb32d1e6897c1e57b31c1a7`.
The two-page command above is a smaller service check and produces a different snapshot.
Schema-1 windows from the initial milestone must be rebuilt into a new directory; the server
rejects them. Never mutate an active snapshot in place.

Defaults: 32 pages per language, 1,000,000,000 stored media bytes, 50 million pixels per page,
split seed 42, no shuffle buffer, one shard partition. Preparation validates the JPEG header
before loading pixels; media quota checks happen before committing a page. The quota covers
stored media, not process RSS, SQLite, temporary data, or the Hugging Face cache. Oversize
pages and exhausted byte budgets fail explicitly; annotation exclusions are reported separately.

Rerun the identical command to resume after interruption. A SQLite transaction stores the
completed page and Datasets iterator state together. No manifest is published until the
requested window is ready. Resume validates all settings, source revision, and Datasets
version. New budgets or page counts require a new output directory.

The corpus is document ordered in places. A short prefix may contain only training documents.
Do not move individual pages into evaluation to fill a quota: that would leak related content.

## 3. Serve locally or in Docker

```bash
NAYANA_SNAPSHOT="$PWD/data/snapshots/real-smoke-v2" \
  uv run --frozen --project envs/nayana_ocr nayana-server

# Alternative: run the same package in a container with a read-only task window.
docker build -t nayana-ocr:local envs/nayana_ocr
docker run --rm -p 8000:8000 \
  -v "$PWD/data/snapshots/real-smoke-v2:/app/snapshot:ro" nayana-ocr:local
```

Playground: `http://localhost:8000/web`. Discovery endpoints:

```text
GET  /healthz
GET  /manifest
GET  /nayana_ocr/splits
POST /nayana_ocr/num_tasks    {"split":"train"}
POST /nayana_ocr/task         {"split":"train","index":0}
POST /nayana_ocr/task_range   {"split":"train","start":0,"stop":10}
GET  /assets/<sha256>
```

Counts refer to **derived tasks in this prepared window**, not the million-page corpus.
Empty splits return zero. Ranges contain at most 1,000 records. Task metadata and observations
exclude references; the binary asset endpoint cannot serve the SQLite catalog. Reset accepts
either `task_id` or `split`/`index`. Seeded random reset is for manual exploration; training
always specifies a task ID. Episodes accept one answer and must be reset before another.

```python
from nayana_ocr.client import connect
from nayana_ocr.models import NayanaAction

with connect("http://localhost:8000") as env:
    task = env.get_task_range("train", 0, 1)[0]
    observation = env.reset(task_id=task["task_id"]).observation
    print(observation.prompt, observation.asset_path)
    result = env.step(NayanaAction(answer="your transcription or option letter"))
    print(result.reward, result.observation.metrics)
```

Each session has independent episode state. The catalog is shared and read-only. The Gradio
playground has a per-user task ID, next/previous/shuffle navigation, and reveals the reference
after scoring. Changing a selection clears the prior answer, reference, and score. It does
not consume a training cursor. Reference exclusion in the OpenEnv API is a training contract,
not a secrecy guarantee for this public-source dataset and demonstration UI. Deployments with
multiple replicas must use the exact same immutable snapshot. A live stream allocator,
cross-replica leases, and continuous window rotation are not implemented here.

## 4. Train with TRL

Prepare a larger window first. This example is a starting configuration, not a verified
benchmark subset; inspect `manifest.json` for document diversity and split coverage.

```bash
uv run --frozen --project envs/nayana_ocr nayana-prepare \
  --output data/snapshots/train-256-v2 --pages-per-language 256 \
  --max-media-bytes 8000000000

uv run --frozen --project envs/nayana_ocr --extra train python train/grpo_nayana.py \
  --snapshot data/snapshots/train-256-v2 --smoke --output-dir results/local-gpu-smoke
```

The runner checks every requested language/task group in train and test before downloading
model weights. It stops if the window lacks held-out coverage. Prepare more documents in a
new directory in that case. An externally hosted environment can replace `--snapshot` with
`--env-url https://YOUR-SPACE.hf.space`.

| Setting | Default |
|---|---|
| Model | `Qwen/Qwen3-VL-2B-Instruct`; resolved Hub commit saved before training |
| Languages / tasks | `en kn hi ar` / `section_ocr mcq_vqa page_ocr` |
| Sampler input | Map Dataset of metadata task IDs; `--task-input iterable` also available |
| Mixing | Equal language × family groups, downsampled to the smallest group, up to 64 tasks/group |
| Evaluation | Fixed document-disjoint test IDs, up to 4 tasks/group, greedy decoding |
| Optimization | One GPU, BF16, SDPA, LoRA rank 16 / alpha 32 / dropout .05, q_proj and v_proj |
| GRPO | 30 steps, 4 generations, temperature .9, learning rate 1e-5, seed 42 |
| Completion / image budget | 2,048 tokens; processor max_pixels 1,048,576 |
| Truncation | Truncated completions masked; failed/zero-update smoke is an error |
| Checkpoints | Every 25 steps or final short-run step; keep 2 |
| Smoke override | 2 steps, 2 generations, 1 eval task per group |

There is no server-side thumbnail reduction. The Qwen processor applies the declared image
budget after task selection; small script details can become harder to read on dense pages.
Record the budget in comparisons. Section OCR, full-page OCR, and VQA are reported separately. Inspect completion truncation
per group before choosing a training budget; dense multilingual pages may need more than
2,048 tokens (`--max-completion-length`). GPU memory use for this default is not yet measured.

`run-metadata.json` records the model revision, snapshot manifest, selected train/test task
order, and configuration. Outputs include `baseline.json`, `summary.json`, the adapter and
processor, and trainer checkpoints. Summary metrics include per-language/task reward and
exact match, plus CER for OCR. The aggregate is a macro average of those groups.

The script verifies finite loss, final step count, and changed adapter weights. A smoke can
complete without an evaluation reward increase; it does not establish model improvement.
The current recorded results are CPU checks only. GPU optimizer behavior and checkpoint
replay have yet to be validated for this experiment.

For optional logging, add `--trackio-space YOUR-ACCOUNT/YOUR-EXPERIMENT --run-name NAME`.
The reward callback logs reward, exact match, CER where applicable, and overlong-answer rate
under `nayana/<language>/<family>/...` using TRL's scalar metric hook.
Use personal experimental assets until a final run is ready for the HuggingEnvs collection.
No Space, bucket, model, or collection is created automatically by preparation or training.

### Replay boundaries

Use the same arguments and output directory, adding `--resume results/local-run/checkpoint-25`
to resume map-input training. Changes in configuration, selected IDs, model commit, or
snapshot manifest are rejected. Pin `--model-revision` on the original run if reproducibility
must survive updates to the model's main branch. The adapter-update check captures weights
after checkpoint loading.

Iterable input demonstrates TRL's native group repetition over a fixed task manifest. Workers
are zero, `dispatch_batches=False`, and `max_steps` is explicit. No second sharding or repetition
layer is inserted. Trainer checkpoint resume for iterable input is deliberately rejected until
validated. General Datasets shuffle-buffer state is not an exact replay guarantee; the data
preparer avoids shuffle buffers and batched user transforms entirely. See
[Datasets iterator state](https://github.com/huggingface/datasets/blob/5.0.1/src/datasets/iterable_dataset.py)
and [TRL 1.13.0](https://github.com/huggingface/trl/blob/v1.13.0/trl/trainer/grpo_trainer.py).

## 5. Run on HF Jobs

Commit and push the branch first. `--revision` must be a full pushed Git SHA. Jobs fetch that
exact repository archive and run with its lockfile; local uncommitted changes are not included.
The launcher was executed locally against pushed commit
`6393c3785259d902b563439a291daf5d39e0b6ac`: archive download, frozen environment,
19 CPU tests (2 optional TRL tests skipped), and HTTP/WebSocket smoke passed.
It has not yet been exercised on HF Jobs.

```bash
SOURCE_REVISION=$(git rev-parse HEAD)

hf jobs uv run --flavor cpu-basic --timeout 20m train/hf_job.py \
  --revision "$SOURCE_REVISION" --mode env-smoke

hf jobs uv run --flavor cpu-basic --timeout 30m train/hf_job.py \
  --revision "$SOURCE_REVISION" --mode real-smoke

hf jobs uv run --flavor a100-large --timeout 2h --secrets HF_TOKEN train/hf_job.py \
  --revision "$SOURCE_REVISION" --mode train --prepare-pages 256 --smoke \
  --artifact-repo YOUR-ACCOUNT/nayana-experiments
```

`--artifact-repo` is optional. If provided, the launcher creates a private dataset repository
if it is absent and uploads run outputs, including partial failure diagnostics, under the
source revision. Existing repository visibility is preserved. Without it, results live in job
logs and ephemeral storage. Give each run a distinct artifact repo to avoid overwriting the
same revision's output. A prepared dataset is not uploaded by this option.

## 6. Publish an explicitly selected Space

```bash
uv run --frozen --project envs/nayana_ocr python train/deploy_space.py \
  --space-id HuggingEnvs/nayana-ocr-env --snapshot data/snapshots/space-preview-v2
```

The publisher stages the package and a finalized real-data snapshot in a Docker Space.
It rejects synthetic fixtures. The Space starts from bundled data without downloading the
corpus on startup; `/manifest` identifies the exact served window. The example publishes a
small preview, not a training/evaluation benchmark. Choose a curated final snapshot before
publishing an organization asset. The final environment Space is
[HuggingEnvs/nayana-ocr-env](https://huggingface.co/spaces/HuggingEnvs/nayana-ocr-env);
its served snapshot is recorded in [results](results/README.md). No test bucket or model
repository is needed for this preview.

## Scoring policy

OCR compares Unicode code points after NFC normalization and whitespace collapse. It
preserves case, diacritics, Indic vowel signs, ZWJ/ZWNJ, punctuation, and RTL logical order.
This is not grapheme-cluster CER. CER is normalized by reference length; reward clips the
similarity component at zero. Predictions longer than `max(1024, 4 × raw reference length)`
score zero before normalization, preventing whitespace-padding bypasses.
CER is still recorded for overlong predictions; the separate overlong rate explains rejected
answers even when their normalized transcription matches the reference.

MCQ derivation requires 2–26 nonempty, unique normalized options and exactly one option
matching the normalized source answer. Other MCQs are audited and excluded rather than
guessed. The model must return the uppercase letter; surrounding whitespace is allowed,
but explanations, multiple letters, lowercase, and option-text answers score zero.


### Full-page OCR derivation (schema 2)

`page_ocr` joins every valid text region using `whitespace-columns-v1`: spanning headings
and footers first, vertical whitespace cuts for columns, horizontal cuts for bands, and
geometric tie breakers. Arabic reverses column order; it does not reverse text code points.
This is a deterministic heuristic, not an official Nayana reading-order label. It does not
reconstruct table formatting. Missing/invalid text or boxes, duplicate region IDs, or any
positive-area region overlap exclude the page from this family; section/VQA tasks can remain.

Full-page PNGs retain native dimensions and original pixels inside annotated boxes. Everything
outside those boxes is white. Visual inspection found source headers outside the annotations;
masking avoids penalizing a correct transcription for unscored visible text. We cannot infer
whether annotation text exactly covers every glyph inside a supplied box. Real multilingual
annotation quality still needs an audit before claiming a model benchmark. VQA uses the raw
JPEG; section OCR uses a lossless crop. The 64-page preview excludes six full-page tasks for
overlapping regions and 11 MCQs with ambiguous answers; descriptive VQA remains deferred.
