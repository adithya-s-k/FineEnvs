# Reproduce the complete-corpus environment

Run commands from `05-multilingual-ocr/`. Python 3.12, OpenEnv 0.4.2, PyArrow 25.0.1,
Hugging Face Hub 1.31.0, Datasets 5.0.1, and TRL 1.13.0 are pinned by the package/lockfile.
`uv run --frozen` installs the checked-in dependency resolution. The corpus source is
`Cognitive-Lab/NayanaOCR_Corpus_2025@b220b074a8c82bb90427051e856e4c4edc79885b`.

## 1. Complete and verify the existing bucket

```bash
uv run --frozen --project envs/nayana_ocr nayana-mirror \
  --bucket HuggingEnvs/NayanaOCR_Corpus_2025_bucket \
  --output data/corpus-source.json

# Read-only audit on subsequent runs.
uv run --frozen --project envs/nayana_ocr nayana-mirror \
  --bucket HuggingEnvs/NayanaOCR_Corpus_2025_bucket \
  --output data/corpus-source.json --verify-only
```

The bucket must already exist. The mirror compares every pinned file with the existing copy,
refuses conflicting files, copies missing Xet objects server-side, and verifies the result.
It does not create a test bucket, re-upload the corpus through the workstation, or delete
unrelated bucket objects. The first audit found four missing Chinese shards; copying them
completed **1,807 source files / 1,784 Parquet files / 813,665,107,295 bytes**.

The inventory includes source Xet IDs, sizes, and LFS SHA-256 values. All Parquet Xet identities
and sizes are verified; small non-Xet repository files are checked by size. Corpus data and
derived indexes/images retain CognitiveLab attribution and CC BY-NC 4.0.

## 2. Build and publish the full index

```bash
uv run --frozen --project envs/nayana_ocr nayana-index \
  --inventory data/corpus-source.json --output data/corpus-index-v1 --workers 8

# Re-audit source identities after indexing, before publishing the ready snapshot.
uv run --frozen --project envs/nayana_ocr nayana-mirror \
  --output data/corpus-source.json --verify-only

# A completed build is reused; this validates databases and publishes them.
uv run --frozen --project envs/nayana_ocr nayana-index \
  --inventory data/corpus-source.json --output data/corpus-index-v1 --workers 8 --publish
cp data/corpus-index-v1/manifest.json data/corpus-manifest.json
```

Indexing reads only annotation and page-ID columns from every source shard. PyArrow retrieves
Parquet footers and selected column chunks; it does not read/decode JPEGs or construct crops.
There is one SQLite database per language, with compressed page annotations and task/location
indexes. Each shard commits transactionally. Rerun the same command after an interrupted build;
completed shards and finalized languages are reused. A writer lock prevents two builds sharing
an output. Version/configuration changes require a new output directory.

The final manifest is written only after every selected language finishes. Verify 22 languages
and 45,735 pages each for the complete corpus; an explicit `--languages` subset builds only
that subset. The command prints totals and the snapshot ID. The snapshot combines index version,
source inventory identity, split seed, and each SQLite SHA-256. Rebuilding with changed index
code requires an index-version change. Byte-identical SQLite output across unrelated software
versions/platforms is not promised; replay pins the published databases, not a future rebuild.

Publication writes all databases under `openenv/indexes/<snapshot_id>/`, then publishes the
ready manifest last. The root source Parquets remain unchanged. The committed
`data/corpus-manifest.json` records the published location and per-file checksums. Large source
and index files are not committed to Git.

## 3. Serve locally or with a mounted bucket

```bash
# HTTP-range source access, indexes fetched from the bucket lazily.
NAYANA_CORPUS_MANIFEST="$PWD/data/corpus-manifest.json" \
NAYANA_CACHE_DIR="$PWD/data/corpus-cache-local" \
  uv run --frozen --project envs/nayana_ocr nayana-server

# Or read an existing bucket mount. Open http://localhost:8000/web.
NAYANA_CORPUS_MANIFEST="$PWD/data/corpus-manifest.json" \
NAYANA_SOURCE_ROOT=/corpus NAYANA_CACHE_DIR=/tmp/nayana-cache \
  uv run --frozen --project envs/nayana_ocr nayana-server
```

Without a mount, the server uses the bucket resolve endpoint with validated HTTP Range reads.
It requires exact `206` and `Content-Range` responses, rejects full-file fallback, checks object
identity before/after cold reads, and meters received payload bytes. Network/timeouts have
bounded retries. With a mount, ordinary file reads use the same row-group cache. Network bytes
inside the mount implementation are not measured by the application counter.

For a fully downloaded offline copy, set `NAYANA_SOURCE_ROOT` to its directory and
`NAYANA_LOCAL_SOURCE=true`. That mode checks local size and SHA-256 on first use and whenever
mtime/size change, without Hub verification calls. Keep all 22 SQLite databases next to
`manifest.json`, and point `NAYANA_CORPUS_MANIFEST` there, for offline metadata access too.
`hf buckets cp`/`sync` support local copies; the full source requires over 813 GB of disk.
Remote or mounted sources do not require that local capacity.

`nayana_ocr.runtime.local_server(manifest, source_root=..., cache_dir=..., local_source=False)`
starts the same server for a notebook/job and closes it afterward. Each process owns its
catalog and caches; replicas must serve the same snapshot. Cache directories can be shared
by same-host processes using POSIX locks, but the initial recipe does not coordinate distributed
leases across hosts. Use independent local caches for Space replicas.

| Setting | Default | Meaning |
|---|---:|---|
| `NAYANA_INDEX_CACHE_BYTES` | 4,000,000,000 | Local copies of language index databases |
| `NAYANA_GROUP_CACHE_BYTES` | 4,000,000,000 | Decoded JPEG-byte columns in Arrow IPC files |
| `NAYANA_ASSET_CACHE_BYTES` | 512,000,000 | Rendered task images plus metadata |
| `NAYANA_MAX_GROUP_BYTES` | 512,000,000 | Maximum source/decoded group entry |
| `NAYANA_PREFETCH_WORKERS` | 2 | Shared limit for cold foreground and prefetch group loads |
| `NAYANA_PREFETCH_PENDING` | 4 | Maximum accepted asynchronous group requests |
| `NAYANA_MAX_SESSIONS` | 16 | Independent OpenEnv sessions |

Gradio preview files use a separate temporary cache, cleaned every five minutes for files
older than ten minutes. Reloading a task recreates its preview. Training uses the binary
asset endpoint and does not write Gradio preview files.

These are byte budgets for published cache entries. Downloads use temporary files; group loads
can temporarily add two entries up to the maximum entry size. Arrow decoding, two concurrent
image renders, index downloads, response buffers, and Python metadata also need working space.
Stale temporary files from killed processes are removed on cache startup when their writer lock is free.
A rendered entry is capped at 64 MB; its metadata header at 2 MB. Cache locks use 4,096 fixed
stripes per cache instead of retaining a lock file for every task ever seen. Collisions can
serialize otherwise independent requests. Eviction skips active leases; if all capacity is
pinned, a load fails explicitly instead of exceeding its budget. Prefetch failures are counted
and can be retried by normal access. `/data/cache` reports loads, hits, bytes, failures, evictions,
and pending groups. HTTP received bytes include retries; mounted I/O is identified separately.

The original Parquets have roughly 100 pages per row group and approximately 70–110 MB of
compressed JPEGs per group. A cold random image lookup fetches its whole projected group.
This design optimizes reuse and locality; it does not promise page-sized network access. A
future page-object or smaller-shard repack would trade another full rewrite/storage layout for
lower random-read amplification. No such repack is claimed here.

## 4. Indexed API, prefetch, and replay

Standard OpenEnv `num_tasks`, `get_task`, and `task_range` operate on the complete index. Ranges
are bounded at 1,000 records. Global split indices use a language prefix sum; language/family
indices support the UI's jump control. Unmaterialized task metadata has no image URL/hash yet.
Reset loads one selected task and returns its hash-addressed binary asset URL. Asset URLs carry
the pinned task ID so eviction can be followed by deterministic reconstruction within the
same rendering runtime. The macOS and Linux check produced identical decoded RGB pixels but
different PNG byte streams. PNG hashes are verified against the serving endpoint, not compared
across operating systems. Use the same built container image and codec libraries for replicas
that share asset URLs/cache state; a Python package lock alone does not freeze system codecs.
See `results/corpus-cross-runtime.json` for the measured example.

```python
from nayana_ocr.corpus_training import CorpusAPI, BlockTaskStream

api = CorpusAPI("https://huggingenvs-nayana-ocr-env.hf.space")
stream = BlockTaskStream(
    api,
    languages=["en", "kn", "hi", "ar"],
    families=["page_ocr", "section_ocr", "mcq_vqa"],
    seed=42,
    prefetch_blocks=2,
)
task = next(iter(stream))
state = stream.state_dict()
replayed = BlockTaskStream(
    api,
    languages=["en", "kn", "hi", "ar"],
    families=["page_ocr", "section_ocr", "mcq_vqa"],
    seed=42,
    prefetch_blocks=2,
)
replayed.load_state_dict(state)  # next task follows the consumed task
api.close()
```

The `/data/blocks`, `/data/block-tasks`, `/data/sample`, and `/data/prefetch` endpoints require
`snapshot_id`; a changed snapshot returns 409. Block metadata contains no image bytes or labels.
The iterator hashes `(seed, epoch, block_id)` for the block order, then shuffles chunks of at
most 128 task records by default. It visits every selected task once per finite epoch. Whole
blocks can be partitioned by rank/world size for independent consumers. The GRPO recipe is
single-GPU; do not add a second distributed sharding layer around it.

The iterator cursor includes snapshot, languages/families, seed, epoch, partition, chunk size,
plan identity, block index, and consumed task offset. Mismatches fail. This is exact **consumer
iterator** replay, tested across chunk boundaries; it is not an optimizer/checkpoint guarantee.
Prefetch affects cache timing, not task order. TRL's own iterable loader repeats each task for
its G completions. Tests exercise this installed loader with a tiny CPU model and real local
HTTP metadata requests, without downloading model weights.

## 5. GRPO and HF Jobs

```bash
uv run --frozen --project envs/nayana_ocr --extra train python train/grpo_nayana.py \
  --env-url https://huggingenvs-nayana-ocr-env.hf.space \
  --task-input corpus --prefetch-blocks 2 --smoke --output-dir results/local-gpu-smoke

# Colocated environment and training with a mounted bucket:
uv run --frozen --project envs/nayana_ocr --extra train python train/grpo_nayana.py \
  --snapshot data/corpus-manifest.json --source-root /corpus \
  --cache-dir /tmp/nayana-cache --task-input corpus --smoke
```

`--task-input auto` selects full-corpus iteration for bucket-backed servers. `map` or `iterable`
explicitly selects a small indexed balanced sample (`--train-per-group`), useful for controlled
experiments. In `corpus` mode that sampling cap is unused: max_steps bounds optimization,
while the dataset can continue through all selected tasks and reshuffle for later epochs.
The full pass preserves natural task proportions; it is not an equal language/family mixture.
A short max_steps run may consume only one source block and therefore one language. Use an
explicit balanced sample for short controlled comparisons, or a sufficiently long full pass
with per-language exposure counts; the smoke is a pipeline check.

Defaults: Qwen/Qwen3-VL-2B-Instruct, resolved model commit recorded; English/Kannada/Hindi/Arabic;
all three families; 30 steps; G=4; BF16/SDPA; LoRA rank16/alpha32/dropout.05 on q_proj/v_proj;
learning rate1e-5; temperature.9; seed42; 2,048 completion tokens; processor max_pixels1,048,576.
`--smoke` uses two steps, G=2, and one evaluation item per group. `--languages` can select any
of the 22 indexed languages. There is no server-side image downscaling; processor limits affect
fine text visibility. Truncated completions are masked. Larger full-page transcriptions may
need a larger token budget; inspect truncation before claiming improvement.

Evaluation uses a fixed, balanced indexed sample of document-disjoint test tasks, up to four
per language/family by default. No full-corpus metadata scan is needed. Run metadata records
configuration, model commit, full served manifest, block plan identity/task count, and exact
evaluation IDs. Outputs include baseline, trained metrics, adapter/processor, and checkpoints.
The runner checks finite loss, expected steps, and changed adapter weights. These do not require
or establish a reward increase. Metrics separate languages/families; OCR includes code-point CER.

Map-input optimizer resume validates configuration, selected IDs, model commit, and manifest.
Iterable/corpus optimizer resume is rejected until validated. Use a pinned model revision to
avoid drift in future runs. No GPU optimizer or HF Jobs execution is recorded for experiment05
at this milestone. Current checks validate the environment and CPU data-loader integration.

Commit and push before using HF Jobs. The launcher fetches an exact 40-character Git revision
and uses its frozen lock. For a colocated environment, attach the bucket to the job:

```bash
SOURCE_REVISION=$(git rev-parse HEAD)
CORPUS_MANIFEST=$(python -c 'import json; m=json.load(open("data/corpus-manifest.json")); print("hf://buckets/"+m["bucket_id"]+"/openenv/indexes/"+m["snapshot_id"]+"/manifest.json")')

hf jobs uv run --flavor cpu-basic --timeout 20m train/hf_job.py \
  --revision "$SOURCE_REVISION" --mode env-smoke

hf jobs uv run --flavor a100-large --timeout 2h --secrets HF_TOKEN \
  --volume hf://buckets/HuggingEnvs/NayanaOCR_Corpus_2025_bucket:/corpus:ro \
  train/hf_job.py --revision "$SOURCE_REVISION" --mode train \
  --corpus-manifest "$CORPUS_MANIFEST" --source-root /corpus --task-input corpus --smoke \
  --artifact-repo YOUR-ACCOUNT/nayana-experiments
```

Alternatively pass `--env-url` to train against the hosted Space, with no job volume. The
optional artifact repo receives run outputs in a private personal dataset repository; no
publication occurs if it is omitted. Give separate runs distinct artifact repos. Keep
experimental outputs personal until a final result is ready for the organization collection.

## 6. Deploy the same environment to the Space

```bash
uv run --frozen --project envs/nayana_ocr python train/deploy_space.py \
  --space-id HuggingEnvs/nayana-ocr-env --corpus-manifest data/corpus-manifest.json \
  --output results/deployment-corpus.json
```

The publisher validates a ready real-source manifest and checks that every index file is already
in the bucket. It uploads code plus the manifest, removes `snapshot/*` from the old preview,
attaches the existing bucket read-only at `/corpus`, and configures local caching. Unrelated
volumes are preserved. Bucket source files and indexes are not embedded in the Docker image.
Space mount configuration uses the official Hub `set_space_volumes` API. The same package works
with HTTP ranges if no mount is configured. See [Hub volumes](https://huggingface.co/docs/huggingface_hub/guides/manage-spaces)
and [buckets](https://huggingface.co/docs/huggingface_hub/guides/buckets).

## 7. Checks and task policy

```bash
uv run --frozen --project envs/nayana_ocr --extra dev --extra train pytest envs/nayana_ocr/tests -q
uv run --frozen --project envs/nayana_ocr nayana-smoke
uv run --frozen --project envs/nayana_ocr nayana-smoke \
  --url https://huggingenvs-nayana-ocr-env.hf.space --languages en kn hi ar

# Metadata addressability, one-block prefetch measurement, and all-language oracle checks.
uv run --frozen --project envs/nayana_ocr --extra train python train/verify_corpus.py \
  --url https://huggingenvs-nayana-ocr-env.hf.space \
  --manifest data/corpus-manifest.json --output results/corpus-hosted.json

# All-language Gradio scoring, indexed navigation, and independent sessions.
uv run --frozen --project envs/nayana_ocr python train/verify_playground.py \
  --url https://huggingenvs-nayana-ocr-env.hf.space \
  --manifest data/corpus-manifest.json --output results/gradio-corpus-hosted.json
```

For a local server, replace the URL with its address, for example `http://127.0.0.1:8005`.
The committed [results](results/README.md) record 38 passing tests, all 66 language/task
combinations through both local and hosted OpenEnv/Gradio, and all six notebook CPU cells.
The full-corpus verifier uses two independent consumers within the 16-session service limit.

The remote smoke selects indexed representatives and checks binary hashes, independent
WebSockets, repeated resets, empty-answer rewards, and client image reuse. Supplying a local
catalog to `probe` also checks known-reference reward1. It does not enumerate millions of IDs.
An all-language image sweep can fetch several GB on a cold cache; metadata checks fetch indexes
only. Recorded results distinguish metadata coverage, image/task samples, and exact oracle checks.

OCR uses NFC and collapsed whitespace while preserving case, diacritics, Indic vowel signs,
ZWJ/ZWNJ, punctuation, and RTL logical order. CER counts Unicode code points, not grapheme
clusters. Reward clips similarity at zero. Answers longer than `max(1024, 4 × raw reference
length)` score zero before normalization. MCQs require 2–26 unique nonempty options and exactly
one normalized source-answer match; ambiguous items are audited/excluded. Only one uppercase
option letter is accepted, with surrounding whitespace allowed.

Full-page OCR uses `whitespace-columns-v1` geometry, with RTL columns for Arabic. It masks
unannotated pixels and excludes incomplete/overlapping region annotations. It is not table
reconstruction or official semantic reading order. Because dimensions are absent from source
annotations, the full index validates metadata and actual image bounds are validated on reset.
Invalid candidates fail explicitly. Counts are not an audit of all source image quality.

The original Arabic JPEG for `document_10026_page_106` has missing/distorted glyphs and
replacement boxes before transformation (SHA-256
`c11acab6056c92836ac8ccd7470b76d04d20441485e9e46c1c3b9b4cb2ad0a62`).
This observation does not establish prevalence; the UI flags it and retains source pixels.

For a small self-contained offline fixture/window, `nayana-prepare` remains available:

```bash
uv run --frozen --project envs/nayana_ocr nayana-prepare \
  --output data/snapshots/real-smoke-v2 --pages-per-language 2
uv run --frozen --project envs/nayana_ocr nayana-smoke --snapshot data/snapshots/real-smoke-v2
```

That optional Datasets streaming workflow commits each page and iterator state transactionally;
its old 64-page deployment is historical evidence in `results/`. It is not the full-corpus path.
