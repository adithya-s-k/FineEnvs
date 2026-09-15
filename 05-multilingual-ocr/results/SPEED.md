# Data-path speed test

This measures the serving and training-adapter data path, including image download,
SHA-256 verification, RGB decoding, and exact-reference reward scoring. It does not
run a model, tensorize images with a model processor, or measure GPU utilization.

**Use a colocated bucket-mounted environment for a training pilot on vetted tasks.**
Cache reuse works; cold random remote access is costly. Unattended full-corpus runs
need an oversized-page policy before the data path can be considered ready.

## Measured results

| Workload | Local HTTP ranges | Space from laptop | Colocated HF CPU Job |
|---|---:|---:|---:|
| Cold random task, median | 17.09 s | 4.71 s | 2.44 s |
| Cold random task, p95 | 30.57 s | 6.39 s | 3.24 s |
| Prefetched new tasks/sec | 0.93 | 0.60 | 6.73 |
| 96 prefetched tasks, total | 103.71 s | 160.01 s | 14.27 s |
| Cached requests/sec, 4 workers | 394.6 | 8.9 | 57.5 |
| Cached request p95, 4 workers | 24.3 ms | 457.1 ms | 195.5 ms |
| Cached operations in sustained test | 11,880 | 308 | 1,760 |
| Balanced 12-ID sample, median | 5.1 ms | 270.9 ms | 13.3 ms |

These are observed operation rates, not model training examples/sec. Local hardware was an
Apple M5 Pro with 18 logical CPUs and 48 GiB RAM; the Space and job used `cpu-basic`.
The [CPU Job](https://huggingface.co/jobs/AdithyaSK/6aa961635527934177ee5574) completed at source
`642e13f29c3da43e4d0ade7eb051bc6349eb1b2d`. Its process completed normally while the benchmark
report correctly records the rejected candidate. Peak RAM/GPU utilization was not measured.

**119 of 120 distinct sampled tasks served successfully on all three deployments.** The
remaining task is the same oversized Kannada page. It is retained as a failure in cold and
short warm tests. Sustained capacity measurements explicitly exclude it after priming and
repeat the same 11 valid tasks for at least 30 seconds. Those sustained operations had no
errors, no source loads, and no image downloads. This is a hot working-set measurement.

All three prefetch workloads served 96 new task images from eight source groups with zero
task errors and cache sizes within their configured limits. Prefetch was not sufficient to
hide every cold read when consuming only 12 tasks per block without model computation.
The sampled blocks contain 192–612 indexed tasks each; a full block iterator gives lookahead
more consumption time than this deliberately sparse test.

The Space-to-laptop path had a 221 ms median health round trip. Warm reset plus scoring
requires separate exchanges, explaining much of its roughly 440 ms request latency.
For new images, inspection also shows that the current client holds one cache lock during
downloads and opens a fresh HTTP request for each miss. Connection pooling and per-image
load coalescing are useful next optimizations; this benchmark does not implement them.

## Training and evaluation decision

- **Pilot training:** use a vetted task set and colocate the OpenEnv server with the trainer
  using the bucket mount. The measured job achieved 6.73 new tasks/sec with prefetch and
  57.5 cached adapter requests/sec; actual GPU step time still determines whether data stalls
  are material. The trainer’s real scheduling and image tensorization are outside this test.
- **Evaluation:** freeze task IDs, validate eligibility, and warm the fixed set or its blocks
  before timed model comparisons. Keep data-loading time separate from model inference time.
  Avoid a cold, scattered remote lookup for every example in a latency-sensitive evaluation.
- **Unattended full corpus:** resolve oversized-page handling first. The selected Kannada
  `document_12413_page_0` is 9,934 × 7,016 pixels (69,696,944), exceeding the renderer and
  adapter’s 50-million-pixel limit. The corpus index includes annotation candidates, so the
  current full iterator can encounter this failure. A versioned resize policy or explicit
  eligibility filter is needed; do not silently replace evaluation items. This one finding
  does not estimate prevalence across the dataset. Details: `speed-readiness.json`.

## Reproduce

From `05-multilingual-ocr/`:

```bash
# Starts and closes an isolated local server with an empty application cache.
uv run --frozen --project envs/nayana_ocr --extra train python train/benchmark_corpus.py \
  --manifest data/corpus-index-v1 --label local-http-range \
  --output results/speed-local.json

# Uses the same task selection against the existing Space.
uv run --frozen --project envs/nayana_ocr --extra train python train/benchmark_corpus.py \
  --manifest data/corpus-index-v1 --label space-from-workstation \
  --url https://huggingenvs-nayana-ocr-env.hf.space \
  --output results/speed-hosted.json

# Sustained warm capacity: 30 seconds with four consumers, after explicit priming.
uv run --frozen --project envs/nayana_ocr --extra train python train/benchmark_corpus.py \
  --manifest data/corpus-index-v1 --label space-sustained-warm \
  --url https://huggingenvs-nayana-ocr-env.hf.space \
  --phases warm --warm-workers 4 --warm-seconds 30 \
  --output results/speed-hosted-sustained.json

# CPU-only colocated benchmark. Commit and push the code before launching.
SOURCE_REVISION=$(git rev-parse HEAD)
uv run --frozen --project envs/nayana_ocr hf jobs uv run \
  --flavor cpu-basic --timeout 20m \
  --volume hf://buckets/HuggingEnvs/NayanaOCR_Corpus_2025_bucket:/corpus:ro \
  train/benchmark_job.py --revision "$SOURCE_REVISION"
```

The local index directory is produced by the reproduction guide. The committed
`data/corpus-manifest.json` also works, but the oracle and local server must then
download language indexes. Index preparation is outside timed operation phases.
The default selection seed is `20260915`; exact task IDs and source block IDs are
recorded. Run the two commands sequentially to avoid competing for workstation
network bandwidth.

The job attaches the corpus read-only through the [HF Jobs volume API](https://huggingface.co/docs/huggingface_hub/guides/jobs).
It runs all phases plus 30 seconds of warm requests with four workers. The complete JSON
report is compressed into numbered `NAYANA_SPEED_REPORT_PART` log entries, with a SHA-256
header. Concatenate their base64 payloads in numeric order, base64-decode, zlib-decompress,
and verify the digest. `speed-job.json` is the decoded report from the recorded job.

## Recorded artifacts

| Artifact | Scope |
|---|---|
| `speed-summary.json` | Comparable metrics, exact sample size, and readiness decision |
| `speed-context.json` | Hardware, source commit, selection seed, and configured budgets |
| `speed-local-cold.json` | Fresh isolated local cache: both cold evaluation phases |
| `speed-local-warm.json` | Local warm bursts and eight-block prefetch, resumed on port 8005 |
| `speed-local-sustained.json` | At least 30 seconds of local cached requests with four workers |
| `speed-hosted.json` | All original phases against the Space from the workstation |
| `speed-hosted-sustained.json` | At least 30 seconds of hosted cached requests with four workers |
| `speed-job.json`, `speed-job-run.json` | Colocated CPU measurements, exact pushed commit, completed job and volume |
| `speed-readiness.json` | Rejected task ID, source location, dimensions, and pixel limit |

The initial local harness stopped when priming encountered the oversized task. Its completed
cold measurements were retained, and the remaining phases ran separately after changing the
harness to record failures and continue. The environment implementation was unchanged.
Warm priming is outside measured phases. The isolated local cold cache was removed on exit;
the resumed warm/prefetch tests used the ordinary local server and verified cache counters.
Sustained tests retain the first 256 detailed records while computing summaries over every
operation. Each report includes the measured harness SHA and exact task IDs.

## Workloads and interpretation

- **Cold evaluation:** 12 held-out tasks, one per language/task combination for
  English, Kannada, Hindi, and Arabic. Every task comes from a distinct source block.
  A second, disjoint 12-task set uses four workers. The datasets differ, so their
  ratio is not a controlled concurrency speedup measurement.
- **Warm GRPO-style requests:** the same 12 tasks are primed, then each is repeated
  four times. The same 48-operation workload is measured at one, four, and eight
  workers. Priming and connection setup are recorded separately. Independent adapter
  sessions measure service capacity; the actual TRL trainer may schedule them differently.
- **Training with prefetch:** eight fresh source blocks, two per language, with
  12 tasks per block, balanced across the three families. Two-block lookahead overlaps
  loading with four consumers. The measured wall time includes the first cold block
  and prefetch calls. There is no artificial model delay to hide I/O. This is a sampled
  locality workload, not the natural task distribution of a full training epoch.
- **Metadata:** 12 balanced indexed sample requests, each returning 12 IDs. Cache
  counters establish whether metadata access caused any image reads.

“Cold” means an application row-group cache miss. The bucket mount, object storage,
CDN, and OS caches are not flushed. Hosted tests are from the same workstation as
the local tests, and include its network round trips to the Space. They do not
represent a trainer colocated with the server. Both environments have bounded
caches; no production cache is flushed for the test.

Latency percentiles describe these small samples, not a production tail-latency
guarantee. Every timed record retains its task/family and any error. The source
retains its original Parquet row groups, so a cold task can read tens of megabytes
that adjacent tasks subsequently reuse.
