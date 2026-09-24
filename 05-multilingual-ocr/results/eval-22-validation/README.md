# Gemma 4 on the 22-language evaluation set

Both Gemma 4 sizes on all 1,100 tasks of `eval_22_validation` — **50 tasks per language**
across 22 languages and five families, half of it Indic script (11 languages, 550 tasks).
Identical prompt from the environment, graded by it, greedy through vLLM. No task failed.

The **validation** variant is scored so `eval_22_test` stays untouched for a post-training
number. `eval_21` nests inside this set, so its score falls out of the same run.

| | gemma-4-E4B-it | gemma-4-E2B-it |
|---|---|---|
| **Macro (`eval_22`, 1100 tasks)** | 0.4115 | 0.3565 |
| Macro (`eval_21` subset, 1050) | 0.4163 | 0.3573 |
| Indic (11 langs, 550 tasks) | 0.3483 | 0.2962 |
| Non-Indic (11 langs, 550 tasks) | 0.4746 | 0.4168 |

| Family | gemma-4-E4B-it | gemma-4-E2B-it |
|---|---|---|
| `mcq_vqa` | 0.7318 | 0.6636 |
| `section_ocr` | 0.6226 | 0.5502 |
| `page_ocr` | 0.5347 | 0.4505 |
| `descriptive_vqa` | 0.1682 | 0.1182 |
| `layout_detection` | 0.0000 | 0.0000 |

**E4B wins on every family and both script groups**, the same result the ASR comparison
reached, and both models land within 0.006 of an earlier run on the retired 500-task set —
so doubling the per-language sample confirmed the picture rather than changing it.

## `layout_detection` still scores zero

Unchanged and still not a grading fault: the models emit well-formed layout JSON in their
own schema — `box_2d` with 0-1000 normalized coordinates — where the prompt asks for `bbox`
in absolute pixels on a canvas whose size it states. A family where every rollout scores
exactly 0.0 has no reward variance, so GRPO gets no gradient from it. It cannot be trained
by this reward as written.

## Cost, measured

| Phase | First run | With the page cache |
|---|---|---|
| Fetch 1,100 pages | 2,317s (26.3 GB read) | **222s** (1,100 from cache) |
| Score one model | ~1,200s | ~1,200s |

The cache removed a prefetch that was reading **23.9 MB per page served**: the corpus packs
~100 pages per parquet row group, and this set deliberately spreads across 342 blocks for
document diversity, so extracting 1,100 pages meant reading 26.3 GB of row groups.

What remains is the judge. `descriptive_vqa` is 220 of the 1,100 tasks and nearly all of the
wall clock: raising grading from 8 to 16 workers moved generation only from 1.0 to 1.3
task/s, so the judge API is rate-limiting rather than worker-starved. For per-checkpoint
scoring, run the four locally-graded families and keep the judge for checkpoints that
matter:

```bash
  --families section_ocr mcq_vqa page_ocr layout_detection
```

## Reproduce

```bash
REV=$(git rev-parse HEAD)          # must be pushed: the job fetches this exact commit
uv run --frozen --project envs/nayana_ocr hf jobs uv run \
  --flavor a100-large --timeout 3h --secrets HF_TOKEN \
  --volume hf://buckets/FineEnvs/NayanaOCR_Corpus_2025_bucket:/corpus:ro \
  --volume hf://buckets/FineEnvs/nayana-eval-pages:/pagecache \
  train/hf_job.py --revision "$REV" --mode eval-vllm \
  --corpus-manifest repo --source-root /corpus \
  --evalset eval-nayana-all-validation.json \
  --artifact-repo FineEnvs/nayana-eval-results \
  --models google/gemma-4-E2B-it google/gemma-4-E4B-it \
  --page-cache /pagecache --workers 48 --grade-workers 16
```
