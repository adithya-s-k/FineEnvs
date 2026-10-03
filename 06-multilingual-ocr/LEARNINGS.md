# What the Kannada run and the environment taught us

Held-out numbers here come from the 300 Kannada crops of Sarvam Indic OCR Bench (`test`), scored by
the benchmark's own `metrics.py`. The untuned and the trained model are scored on the same vLLM
engine, and changes are paired crop by crop. Every crop's prediction is in
[FineEnvs/multilingual-multimodal-rl-runs](https://huggingface.co/datasets/FineEnvs/multilingual-multimodal-rl-runs).

## The headline

500 steps of GRPO on 4,000 Kannada crops from the Nayana corpus cut Sarvam CER from 0.4277 to
0.3601: a paired change of −0.068, 95% CI (−0.083, −0.052). The benchmark is a different source
from the training data, so this measures transfer, not memory. Most of the gain is in the first 100
steps, and nothing improves after step 300.

## 1. The first thing it fixed was the script

The untuned model writes letters from another Indic script into Kannada text, and occasionally
whole Latin fragments. That happens on 37 of 300 benchmark crops. After training, it happens on 15.
Outputs that loop or collapse, by Sarvam's own detector, fall from 16 to 10.

The ASR sibling learned the same thing first. A reward that counts characters charges for a wrong
script on every letter, so it is the first habit to go.

## 2. Then it stopped

After step 300 the held-out CER sits at 0.360 ± 0.001 for 200 more steps. Training reward had
already levelled off at about 0.59 by step 150. Word error barely moved over the whole run (0.773 to 0.748), and no crop is
transcribed exactly, before or after.

The remaining errors are a letter or two inside most words. A character reward pays for each one
fixed, but slowly, and a word metric gives nothing until the word is entirely right. More steps on
the same mix are unlikely to help. More varied crops, or a larger model, are the next things to
try.

## 3. More crops or more samples: we chose crops

The run had about 12 hours. An earlier run used 16 transcriptions per crop and 4 crops per step at
about 65 seconds a step, which would have covered about 2,500 distinct crops. This one used 8 and 8:
the same 64 completions per step and nearly the same cost, about 75 seconds of wall clock, for twice
the distinct crops. Eight samples were enough for spread: on average, under 2% of groups had every
sample score the same.

Covering the 164,366 Kannada crops in Nayana's train split once would take about 20,500 steps at
this rate, roughly 18 days on one GPU. The 20,000 crops first asked for would take about 52 hours.

## 4. A benchmark has to be served, not just loaded

We served Sarvam Indic OCR Bench as evaluation-only splits of the same environment, rather than
scoring it with a separate script. That way a benchmark crop is an ordinary task with the same
prompt and reward, and a number on the benchmark means the same thing as a number on the corpus.
Sarvam's `metrics.py` is copied in unmodified and its CER and WER are reported beside the reward,
never in it.

Three things turned up while publishing it:
- **One test row has no image.** `indic_ocr_bench_test_eng_5` has ground truth and an empty image
  field, so the test split is served as 6,908 of 6,909 rows. The exclusion is pinned in code, and a
  rebuild that finds any other missing image stops.
- **Some "PNG" crops are JPEG.** The dataset card says PNG. 141 of the small split's 1,173 crops are
  JPEG, so the image type is read from the bytes.
- **It is served from a bucket, like the corpus.** 6,648 crops by hash plus a small index per split.
  The Space and the training jobs mount it, and a local server downloads a split on first use.

## 5. Score checkpoints while the run trains

A second job follows the run's bucket and scores each checkpoint as it lands, so the curve is
readable a few minutes after each save instead of after the run. Two things made it reliable:
- **Each save writes a manifest.** `ready.json` records the adapter's size and hash. The evaluator
  scores an adapter only when every byte matches, so it never reads a half-uploaded file.
- **The smoke test caught a real bug.** The evaluator starts vLLM before any checkpoint exists, and
  LoRA was only switched on when adapters were named at launch. The first checkpoint failed to
  load. A four-step training run plus its evaluator, launched before the long run, found it.

The evaluator also waited 1.5 hours for an L40S that never came. It resumes from its own
`curve.json`, so relaunching it on an A100 cost nothing but the wait.

## 6. From building the environment

These came from serving a million pages before any training run.

- **A renamed bucket breaks mounts, not HTTP.** After the `HuggingEnvs` to `FineEnvs` rename, HTTP
  reads followed a redirect and kept working, so local runs never noticed. The Space mounts the
  bucket, and every task load failed in 0.2 seconds while the health check stayed green. The deploy
  script now resolves the bucket's current name, and the recorded provenance stays untouched.
- **Picking evaluation tasks by density picks the same documents.** The corpus is largely parallel
  translations, so the densest blocks are the same documents in every language: ordering by density
  put all 22 languages onto 28 documents. Ordering by hash covers 181, and reads less data.
- **Some pages cannot be rendered.** The corpus has scans as large as 14,044 × 9,934 pixels, past the
  50-megapixel limit, and that is only known once the image is decoded. Evaluation sets now draw
  seeded spares and record what was replaced, so a rebuild reproduces the same set.
- **The judge has a best concurrency, and it is 16.** Descriptive-VQA grading ran at 0.47, 1.60 and
  0.64 calls per second at 4, 16 and 32 concurrent calls. Past 16 the provider queues. 9 of 100
  gradings failed at 16 workers, all on non-Latin scripts, and all 9 passed when retried, so every
  grading path now retries transient failures.
- **Evaluate locally, train next to the bucket.** A local server answers at 436 requests a second on
  one worker; the hosted Space manages 0.77. A cold block reads in about 4.4 seconds from a mount
  and about 31 over HTTP.

## Known limits

- Rendered PNGs are identical pixel for pixel across deployments but not byte for byte: the
  Space's encoder writes 40,280 bytes where a laptop writes 38,482. Evaluation sets are checked on
  pixels.
- Gemma 4 pools any image to at most 280 soft tokens, roughly 800 × 800. That suits section crops
  and is tight for dense full pages.
- Both Gemma 4 sizes score zero on layout detection out of the box. They do not produce valid JSON
  boxes.
- Training runs on one GPU.
- Only Kannada section OCR has a trained result. The other languages and task families have smoke
  runs.
