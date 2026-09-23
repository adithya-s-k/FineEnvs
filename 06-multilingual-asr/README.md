# 06 · Multilingual ASR

**Serve FLEURS speech through OpenEnv, scored per word or per character by script.**
The source is [Google's FLEURS](https://huggingface.co/datasets/google/fleurs): **102
languages**, 271,798 train / 34,452 validation / 77,810 test utterances, CC BY 4.0. The
pinned copy lives in the [FineEnvs bucket](https://huggingface.co/buckets/FineEnvs/fleurs-bucket)
(878 GB, 1,004 files), mirroring both the raw `data/` layout and `parquet-data/`.

| Task | Observation | Answer | Reward |
|---|---|---|---|
| `transcription` | 16 kHz utterance | Normalized transcript in the spoken language | `0.8 × max(0, 1−ER) + 0.2 × exact_match` |
| `verbatim_transcription` | Same audio | Transcript with capitalization, punctuation, numerals | Same, scored on unnormalized text |
| `language_id` | Same audio | FLEURS language code | Exact match |

`ER` is **word error rate**, except for scripts that do not delimit words with spaces,
where it is **character error rate**.

<!-- BEGIN:matrix -->
| Env | Tools | Backend | `openenv` |
|---|---|---|---|
| **multilingual_asr** | — | `http` | — |
<!-- END:matrix -->

## Scoring policy

The reward is the part of this environment worth arguing about, so it is explicit.

- **Error unit follows the script, not the task.** Word error rate on Chinese, Cantonese,
  Japanese, Thai, Lao, Burmese, or Khmer is degenerate: the reference is effectively one
  token, so any error at all scores 1.0. Those seven FLEURS languages are scored per
  character. Every observation carries the `error_unit` it will be graded in, and the
  metric is keyed `wer` or `cer` so a logged number always says which it is.
- **Normalization differs by family, because the targets differ.** `transcription` grades
  FLEURS' already-normalized field, so case and punctuation are stripped from both sides:
  a model should not lose reward for a comma the reference never had.
  `verbatim_transcription` grades `raw_transcription`, where case and punctuation *are*
  the task, so only Unicode form and whitespace are normalized. The same normalization is
  applied to the prediction and the reference, so it can only remove a difference both
  sides agree is not an error.
- **The rate is unbounded above; the reward is not.** Padding a transcript with filler
  inserts edits and pushes the rate past 1.0. The reward floors at zero rather than going
  negative, and the raw rate is still reported.
- **An unscorable reference is rejected, not scored.** A reference that normalizes to
  nothing cannot define an error rate, so it raises instead of silently scoring 0 or 1.
- `language_id` is matched exactly after trimming and case folding. It is a code, not
  prose; normalizing it as text would only mask a wrong answer.

Measured on real FLEURS test utterances: exact answers score 1.000, dropping one unit
scores 0.73–0.78, dropping half scores 0.38–0.40, and empty or wrong-language answers
score 0.000 in both error units.

## Data flow

**The whole corpus is addressable without copying it.** Each language gets a SQLite index
of every eligible utterance and where it physically lives; audio stays in the bucket and is
fetched on demand into a byte-bounded cache. The same manifest serves a local checkout, a
job colocated with a bucket mount, and the Space — so what the Space serves is what a
training or evaluation run sees: **1,151,940 tasks across 102 languages in every split**.

Indexing reads metadata columns only: 0.3 MB against 309.6 MB for one measured shard,
because each published file is a single row group that the audio column dominates. All 102
languages index in **143 seconds** into 257 MB.

That single-row-group layout is also the cost to respect at serve time. A shard is 310 MB
for test and up to 1.5 GB for train, so a cold read is expensive and a warm one is free:
measured, the first task of a language costs ~43 s and the next from the same shard costs
0.0 s. Task order therefore decides throughput — serve a language and split together.

FLEURS' own train/validation/test splits are used as published; re-splitting would silently
break comparison with every reported FLEURS score.

Three ways to run it, all on the same index:

| | Setup |
|---|---|
| Local, no copy | `FLEURS_CORPUS_MANIFEST=data/corpus-manifest.json`; audio over HTTP range |
| Local, synced | add `FLEURS_SOURCE_ROOT=/path/to/bucket` after `hf` sync; reads the mount |
| Space / job | bucket mounted read only at `/fleurs`, same manifest |

`asr-prepare` still builds a small self-contained snapshot for a machine with no bucket
access at all; it is the offline path, not the main one.

## Run it

```bash
# Build a bounded snapshot from the pinned bucket copy.
uv run --frozen --project envs/multilingual_asr asr-prepare \
  --output data/snapshots/dev --languages en_us hi_in cmn_hans_cn \
  --splits test --per-split 8

ASR_SNAPSHOT="$PWD/data/snapshots/dev" \
  uv run --frozen --project envs/multilingual_asr asr-server

# No download: synthetic HTTP/transport/reward smoke.
uv run --frozen --project envs/multilingual_asr asr-smoke
# Or against a prepared snapshot.
uv run --frozen --project envs/multilingual_asr asr-smoke --snapshot data/snapshots/dev
```

## Evaluation sets

Four frozen sets are committed, a **test** and a **validation** variant of each, so model
selection during training never touches the set a final number is reported on. The two
splits share no task.

| Set | Languages | Tasks |
|---|---|---|
| `eval-fleurs-all-{test,validation}.json` | 102 | 510 each |
| `eval-fleurs-ocr-overlap-{test,validation}.json` | 21 | 504 each |

Each is also served as **its own split**, named for what it covers, so a frozen set can
be browsed and served like any other split rather than only loaded from JSON:

| Split | Tasks |
|---|---|
| `train` / `validation` / `test` | 815,226 / 103,326 / 233,388 |
| `eval_21_test` / `eval_21_validation` | 504 each |
| `eval_102_test` / `eval_102_validation` | 510 each |

An eval split is a **view over the corpus, not a copy**: its tasks are the same rows the
source splits serve. Within a split the two sets nest: the overlap set's tasks are those of the
all-language set for its 21 languages. The overlap is 21 rather than 22 because FLEURS has
no Sanskrit. Selection reads metadata columns only — 0.3 MB against 309.6 MB per shard — so
both sets build in about 70 seconds; decoding every selected utterance is available through
`--verify`. See [REPRODUCE.md](REPRODUCE.md#5b-frozen-evaluation-sets).

## Limits

Utterances outside 0.5–60 seconds are excluded rather than truncated, and the exclusion is
counted in the snapshot manifest. This serves prepared snapshots; the indexed
whole-corpus path used by 05 is not implemented, and no Space is deployed yet. Audio is
served as stored, at FLEURS' 16 kHz; no resampling, denoising, or VAD is applied. Reward
numbers here describe scoring behaviour, not any model's accuracy. Source audio and
transcripts remain **CC BY 4.0** with Google attribution; see [data/README.md](data/README.md).
