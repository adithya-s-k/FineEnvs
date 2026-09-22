# Reproduce the multilingual ASR environment

Every command runs from `06-multilingual-asr/`. The environment serves prepared snapshots;
nothing here downloads the whole 878 GB corpus.

## 1. Prepare a snapshot from the pinned bucket

```bash
uv run --frozen --project envs/multilingual_asr asr-prepare \
  --output data/snapshots/dev \
  --languages en_us hi_in cmn_hans_cn --splits test --per-split 8
```

Shards are discovered through the Hub API rather than by guessing the `of-000NN` suffix,
which differs per language. **Each published file is a single row group** — hi_in test is
418 rows in one 310 MB group — so `--per-split` bounds what the snapshot *stores*, not what
is transferred: preparing any number of utterances from a language/split reads that whole
shard. Budget roughly 310 MB per language for `test` and about 1.5 GB for `train`.

The verified run above took 144s and produced 72 tasks from 17.1 MB of stored audio, with
no exclusions.

## 2. Serve it

```bash
ASR_SNAPSHOT="$PWD/data/snapshots/dev" \
  uv run --frozen --project envs/multilingual_asr asr-server   # port 8006
```

`/healthz` returns the snapshot id, `/manifest` returns the snapshot plus the grading
policy, and `/assets/<sha256>` serves the audio with an ETag. Discovery
(`task_range`, `get_task`) never includes a reference and never reads an audio file.

## 3. Smoke

```bash
uv run --frozen --project envs/multilingual_asr asr-smoke            # synthetic fixtures
uv run --frozen --project envs/multilingual_asr asr-smoke \
  --snapshot data/snapshots/dev                                      # real FLEURS audio
```

The smoke walks every language/family group and checks, over real HTTP and two independent
sessions, that: a task resets identically in both, discovery leaks no reference and no
metrics, the served audio hashes to the advertised SHA-256 with the advertised media type,
the exact reference scores **1.0** with `exact_match`, and an empty answer scores **0.0**.
It reports which error units were actually exercised, so a run that happened to touch only
word-scored languages cannot look like full coverage.

Verified on real FLEURS test audio: all 9 groups across `en_us`, `hi_in`, and
`cmn_hans_cn` pass, exercising 4 word-scored and 2 character-scored gradings.

## 4. Checks

```bash
uv run --frozen --project envs/multilingual_asr --extra dev pytest envs/multilingual_asr/tests -q
uv run --frozen --project envs/multilingual_asr --extra dev ruff check envs/multilingual_asr
```

17 tests cover the scoring policy and the environment contract: word against character
scoring and the degenerate word rate it avoids, per-family normalization, punctuation
counting for `verbatim_transcription` but not `transcription`, reward clamping under
padding, rejection of an unscorable reference, numeric-only metrics, single-grade episodes,
a failed reset leaving nothing gradable, content-addressed playable audio, and exclusion of
utterances outside the duration policy.

## 5. Scoring reference

| | Unit | Normalization |
|---|---|---|
| `transcription` | wer, or cer for spaceless scripts | NFKC, punctuation stripped, case folded |
| `verbatim_transcription` | wer, or cer for spaceless scripts | NFKC and whitespace only |
| `language_id` | exact match | trim and case fold |

Character-scored languages: `cmn_hans_cn`, `yue_hant_hk`, `ja_jp`, `th_th`, `lo_la`,
`my_mm`, `km_kh`. Measured on real test utterances, an exact answer scores 1.000, dropping
one unit scores 0.73–0.78, dropping half scores 0.38–0.40, and an empty or wrong-language
answer scores 0.000 in both units.

## 6. Verified single-GPU smoke

Both Gemma 4 sizes run end to end on one A100: the job builds its snapshot from the
attached bucket mount, serves it locally, evaluates, trains, and evaluates again.
Summaries are committed under `results/`.

```bash
REV=$(git rev-parse HEAD)          # must be pushed: the job fetches this exact commit
uv run --frozen --project envs/multilingual_asr hf jobs uv run \
  --flavor a100-large --timeout 2h --secrets HF_TOKEN \
  --volume hf://buckets/FineEnvs/fleurs-bucket:/fleurs:ro \
  train/hf_job.py --revision "$REV" --mode train --source-root /fleurs \
  --languages en_us hi_in --splits train test --per-split 8 \
  --model google/gemma-4-E2B-it --smoke --families transcription
```

A CPU job runs the tests and the fixture smoke without a GPU:

```bash
uv run --frozen --project envs/multilingual_asr hf jobs uv run \
  --flavor cpu-basic --timeout 30m --secrets HF_TOKEN \
  train/hf_job.py --revision "$REV" --mode env-smoke
```

| | E2B | E4B |
|---|---|---|
| Status | passed | passed |
| Optimizer steps | 2 | 2 |
| Training loss | 0.02382 | 0.03651 |
| Adapter weights changed | yes | yes |
| LoRA target modules | 106 | 122 |
| `en_us` WER | 0.0952 | 0.0952 |
| `hi_in` WER | 0.0417 | 0.0000 |
| Macro reward, before -> after | 0.7452 -> 0.7452 | 0.7286 -> 0.8619 |

**This verifies execution, not quality.** Two optimizer steps over one utterance per
language cannot establish an improvement; the E4B before/after difference is one sample
changing, which is noise at this size and must not be read as a training result.

Unlike the OCR environment, the audio tower is **adapted, not skipped**: it is the part of
an omni checkpoint this task depends on, which is why the target counts here (106 and 122)
are the full sets rather than the 82 and 98 used there.

Four things the smoke settled that reading the code did not:

- **FLEURS is IEEE float WAV, not PCM.** The decoder used the standard library, which
  rejects that subtype outright. The whole suite passed because the fixtures were PCM, and
  the first GPU run died on real audio with `unknown format: 3`. Fixtures now write float
  WAV like the corpus and a test asserts they do.
- **`environment_factory` returns one environment per call**, not a pool.
- **A `prompt` column in the dataset is read by TRL as a conversation.** The environment
  owns the prompt; the sampler gets only immutable identifiers.
- Preparing from an attached bucket mount avoids pulling shards through the Hub, which
  matters because each file is a single row group.

## Not done yet

No Space is deployed and no GRPO recipe is wired up: this PR establishes the environment,
its scoring policy, and its verification. The indexed whole-corpus path used by
`05-multilingual-ocr` is not implemented, and would need a different design here because
FLEURS' single-row-group files defeat row-group caching.
