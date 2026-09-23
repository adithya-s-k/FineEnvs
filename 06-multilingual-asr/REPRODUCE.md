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

## 5b. Frozen evaluation sets

Two sets are committed, built from one pass:

| File | Languages | Tasks | Per language | Per family |
|---|---|---|---|---|
| `data/eval-fleurs-all.json` | 102 | 510 | 5 | 170 |
| `data/eval-fleurs-ocr-overlap.json` | 21 | 504 | 24 | 168 |

The overlap set covers the languages shared with `05-multilingual-ocr`, so an ASR score can
be read against an OCR score language for language. **The overlap is 21, not 22:** the OCR
corpus includes Sanskrit and FLEURS does not, which the mapping records explicitly rather
than dropping silently.

**They nest.** Families take bands of the hash order fixed by family position rather than
by how many tasks are wanted, so a larger set is a superset of a smaller one for any shared
language. All 105 `fleurs-all` tasks in the 21 shared languages are also in the overlap
set, so the two agree on every task they share instead of sampling independently.

```bash
uv run --frozen --project envs/multilingual_asr asr-evalset build \
  --output-dir data --size 510 --overlap-size 504 --workers 12 --verify

uv run --frozen --project envs/multilingual_asr asr-evalset verify \
  --evalset data/eval-fleurs-ocr-overlap.json
```

**Selection reads metadata columns only.** Measured on one shard: 0.3 MB and 5.9s for the
metadata columns against 309.6 MB and 34.3s for the same rows with audio, because each
published file is a single row group and the audio column dominates it. Every eligibility
rule this environment applies — the duration policy from `num_samples`, and whether a
reference survives normalization — is decidable from metadata, so the cheap pass is also
the correct one. Both sets over all 102 languages build in about **70 seconds**.

The committed sets record `validated: false`: they are **selection verified** from
metadata, which covers every eligibility rule this environment applies - the duration
policy and whether a reference survives normalization - but does not decode the audio.

`--verify` additionally decodes every selected utterance, which does cost the full shard
per language (~31 GB over 102 languages). Whole-shard reads are bandwidth bound, so it
uses four streams by default: a first attempt at twelve exhausted fsspec's HTTP timeout
rather than finishing faster. Because the sets nest, the union is verified in
a single pass rather than reading a shared language's shard twice, and a set that records
any failure is refused by `load()` rather than quietly scored.

To evaluate a set, build a snapshot holding exactly its utterances:

```bash
uv run --frozen --project envs/multilingual_asr asr-prepare \
  --output data/snapshots/eval-overlap \
  --evalset data/eval-fleurs-ocr-overlap.json

uv run --frozen --project envs/multilingual_asr --extra train python train/grpo_asr.py \
  --snapshot data/snapshots/eval-overlap \
  --evalset data/eval-fleurs-ocr-overlap.json --max-steps 50
```

The runner uses a frozen set whole unless `--eval-limit` is given, and warns if the served
snapshot is missing any of its tasks, because a partial set is not comparable with a full
one.

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
