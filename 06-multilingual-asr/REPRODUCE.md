# Reproduce the multilingual ASR environment

Every command runs from `06-multilingual-asr/`. The whole 878 GB corpus is **addressable
without copying it**: `data/corpus-manifest.json` indexes every eligible utterance and where
it physically lives, and audio is fetched from the bucket on demand into a bounded cache.

## 1. Serve the corpus

```bash
FLEURS_CORPUS_MANIFEST="$PWD/data/corpus-manifest.json" \
  uv run --frozen --project envs/multilingual_asr asr-server   # port 8006
```

`/healthz` returns the snapshot id, `/manifest` returns the index plus the grading policy
and the frozen evaluation splits, `/group_count` and `/group_tasks` address tasks by
language and family, and `/assets/<sha256>?task_id=…` serves the audio with an ETag.
Discovery (`task_range`, `get_task`) never includes a reference and never reads audio.

Three ways to run it, all on the same index:

| | Setup |
|---|---|
| Local, no copy | `FLEURS_CORPUS_MANIFEST=data/corpus-manifest.json`; audio over HTTP range |
| Local, synced | add `FLEURS_SOURCE_ROOT=/path/to/bucket` after `hf` sync; reads the mount |
| Space / job | bucket mounted read only at `/fleurs`, same manifest |

The deployed [Space](https://huggingface.co/spaces/FineEnvs/fleurs-asr-env) runs this same
manifest, so what it serves is what a training or evaluation run sees.

## 2. Rebuild the index (optional)

```bash
uv run --frozen --project envs/multilingual_asr asr-index \
  --output data --source-root /fleurs --workers 12
```

Indexing reads metadata columns only: 0.3 MB against 309.6 MB for one measured shard,
because each published file is a single row group that the audio column dominates. All 102
languages index in **143 seconds** into 257 MB, giving **1,151,940 tasks**: 815,226 train,
103,326 validation, 233,388 test.

That single-row-group layout is the cost to respect at serve time. A shard is 310 MB for
test and up to 1.5 GB for train, so a cold read is expensive and a warm one is free:
measured, the first task of a language costs ~43 s and the next from the same shard costs
0.0 s. Task order therefore decides throughput — serve a language and split together.

**Identity comes from `path`, not `id`.** FLEURS' `id` is a *sentence* id: several speakers
record the same sentence, so hi_in test holds 418 rows under 265 ids. Keying on it would
have collapsed 37% of the corpus and given two different clips one task id.

## 3. Smoke

```bash
uv run --frozen --project envs/multilingual_asr asr-smoke            # synthetic fixtures
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

41 tests cover the scoring policy, the corpus index, the frozen sets, and the environment
contract: word against character scoring and the degenerate word rate it avoids, per-family
normalization, punctuation counting for `verbatim_transcription` but not `transcription`,
reward clamping under padding, rejection of an unscorable reference, numeric-only metrics,
single-grade episodes, a failed reset leaving nothing gradable, content-addressed playable
audio, the float-WAV encoding FLEURS actually publishes, group-addressed task selection,
seeded sampling that never lists a split, evaluation splits as views rather than copies,
and the name collision two frozen sets covering the same languages would otherwise cause.

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

## 6. Frozen evaluation sets

Four sets are committed, a **test** and a **validation** variant of each, so model selection
during training never touches the set a final number is reported on. The two splits share no
task.

| File | Languages | Tasks | Per language | Per family |
|---|---|---|---|---|
| `data/eval-fleurs-all-{test,validation}.json` | 102 | 510 | 5 | 170 |
| `data/eval-fleurs-ocr-overlap-{test,validation}.json` | 21 | 504 | 24 | 168 |

The overlap set covers the languages shared with `05-multilingual-ocr`, so an ASR score can
be read against an OCR score language for language. **The overlap is 21, not 22:** the OCR
corpus includes Sanskrit and FLEURS does not, which the mapping records explicitly rather
than dropping silently.

**They nest.** Families take bands of the hash order fixed by family position rather than
by how many tasks are wanted, so a larger set is a superset of a smaller one for any shared
language. All 105 `fleurs-all` tasks in the 21 shared languages are also in the overlap
set, so the two agree on every task they share instead of sampling independently.

**Each is served as its own split**, named for what it covers, so a frozen set is browsed
and served like any other split rather than only loaded from JSON:

| Split | Tasks |
|---|---|
| `train` / `validation` / `test` | 815,226 / 103,326 / 233,388 |
| `eval_21_test` / `eval_21_validation` | 504 each |
| `eval_102_test` / `eval_102_validation` | 510 each |

An evaluation split is a **view over the corpus, not a copy**: its tasks are the same rows
the source splits serve. Two sets covering the same languages and split would resolve to one
name, so the server refuses to start rather than serve whichever one won.

```bash
# One pass per source split; each writes both the 102-language and the overlap set.
for SPLIT in test validation; do
  uv run --frozen --project envs/multilingual_asr asr-evalset build \
    --output-dir data --split "$SPLIT" --size 510 --overlap-size 504 --workers 4 --verify
done

uv run --frozen --project envs/multilingual_asr asr-evalset verify \
  --evalset data/eval-fleurs-ocr-overlap-test.json
```

**Selection reads metadata columns only.** Every eligibility rule this environment applies —
the duration policy from `num_samples`, and whether a reference survives normalization — is
decidable from metadata, so the cheap pass is also the correct one. All four sets build in
about **70 seconds**.

The committed sets record `validated: false`: they are **selection verified** from metadata,
which covers every eligibility rule, but does not decode the audio. `--verify` additionally
decodes every selected utterance, which does cost the full shard per language (~31 GB over
102 languages). Whole-shard reads are bandwidth bound, so it uses four streams by default: a
first attempt at twelve exhausted fsspec's HTTP timeout rather than finishing faster. Because
the sets nest, the union is verified in a single pass rather than reading a shared language's
shard twice, and a set that records any failure is refused by `load()` rather than quietly
scored.

## 7. Verified single-GPU smoke

Both Gemma 4 sizes run end to end on one A100 against the indexed corpus: the job reads the
committed manifest, serves it locally from the attached bucket mount, evaluates on a frozen
split, trains, and evaluates again. Summaries are committed under `results/`.

```bash
REV=$(git rev-parse HEAD)          # must be pushed: the job fetches this exact commit
uv run --frozen --project envs/multilingual_asr hf jobs uv run \
  --flavor a100-large --timeout 3h --secrets HF_TOKEN \
  --volume hf://buckets/FineEnvs/fleurs-bucket:/fleurs:ro \
  train/hf_job.py --revision "$REV" --mode train --source-root /fleurs \
  --model google/gemma-4-E2B-it --smoke --families transcription \
  --languages en_us hi_in --eval-split eval_21_test --eval-limit 8 \
  --num-generations 8 --max-steps 4
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
| Optimizer steps | 4 | 4 |
| Training loss | 0.02406 | 0.01089 |
| Adapter weights changed | yes | yes |
| Reward group without variance | no | no |
| LoRA target modules | 106 | 122 |
| Macro reward on `eval_21_test` (8 tasks), before -> after | 0.3092 -> 0.3092 | 0.3352 -> 0.3352 |

**This verifies execution, not quality.** Four optimizer steps over eight evaluation tasks
cannot establish an improvement, and the macro reward is unchanged in both runs.

Unlike the OCR environment, the audio tower is **adapted, not skipped**: it is the part of
an omni checkpoint this task depends on, which is why the target counts here (106 and 122)
are the full sets rather than the 82 and 98 used there.

Five things the smokes settled that reading the code did not:

- **FLEURS is IEEE float WAV, not PCM.** The decoder used the standard library, which
  rejects that subtype outright. The whole suite passed because the fixtures were PCM, and
  the first GPU run died on real audio with `unknown format: 3`. Fixtures now write float
  WAV like the corpus and a test asserts they do.
- **The trainer must ask for audio by task, not by hash.** A snapshot stores one file per
  hash; the indexed corpus has to know the task before it can find the row group. Every
  fetch returned 500 until the task id travelled with the request.
- **Four generations is not enough for reward variance.** On an easy English clip all four
  completions scored identically, the advantage was zero, and the runner failed the check
  rather than reporting a pass. Eight generations produced variance on the same tasks.
- **`environment_factory` returns one environment per call**, not a pool.
- **A `prompt` column in the dataset is read by TRL as a conversation.** The environment
  owns the prompt; the sampler gets only immutable identifiers.

`language_id` scored 0.0 on all three of its evaluation tasks in both runs, and the
predictions show why: the models answered `ar_sa` for `ar_eg` and `es_es` for `es_419`.
They identified the language and missed the FLEURS locale. Exact match on the locale is the
policy — the corpus distinguishes those variants — so this is the task being hard, not the
reward misfiring.

## 8. Deploy

```bash
uv run --frozen --project envs/multilingual_asr python train/deploy_space.py \
  --space-id FineEnvs/fleurs-asr-env \
  --corpus-manifest data/corpus-manifest.json \
  --evalset data/eval-fleurs-all-test.json \
  --evalset data/eval-fleurs-ocr-overlap-test.json \
  --evalset data/eval-fleurs-all-validation.json \
  --evalset data/eval-fleurs-ocr-overlap-validation.json
```

The Space bundles code, the manifest, and the evaluation sets; indexes and audio are fetched
lazily from the bucket mounted at `/fleurs`. Evaluation sets a deployment no longer carries
are deleted, because an upload adds and replaces but never removes and a leftover set
collides with its renamed successor under the same split name.
