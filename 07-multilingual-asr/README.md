# 07 · Multilingual ASR

<div align="center">

[![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-Multilingual%20Multimodal%20Envs-FFD21E?style=for-the-badge&labelColor=1a1a1a)](https://huggingface.co/collections/FineEnvs/multilingual-multimodal-envs-6ac0c27c137f93e0799603e4)
[![Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-fleurs--asr--env-FFD21E?style=for-the-badge&labelColor=1a1a1a)](https://huggingface.co/spaces/FineEnvs/fleurs-asr-env)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-kannada--asr--grpo-FFD21E?style=for-the-badge&labelColor=1a1a1a)](https://huggingface.co/FineEnvs/gemma-4-E4B-it-kannada-asr-grpo)
[![Trackio](https://img.shields.io/badge/%F0%9F%93%88%20Trackio-runs-FFD21E?style=for-the-badge&labelColor=1a1a1a)](https://huggingface.co/spaces/FineEnvs/multilingual-multimodal-trackio)

</div>

> **All of FLEURS, 102 languages and 1.15M tasks, behind one OpenEnv server, and a Kannada speech model that learned to listen.**

<div align="center">

<img src="./assets/curves.gif" alt="Training reward and held-out CER/WER over 575 GRPO steps" width="100%">

<sub>Gemma 4 E4B, GRPO on every Kannada training clip in FLEURS. Right: the 838 Kannada test clips it never trains on.</sub>

</div>

The environment serves [Google's FLEURS](https://huggingface.co/datasets/google/fleurs) in place:
**102 languages**, 271,798 train / 34,452 validation / 77,810 test utterances (CC BY 4.0), from a
pinned copy in the [FineEnvs bucket](https://huggingface.co/buckets/FineEnvs/fleurs-bucket). Audio is
fetched per task; nothing is downloaded up front.

| Part | Where |
|---|---|
| Environment: server, catalog, rewards, playground, Docker, tests | [`envs/multilingual_asr/`](./envs/multilingual_asr/) |
| GRPO trainer, HF Jobs launcher, live checkpoint evaluation, deployment | [`train/`](./train/) |
| Kannada run: per-checkpoint scores, curves, launch commands | [`results/kannada-grpo/`](./results/kannada-grpo/) |
| Published index manifest and frozen evaluation sets | [`data/`](./data/) |
| Exact commands, defaults, provenance | [`REPRODUCE.md`](./REPRODUCE.md) |
| Hosted environment and playground | [FineEnvs/fleurs-asr-env](https://huggingface.co/spaces/FineEnvs/fleurs-asr-env) |
| Trained adapter | [FineEnvs/gemma-4-E4B-it-kannada-asr-grpo](https://huggingface.co/FineEnvs/gemma-4-E4B-it-kannada-asr-grpo) |
| Every held-out prediction, curves, logs, job scripts | [FineEnvs/multilingual-multimodal-rl-runs](https://huggingface.co/datasets/FineEnvs/multilingual-multimodal-rl-runs) |
| Training and evaluation dashboard | [FineEnvs/multilingual-multimodal-trackio](https://huggingface.co/spaces/FineEnvs/multilingual-multimodal-trackio) |
| Collection, with the OCR sibling project | [Multilingual Multimodal Envs](https://huggingface.co/collections/FineEnvs/multilingual-multimodal-envs-6ac0c27c137f93e0799603e4) |

## Result: Kannada speech recognition

`google/gemma-4-E4B-it` with a LoRA adapter, trained by GRPO on all 2,282 FLEURS `kn_in` training
clips, one epoch (575 steps) on one A100 in 4.7 hours. A second job scored every checkpoint on all
838 `kn_in` test clips as it was saved.

| FLEURS `kn_in` test · 838 clips | base | step 575 | change (95% CI) |
|---|---:|---:|---:|
| **CER**, per clip, capped at 1 | 0.1047 | **0.0571** | **−0.048** (−0.057, −0.039) |
| WER, per clip, capped at 1 | 0.312 | 0.243 | −0.069 (−0.079, −0.059) |
| median CER | 0.0676 | 0.0399 | |
| exact transcripts | 3.3% | 8.0% | |
| transcripts containing another Indic script | 174 (20.8%) | 13 (1.6%) | |
| clips that loop (CER above 1) | 10 | 1 | |

Changes are paired clip by clip against the untuned base on the same engine. 565 clips improve,
139 get worse. Most of the gain comes in the first 150 steps, and the curve is flat after about
step 250. Every checkpoint is in [`results/kannada-grpo/`](./results/kannada-grpo/).

**What it learned first: stay in the script.** One transcript in five from the untuned model
switches script mid-word, writing a Gujarati or Malayalam letter where the Kannada one belongs.
After training it is one in sixty. A clip at the median improvement:

| | text | CER |
|---|---|---:|
| reference | **ತಾಂತ್ರಿಕ** ನಿರ್ಣಾಯಕತೆಯ ಹೆಚ್ಚಿನ ವ್ಯಾಖ್ಯಾನಗಳು … **ಅವುಗಳೆಂದರೆ** ತಂತ್ರಜ್ಞಾನದ … | |
| base | **ತಾന്ത്രിಕ** ನಿರ್ಣಾಯಕತೀಯ ಹೆಚ್ಚಿನ ವ್ಯಾಖ್ಯಾನಗಳು … **ಅವು ಎಂದರೆ** ತಂತ್ರಜ್ಞಾನದ … | 0.047 |
| step 575 | **ತಾಂತ್ರಿಕ** ನಿರ್ಣಾಯಕತೆಯೇ ಹೆಚ್ಚಿನ ವ್ಯಾಖ್ಯಾನಗಳು … **ಅವುಗಳೆಂದರೆ** ತಂತ್ರಜ್ಞಾನದ … | 0.010 |

More examples, chosen by percentile, are on the
[model card](https://huggingface.co/FineEnvs/gemma-4-E4B-it-kannada-asr-grpo).

### What made it work

Earlier Kannada runs raised training reward while the held-out reward barely moved. Three
fixes, each measured, changed that:

1. **The loss has to hear the clip.** TRL 1.13 passes image features into the forward pass the
   loss is computed from, and drops audio features. GRPO was raising p(transcript | no audio): a
   language prior over the training sentences, not listening. `AudioGRPOTrainer` in
   [`envs/multilingual_asr/training.py`](./envs/multilingual_asr/training.py) carries the clip into
   every log-prob forward and checks on the first batch that it arrives. Measured on that batch:
   the sampled transcripts score **−0.38 nats/token with the clip and −6.22 without it**, a gap
   the old gradient never saw.
2. **Reward characters, not words.** A Kannada word is a long inflected string, so fixing three
   letters of it scored nothing under word error. An earlier run cut CER 14% while WER moved 2%.
   The `cer` policy (`ASR_REWARD_UNIT=cer`) rewards characters; WER is still reported.
3. **Pad features, not audio.** Batching padded the waveform to 30 s, which the feature extractor
   marked as valid speech. Features are now padded with the padding masked, so the model hears
   exactly what vLLM serves at evaluation.

### How it trains

<img src="./assets/how-it-trains.gif" alt="One GRPO step, stage by stage, beside the code that runs it" width="100%">

The environment owns the task and the reward; the trainer only ever sees task IDs. TRL's
`GRPOTrainer` resets one OpenEnv session per rollout through `environment_factory`, samples 16
transcripts per clip, and asks the server to grade each. Each saved checkpoint writes a hash
manifest beside its adapter; a second job (`eval_vllm.py --watch`) follows the run's bucket, loads
each complete adapter into one vLLM engine, scores it on the held-out set, and logs the curve,
plots and paired intervals to Trackio. Commands are in [`train/README.md`](./train/README.md).

## Tasks and reward

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
| **multilingual_asr** | — | `http` | ✅ |
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
- **Two policies, one set of metrics.** The default, `fleurs-asr-error-rate-v1`, rewards
  the script's natural unit. A word rate is coarse on an agglutinative language, though.
  One Kannada word is a long inflected string, so fixing three of its letters earns
  nothing. A Kannada run cut character errors 14% while word errors moved 2%. With
  `ASR_REWARD_UNIT=cer` the server grades by `fleurs-asr-cer-v1` instead: characters for
  every language, spaces included, so a transcript cannot gain by dropping word boundaries.
  Spaced scripts report both `wer` and `cer` under either policy, and the observation's
  `error_unit` and `grading_policy_id` name the one rewarded.
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
access at all. It is the offline fallback, not the path training and evaluation use.

## Run it

```bash
# Serve the whole corpus from the committed index. Nothing is downloaded up front.
FLEURS_CORPUS_MANIFEST="$PWD/data/corpus-manifest.json" \
  uv run --frozen --project envs/multilingual_asr asr-server   # port 8006

# No download at all: synthetic HTTP/transport/reward smoke.
uv run --frozen --project envs/multilingual_asr asr-smoke
```

The deployed Space runs this same manifest, so
[FineEnvs/fleurs-asr-env](https://huggingface.co/spaces/FineEnvs/fleurs-asr-env) serves
exactly what a local run does. See [REPRODUCE.md](REPRODUCE.md).

## Evaluation sets

Six frozen sets are committed, a **test** and a **validation** variant of each, so model
selection during training never touches the set a final number is reported on. The two
splits share no task.

| Set | Languages | Tasks |
|---|---|---|
| `eval-fleurs-all-{test,validation}.json` | 102 | 1,836 each (18 per language) |
| `eval-fleurs-ocr-overlap-{test,validation}.json` | 21 | 1,050 each (50 per language) |
| `eval-fleurs-kn-transcription-{test,validation}.json` | Kannada, transcription | 838 / 200 (the whole split) |

Each is also served as **its own split**, named for what it covers, so a frozen set can
be browsed and served like any other split rather than only loaded from JSON:

| Split | Tasks |
|---|---|
| `train` / `validation` / `test` | 815,226 / 103,326 / 233,388 |
| `eval_21_test` / `eval_21_validation` | 1,050 each |
| `eval_102_test` / `eval_102_validation` | 1,836 each |
| `eval_1_test` / `eval_1_validation` | 838 / 200 |

An eval split is a **view over the corpus, not a copy**: its tasks are the same rows the
source splits serve. Within a split the two sets nest: the overlap set's tasks are those of the
all-language set for its 21 languages. The overlap is 21 rather than 22 because FLEURS has
no Sanskrit. Selection reads metadata columns only — 0.3 MB against 309.6 MB per shard — so
both sets build in about 70 seconds; decoding every selected utterance is available through
`--verify`. See [REPRODUCE.md](REPRODUCE.md#5b-frozen-evaluation-sets).

## Limits

Utterances outside 0.5–60 seconds are excluded rather than truncated, and the exclusion is
counted per language in the corpus manifest. Audio is served as stored, at FLEURS' 16 kHz;
no resampling, denoising, or VAD is applied. The trained result covers one language: FLEURS
is read speech, mostly Wikipedia sentences, so the Kannada numbers say little about
conversational or noisy speech. The other 101 languages have smoke runs, not trained results.
Source audio and transcripts remain **CC BY 4.0** with Google attribution; see
[data/README.md](data/README.md).

## Citation

```bibtex
@misc{fineenvs,
  author = {Kolavi, Adithya S},
  title  = {FineEnvs: Open Source RL Environments for LLM Agents},
  year   = {2026},
  url    = {https://github.com/adithya-s-k/FineEnvs}
}

@inproceedings{conneau2023fleurs,
  title     = {FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author    = {Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and
               Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  booktitle = {IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2023}
}
```
