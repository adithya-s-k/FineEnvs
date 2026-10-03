# What the Kannada runs taught us

Every held-out number here comes from the 838 clips of FLEURS `kn_in` test. The untuned and the
trained model are scored on the same vLLM engine and graded by the same environment, and changes are
paired clip by clip. The per-clip predictions behind every table are in
[FineEnvs/multilingual-multimodal-rl-runs](https://huggingface.co/datasets/FineEnvs/multilingual-multimodal-rl-runs).

## The headline

One epoch of GRPO on all 2,282 Kannada training clips cuts character error by 45%: 0.1047 to 0.0571,
paired change −0.048 with a 95% CI of (−0.057, −0.039). That took 4.7 hours on one A100. Most of it
arrives in the first 150 steps, and the curve is flat from about step 250, so a run a third as long
would have found nearly all of it.

It took four attempts to get there. The first three are the more useful part of this page.

## 0. Choosing the model

Before training anything we scored four candidates on the 21 languages this project shares with the
OCR one. On the eleven non-Indic languages their average rewards were within 0.023 of each other. On
the Indic languages they were not:

| model | reward on Kannada | average over the 10 Indic languages |
|---|---:|---:|
| **Gemma 4 E4B** | **0.721** | **0.579** |
| Gemma 4 E2B | 0.336 | 0.352 |
| Qwen2.5-Omni-7B | 0.060 | 0.183 |
| Qwen2.5-Omni-3B | 0.027 | 0.129 |

A model that scores 0.03 on Kannada has nothing for GRPO to sharpen, so Gemma 4 E4B was the only real
choice. The full comparison is in [`results/eval-21-validation/`](./results/eval-21-validation/).

## 1. The trainer never let the model hear the clip

Earlier runs raised training reward and left held-out reward almost where it started. The obvious
suspects were the learning rate and the amount of data, and neither was it.

We first checked whether there was anything to learn at all. Sampling sixteen transcripts per clip
from the base model, the best of the sixteen beat the greedy answer by 0.09 reward on average, and
on 69% of clips at least one sample beat greedy. So the policy already produced better transcripts.
GRPO only has to make them more likely, and it was not doing that.

The reason was in TRL. GRPO takes its loss from a second forward pass over the prompt and the
sampled transcript. TRL 1.13 rebuilds image inputs for that pass and drops audio inputs. The model
generated while listening and was graded without the audio, so the gradient pushed up
p(transcript | no audio). That teaches a language model the training sentences. It does not teach
it to listen.

The size of the gap is measurable. On the first batch of the final run, the model's own transcripts
average **−0.38 nats per token with the clip and −6.22 without it**. Every earlier run took its
gradient against the second number.

There was an earlier symptom we misread. An adapter on the audio encoder stayed exactly zero after
250 steps: all 24 of its `lora_B` matrices. We concluded the audio tower could not be trained
through TRL and stopped adapting it. That was true, but it was the visible half of the bug. The
language model never saw audio in the loss either.

`AudioGRPOTrainer` stores each batch's audio features when the prompts are tokenized. It carries
them through TRL's shuffle and split, and attaches the right clips to every log-prob pass. On the
first batch it measures the gap above and refuses to train if the clip adds less than half a nat
per token.

The general lesson: if a model takes more than one modality, check that each one reaches the loss.
Generation can use an input that training silently ignores, and then every metric looks healthy
except the held-out one.

## 2. Word error is the wrong reward for Kannada

The environment's default reward is word error for languages that put spaces between words. A
Kannada word is a long, inflected string, so fixing three wrong letters in it changes nothing until
the last one is right. One earlier run cut character error by 14% while word error moved 2.3%: the
model was improving, and the reward barely registered it.

The `cer` policy (`ASR_REWARD_UNIT=cer`) rewards characters instead, spaces included, so a model
cannot gain by deleting word boundaries. Word error is still reported next to it. The deployed Space
keeps the word-error default, which is the right choice for most of its 102 languages.

## 3. Padding was heard as speech, but it mattered less than we thought

To stack clips of different lengths into one batch, every waveform was padded to 30 seconds. The
feature extractor cannot tell padded zeros from silence, so it marked all 30 seconds as audio, and
a typical clip arrived with about 18 seconds of nothing appended. Evaluation through vLLM never pads.

Padding now happens on the extracted features, with the padding masked, so only real frames become
audio tokens. We expected this to explain a gap we had seen: training reward started at 0.43 where
vLLM scored 0.50 on the same clips. It explained little of it. Greedy transcripts from the trainer
now match vLLM's on 6 of 8 clips, against 27 of 48 before, but word error was already the same
either way. Most of that gap was sampling at temperature 0.9 against greedy decoding.

## 4. Mean error is not a safe headline

Error rates have no upper bound. A transcript that loops, say "ಮೀಮೀಮೀ…" for a hundred characters,
can score a CER of 5 on one clip. At step 200 the mean CER rose from 0.087 to 0.095. On that
checkpoint 167 clips got better, 130 got worse, and the median improved. The whole rise came from
two clips that looped.

Every headline here caps the error at 1 per clip before averaging, and the uncapped means are
published beside it. By the capped measure the final checkpoint is the best of all 23. By the
uncapped mean, step 300 edges it by 0.0005, which is well inside the noise.

The trained model loops much less than the base did: 10 clips before, 1 after.

## 5. It needs far less than the 12 hours

The run was sized from a smoke test at 59 seconds a step. It averaged 27 to 30. As the model
stopped looping, its transcripts shrank from about 80 tokens to 48, and generation is most of
a step. One epoch took 4.7 hours.

Since the curve is flat from step 250, more epochs of the same clips are unlikely to help. The
remaining time is better spent on more languages or a larger model.
[`gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) also accepts audio.

## 6. Smaller things that broke

- **The first full-data run died at startup.** The server answers at most 1,000 task positions per
  request, and the trainer asked for all 2,282 at once. It now fetches them in pieces.
- **Checkpoints are scored while they upload.** The evaluation job reads checkpoints from the
  bucket the training job is writing to. Each save writes `ready.json` with the adapter's size and
  hash, and the evaluator scores an adapter only once every byte matches.
- **GPU types run out.** The OCR run's evaluator waited 1.5 hours for an L40S that never came. It
  resumes from its own `curve.json`, so relaunching it on an A100 lost nothing.

## 7. From building the environment

These came from the first smoke runs, before any training result.

- **FLEURS' `id` is a sentence, not a recording.** Several speakers read the same sentence, so
  `hi_in` test holds 418 recordings under 265 ids. Keying tasks on it merged 37% of the corpus and
  gave two different clips one task id. Tasks are now keyed on the recording itself.
- **FLEURS ships 32-bit float WAV.** Python's standard library rejects that format. Every test
  passed because the fixtures were 16-bit PCM, and the first GPU run died on real audio. The
  fixtures now write float WAV like the corpus.
- **Audio is fetched by task, not by hash.** The indexed corpus has to know which task a clip
  belongs to before it can find it in the bucket. Every fetch failed until the task id travelled
  with the request.
- **Four samples per clip are not enough on easy clips.** On a short English clip all four
  transcripts scored the same, the advantage was zero, and nothing could be learned. The trainer
  now fails loudly when that happens to every group.
- **Language ID wants the FLEURS locale, not the language.** Both models answered `ar_sa` for
  `ar_eg` and `es_es` for `es_419`. They recognised the language and missed the locale. That is the
  task being hard, not the reward misfiring.
