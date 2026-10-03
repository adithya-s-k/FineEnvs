---
title: FLEURS Multilingual ASR
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
license: cc-by-4.0
short_description: OpenEnv speech recognition over 102 FLEURS languages
tags:
  - openenv
  - reinforcement-learning
  - multilingual
  - automatic-speech-recognition
datasets:
  - google/fleurs
---

# Multilingual ASR environment

One OpenEnv environment serves [FLEURS](https://huggingface.co/datasets/google/fleurs)
speech for **normalized transcription**, **verbatim transcription**, and **spoken language
identification**, scored per word or per character according to the script.

**Trained against it:** [gemma-4-E4B-it-kannada-asr-grpo](https://huggingface.co/FineEnvs/gemma-4-E4B-it-kannada-asr-grpo),
GRPO on every Kannada training clip, cuts held-out character error from 0.105 to 0.057 on the 838
Kannada test clips. Part of the [Multilingual Multimodal Envs](https://huggingface.co/collections/FineEnvs/multilingual-multimodal-envs-6ac0c27c137f93e0799603e4) collection;
source in [`07-multilingual-asr/`](https://github.com/adithya-s-k/FineEnvs/tree/codex/multilingual-asr/07-multilingual-asr).

| Task | Observation | Answer | Reward |
|---|---|---|---|
| `transcription` | 16 kHz utterance | Normalized transcript | `0.8 × max(0, 1−ER) + 0.2 × exact_match` |
| `verbatim_transcription` | Same audio | Transcript with case and punctuation | Same, on unnormalized text |
| `language_id` | Same audio | FLEURS language code | Exact match |

`ER` is **word error rate**, except for scripts that do not delimit words with spaces —
Mandarin, Cantonese, Japanese, Thai, Lao, Burmese, and Khmer — where it is **character
error rate**. Word error rate on those is degenerate: the reference is effectively one
token, so any error at all scores 1.0. Every observation carries the `error_unit` it will
be graded in, and the metric is keyed `wer` or `cer`. Spaced scripts report both. A server
started with `ASR_REWARD_UNIT=cer` rewards characters for every language instead
(`fleurs-asr-cer-v1`); this is the policy the Kannada model was trained with.

Normalization differs by family because the targets differ. `transcription` grades
FLEURS' already-normalized field, so case and punctuation are stripped from both the
prediction and the reference; `verbatim_transcription` grades `raw_transcription`, where
both are the task. The same normalization is applied to both sides, so it can only remove
a difference both sides agree is not an error.

## API

- `GET /healthz` — snapshot id
- `GET /manifest` — snapshot plus the grading policy
- `GET /assets/<sha256>` — the utterance audio
- OpenEnv task discovery and `reset`/`step` over HTTP and WebSocket

Discovery never returns a reference and never reads an audio file.

Source audio and transcripts remain **CC BY 4.0** with Google attribution. The environment
code is Apache-2.0.
