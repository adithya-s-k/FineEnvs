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

# Multilingual ASR

Listen to a clip, write down what was said, and see how it scores. This Space serves all of
[FLEURS](https://huggingface.co/datasets/google/fleurs), 102 languages and 1,151,940 tasks, as an
[OpenEnv](https://github.com/huggingface/OpenEnv) environment you can play in the browser or train
a model against.

Open [the playground](https://fineenvs-fleurs-asr-env.hf.space/web/), pick a language, and press
play. The reference is revealed after you score.

**A model trained here:**
[gemma-4-E4B-it-kannada-asr-grpo](https://huggingface.co/FineEnvs/gemma-4-E4B-it-kannada-asr-grpo).
GRPO on every Kannada clip cut its character error on the 838 held-out Kannada clips from 0.105 to
0.057. It is part of the [Multilingual Multimodal Envs](https://huggingface.co/collections/FineEnvs/multilingual-multimodal-envs-6ac0c27c137f93e0799603e4) collection, and the source is in
[`06-multilingual/asr/`](https://github.com/adithya-s-k/FineEnvs/tree/main/06-multilingual/asr).

## Tasks

| Task | You hear | You answer | Reward |
|---|---|---|---|
| `transcription` | a 16 kHz clip | what was said, lowercase, no punctuation | `0.8 × (1 − error rate) + 0.2 × exact match` |
| `verbatim_transcription` | the same clip | what was said, with case and punctuation | the same, on the raw text |
| `language_id` | the same clip | the FLEURS language code | exact match |

The error rate counts words for languages that put spaces between words. It counts characters for
the seven that do not: Mandarin, Cantonese, Japanese, Thai, Lao, Burmese and Khmer. A word error
rate on those would score any mistake as a total miss. Every result says which unit it was
measured in, and spaced scripts report both.

A server started with `ASR_REWARD_UNIT=cer` rewards characters for every language. That is how
the Kannada model was trained, because a single wrong vowel sign makes a long Kannada word wrong.

For `transcription`, case and punctuation are stripped from both your answer and the reference,
because FLEURS' normalised reference has none. For `verbatim_transcription` they are the point, so
only spacing and Unicode form are normalised.

## API

- `GET /healthz`: the snapshot id
- `GET /manifest`: the snapshot, the splits, and the grading policy
- `GET /assets/<sha256>`: a clip's audio
- OpenEnv task discovery, and `reset` / `step` over HTTP and WebSocket

Task discovery never returns a reference and never reads audio.

The audio and transcripts are FLEURS', **CC BY 4.0**, with credit to Google. The environment code is
Apache-2.0.

```bibtex
@misc{fineenvs,
  author = {Kolavi, Adithya S},
  title  = {FineEnvs: Open Source RL Environments for LLM Agents},
  year   = {2026},
  url    = {https://github.com/adithya-s-k/FineEnvs}
}
```
