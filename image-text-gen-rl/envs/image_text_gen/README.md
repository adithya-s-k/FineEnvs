---
title: Image Text Gen RL
emoji: 🔤
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 8000
license: apache-2.0
tags:
  - openenv
  - reinforcement-learning
  - text-to-image
  - text-rendering
datasets:
  - AdithyaSK/image-text-gen-rl-prompts
  - leffff/Diffusion-Reward-Modeling-for-Text-Rendering-Dataset
---

# Image text generation RL environment

An OpenEnv environment for training **text-to-image models to render text correctly**. `reset()`
hands out a prompt that quotes the text to draw; `step()` takes one generated image (base64
PNG/JPEG/WebP) and returns a reward in [0, 1]. Open `/web` to try prompts and see every verifier's
reading.

**Blind verification.** The image is resized to a 1536 px long side and transcribed literally, in
parallel, by two vision models on Hugging Face Inference Providers (Gemma 4 31B and Qwen3.6-35B-A3B, chosen by calibration). Neither sees the prompt or the
target, so neither can bend its reading toward the answer. Broken or non-letter glyphs are marked
`�`.

**Reward.** Text accuracy is 1 − CER of the transcription span closest to the target, taken from the
best reading. Each malformed glyph, counted from the worst reading, multiplies the score by 0.8.
Unrequested text costs 0.05 per character beyond 2, up to 0.5, whatever the target's length; words
the prompt names outside its quoted target are excused. Wrong letter case costs 10%. Invalid or blank images score 0 without a provider call. Provider failures assign
**no** reward and leave the episode open for a retry.

`/manifest` reports the pinned dataset revision, split counts, scoring policy and verifier models;
`/image_text_gen/splits|num_tasks|task|task_range` is the OpenEnv Task API.

Tasks: 13,434 prompts from [AdithyaSK/image-text-gen-rl-prompts](https://huggingface.co/datasets/AdithyaSK/image-text-gen-rl-prompts)
(mounted read-only at `/dataset`), a screened derivative of
[leffff/Diffusion-Reward-Modeling-for-Text-Rendering-Dataset](https://huggingface.co/datasets/leffff/Diffusion-Reward-Modeling-for-Text-Rendering-Dataset)
by leffff (MIT); 300 sexual, vulgar, hateful, violent, self-harm and drug prompts were removed.
Grading records are written to the private bucket `AdithyaSK/image-text-gen-rl`.
Environment source is Apache-2.0; see `LICENSE` and `NOTICE`.
