---
title: "How to turn a game into an RL environment"
short_description: "From an idea to a trained 4B, with the dead ends left in"
emoji: 🌍
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8080
header: mini
pinned: false
license: mit
thumbnail: >-
  https://fineenvs-geoguesser-article.hf.space/thumb.auto.jpg
tags:
  - research-article-template
  - rl-environments
  - openenv
  - trl
  - grpo
  - reinforcement-learning
  - vision-language-model
  - visual-geolocation
  - reward-design
  - qwen3.5
---

# How to turn a game into an RL environment

The technical intuition, worked end to end on one example: curating a dataset that makes the task
learnable, designing the environment, shipping it with [OpenEnv](https://github.com/huggingface/OpenEnv)
so other people can run it, and training a Qwen3.5-4B against it with
[TRL](https://github.com/huggingface/trl) until it outscored `gpt-5.4-mini` and `claude-haiku-4.5`
at GeoGuessr.

Every figure is built from the published records, and everything it cites is on the Hub:

- **[`geoguesser-env`](https://huggingface.co/spaces/FineEnvs/geoguesser-env)** is the environment, playable in a browser
- **[`geoguesser-tasks`](https://huggingface.co/datasets/FineEnvs/geoguesser-tasks)** holds both task splits
- **[`geoguesser-panos`](https://huggingface.co/buckets/FineEnvs/geoguesser-panos)** holds the 22 GB of imagery
- **[`geoguesser-trackio`](https://huggingface.co/spaces/FineEnvs/geoguesser-trackio)** has all four training runs
- **[the collection](https://huggingface.co/collections/FineEnvs/geoguesser-env-6a969f8db267fe0e85fa1ab6)** gathers the lot

Source, environment, eval harness and training script live in
[FineEnvs](https://github.com/adithya-s-k/FineEnvs) under `03-geoguesser/`, with the article
itself under `content/articles/geoguesser/`.
