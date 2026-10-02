---
title: Wordle NeMo Gym
emoji: 🟨
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 11000
---

# Wordle NeMo Gym Resources Server

Wordle exposed as an [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) Resources Server.

**Endpoints:** `POST /seed_session`, `POST /guess`, `POST /get_history`, `POST /verify`, `GET /health`

`/verify` pays the shared Wordle reward from the game's own tool outputs: `1 + 0.5 * (1 - guesses/6)` for a win, `0.1 ×` the best green count for a loss.

Source: [FineEnvs `00-environments-101/envs/wordle`](https://github.com/adithya-s-k/FineEnvs/tree/main/00-environments-101/envs/wordle). Part of the [RL Envs 101](https://huggingface.co/collections/FineEnvs/rl-envs-101-6abca516b22e765e0b1aa0d1) collection.
