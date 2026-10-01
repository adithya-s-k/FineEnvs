---
title: SmolDataEnvs Multi-harness | Harbor
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
hf_oauth: true
hf_oauth_scopes:
  - inference-api
---

# SmolDataEnvs: Harbor multi-harness

**[Read the multi-harness RL article](https://huggingface.co/spaces/FineEnvs/multi-harness-rl)** · [Collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212) · [Training tutorial](https://github.com/adithya-s-k/FineEnvs/tree/main/05-multi-harness-rl)

Harbor runs OpenCode, Claude Code, Codex or Mini-SWE-Agent. Its OpenEnv UI shows rollouts, live traces and downloadable training captures.

Use the task browser to choose a problem, connect a model and watch an agent solve it. This environment also serves the four-harness evaluation used to compare both blackbox policies.

This folder contains the complete Space source. You can run it locally or deploy it on its own.

[Open the deployed UI](https://fineenvs-smoldataenv-multi-harness-harbor.hf.space/web/) · [Space](https://huggingface.co/spaces/FineEnvs/smoldataenv-multi-harness-harbor)

## Read the code

- [`smoldataenv_harbor/environment.py`](smoldataenv_harbor/environment.py): task assignment and tool counts.
- [`smoldataenv_harbor/server.py`](smoldataenv_harbor/server.py): OpenEnv server and UI.
- [`prepare.py`](prepare.py) and [`data/`](data/): the pinned 1,000 train / 250 test tasks, with grader and split checks.
- [`Dockerfile`](Dockerfile): exactly what the Space builds.

## Run locally

Use Python 3.12. From this directory:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
bash install.sh
hf auth login
export HF_TOKEN="$(hf auth token)"
export DAYTONA_API_KEY="your-key"
bash start.sh
```

Open **http://localhost:7860/**. It opens the UI at `/web/`; the API is at `/docs` and health at `/health`. `install.sh` installs OpenEnv from main and its environment examples from the same checkout. This server needs no GPU or TRL installation.

The UI supports Hugging Face sign-in and provider selection. You can also supply an OpenAI-compatible endpoint for a rollout. Set `OPENENV_LLM_URL` and `OPENENV_MODEL` only if you want a default endpoint. Training requires a token-capturing endpoint; hosted providers may support evaluation only.

## Deploy this folder

```bash
python deploy.py --repo YOUR_ORG/smoldataenv-multi-harness-harbor --public
```

Add `HF_TOKEN` and `DAYTONA_API_KEY` as Space secrets. The uploader sends the files in this directory, without generating or swapping application code. It enables the UI and preserves an existing concurrency limit. For a new Space the limit defaults to 40; use `--concurrency` to change it. CPU Basic serves the API; actual tasks run in Daytona.

You can test the same image locally:

```bash
docker build -t smoldataenv-harbor .
docker run --rm -p 7860:7860 -e HF_TOKEN -e DAYTONA_API_KEY smoldataenv-harbor
```

Runtime downloads, local credentials and old deployment archives are excluded from uploads and Docker builds. Task questions and held-out answers are never bundled into the image; the pinned tasks are prepared at startup.
