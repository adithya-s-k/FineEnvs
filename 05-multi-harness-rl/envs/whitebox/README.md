---
title: SmolDataEnvs Multi-harness | SETA Whitebox
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# SmolDataEnvs: SETA whitebox

**[Read the multi-harness RL article](https://huggingface.co/spaces/FineEnvs/multi-harness-rl)** · [Collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212) · [Training tutorial](https://github.com/adithya-s-k/FineEnvs/tree/main/05-multi-harness-rl)

TRL drives the bash, file and submit-solution tools. Each WebSocket session owns one sandbox.

This is the hands-on companion to the article: explore a dataset, run a command and submit an answer. Training uses these same tools with TRL driving the model.

This folder contains the complete Space source. You can run it locally or deploy it on its own.

[Open the deployed UI](https://fineenvs-smoldataenv-multi-harness-whitebox.hf.space/web/) · [Space](https://huggingface.co/spaces/FineEnvs/smoldataenv-multi-harness-whitebox)

## Read the code

- [`smoldataenv_whitebox/environment.py`](smoldataenv_whitebox/environment.py): task execution and grading.
- [`smoldataenv_whitebox/server.py`](smoldataenv_whitebox/server.py): OpenEnv API.
- [`smoldataenv_whitebox/ui.py`](smoldataenv_whitebox/ui.py) and [`playground.py`](smoldataenv_whitebox/playground.py): browser layout and per-visitor sandbox lifecycle.
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

The playground has three steps:

1. Choose a split, difficulty and task, then click **Start this task**.
2. Run Bash or Python commands against `/home/user/input`. The file-list and CSV-preview buttons prepare example commands for you.
3. Enter your answer and click **Submit and score**. Correctness, tool calls and total reward appear above the workspace.

Each browser session gets its own sandbox. **Release sandbox**, grading, or session expiry closes it. Execution history keeps your commands and outputs. The UI calls the same `BashEnvironment` used by training; no model endpoint is needed to try it yourself.

## Deploy this folder

```bash
python deploy.py --repo YOUR_ORG/smoldataenv-multi-harness-whitebox --public
```

Add `HF_TOKEN` and `DAYTONA_API_KEY` as Space secrets. The uploader sends the files in this directory, without generating or swapping application code. It enables the UI and preserves an existing concurrency limit. For a new Space the limit defaults to 40; use `--concurrency` to change it. CPU Basic serves the API; actual tasks run in Daytona.

You can test the same image locally:

```bash
docker build -t smoldataenv-whitebox .
docker run --rm -p 7860:7860 -e HF_TOKEN -e DAYTONA_API_KEY smoldataenv-whitebox
```

Runtime downloads, local credentials and old deployment archives are excluded from uploads and Docker builds. Task questions and held-out answers are never bundled into the image; the pinned tasks are prepared at startup.
