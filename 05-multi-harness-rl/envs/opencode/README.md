---
title: SmolDataEnvs Multi-harness | Native OpenCode
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# SmolDataEnvs: Native OpenCode

**[Read the multi-harness RL article](https://huggingface.co/spaces/FineEnvs/multi-harness-rl)** · [Collection](https://huggingface.co/collections/FineEnvs/smoldataenvs-multi-harness-rl-6abdfaaa8d74dacd481d5212) · [Training tutorial](https://github.com/adithya-s-k/FineEnvs/tree/main/05-multi-harness-rl)

Native OpenCode runs inside the sandbox. The server grades its answer and returns a typed token trace for AsyncGRPO.

`opencode_env` is deprecated since OpenEnv 0.7.0. This environment remains available for the historical comparison; new runs should use the [Harbor Space](https://huggingface.co/spaces/FineEnvs/smoldataenv-multi-harness-harbor) with `harness="opencode"`. Rebuilding this standalone Space from main will stop working once upstream removes `opencode_env`.

This folder contains the complete Space source. You can run it locally or deploy it on its own.

[Open the deployed UI](https://fineenvs-smoldataenv-multi-harness-opencode.hf.space/web/) · [Space](https://huggingface.co/spaces/FineEnvs/smoldataenv-multi-harness-opencode)

## Read the code

- [`smoldataenv_opencode/environment.py`](smoldataenv_opencode/environment.py): task execution and grading.
- [`smoldataenv_opencode/server.py`](smoldataenv_opencode/server.py): OpenEnv server and UI.
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

For agent rollouts, set `SANDBOX_VLLM_URL` and the secret `SANDBOX_VLLM_KEY` to an inference endpoint reachable by the sandbox. Training requires engine token IDs and logprobs.

## Deploy this folder

```bash
python deploy.py --repo YOUR_ORG/smoldataenv-multi-harness-opencode --public
```

Add `HF_TOKEN` and `DAYTONA_API_KEY` as Space secrets. The uploader sends the files in this directory, without generating or swapping application code. It enables the UI and preserves an existing concurrency limit. For a new Space the limit defaults to 40; use `--concurrency` to change it. CPU Basic serves the API; actual tasks run in Daytona.

You can test the same image locally:

```bash
docker build -t smoldataenv-opencode .
docker run --rm -p 7860:7860 -e HF_TOKEN -e DAYTONA_API_KEY smoldataenv-opencode
```

Runtime downloads, local credentials and old deployment archives are excluded from uploads and Docker builds. Task questions and held-out answers are never bundled into the image; the pinned tasks are prepared at startup.
