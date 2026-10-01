# Run the environment

This folder contains the environment code used by training and evaluation. Each episode gets its own Daytona sandbox. The environment stages the task's data, runs tools or an agent, grades the answer and releases the sandbox. The grader runs outside the agent's workspace.

| Environment | Implementation | Server | Training client |
|---|---|---|---|
| SETA whitebox | [Tools and reward](whitebox/environment.py) | [OpenEnv server](whitebox/server.py) | [Python tool client](whitebox/client.py) |
| Native OpenCode | [Task setup and sessions](opencode/environment.py) | [OpenEnv server](opencode/server.py) | [TrainingTrace client](opencode/client.py) |
| Harbor multi-harness | [Task assignment and action counts](harbor/environment.py) | [OpenEnv Harbor service](harbor/server.py) | Public `HarborSession` |

The whitebox model calls `bash`, `read`, `write`, `edit`, `ls`, `grep`, `glob` and `submit_solution`. Native OpenCode owns its tool loop. Harbor selects one of four agent programs. Both blackbox paths return OpenEnv's typed token IDs, logprobs and masks. Shared [task code](tasks.py) stages the input files and grades answers; [Daytona code](daytona.py) implements sandbox creation, file access, commands and cleanup.

The native OpenCode SDK is deprecated upstream. It is kept here to reproduce the separate native interface requested by this comparison. Harbor's OpenCode harness is the maintained alternative, but changing to it would change this experiment.

## Local server

Run from `05-multi-harness-rl`, after installing dependencies and running `prepare.py`:

```bash
python -m uvicorn envs.whitebox.server:app --host 0.0.0.0 --port 7860
# Or: envs.opencode.server:app / envs.harbor.server:app
```

Open `/web` for OpenEnv's UI or `/docs` for the API. The Task API exposes the fixed train/test tasks. Use persistent WebSocket clients for whitebox episodes; closing the connection releases its sandbox.

These servers use the UI shipped with OpenEnv main. Harbor keeps its native rollout and trace views, with SmolDataEnvs task configuration supplied by this folder.

`jobs/run.py` starts the selected server inside the training or evaluation allocation. The same source can instead run on a CPU Space. A Space does not host the model or replace the Daytona sandbox.

## Docker or a Hub Space

Build from the tutorial folder, which is the Docker build context:

```bash
docker build -f envs/Dockerfile -t smoldataenv .
docker run --rm -p 7860:7860 \
  -e HF_TOKEN -e DAYTONA_API_KEY -e ENV_MODE=whitebox \
  smoldataenv
```

Use `ENV_MODE=opencode` or `ENV_MODE=harbor` for the other interfaces. The container downloads the same checked task revisions when it starts. It serves port 7860, matching [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).

To upload this code to your own Space:

```bash
python -m envs.deploy --mode whitebox --repo YOUR_ORG/smoldataenv-whitebox --public
python -m envs.deploy --mode opencode --repo YOUR_ORG/smoldataenv-opencode --public
python -m envs.deploy --mode harbor --repo YOUR_ORG/smoldataenv-harbor --public
```

Add `HF_TOKEN` and `DAYTONA_API_KEY` as **Space secrets**. The uploader never copies your local credentials. Use CPU Basic initially; `MAX_CONCURRENT_ENVS` defaults to 40. Concurrency also depends on your sandbox quota and inference capacity. Private Spaces require authenticated access; the example clients below target public or local servers.

The Harbor Space card enables Hugging Face sign-in with the `inference-api` scope. OpenEnv's current UI handles model-provider selection and visitor credentials. This does not replace the token-capturing vLLM endpoint used for training.

For native OpenCode, also set `SANDBOX_VLLM_URL` and `SANDBOX_VLLM_KEY` to the sandbox-accessible inference endpoint. For Harbor, set `OPENENV_LLM_URL` and `OPENENV_MODEL`, or provide them for each rollout. The model endpoint must be reachable from the Space, not just the training machine. Training endpoints must supply token IDs and logprobs; ordinary hosted inference may only support evaluation.

## Point the trainer at the server

Keep the trainer and weight-synchronization connection on the GPU machine. The environment server URL is a separate setting:

```bash
python -m train.whitebox --server https://YOUR_SPACE.hf.space \
  --vllm-url http://127.0.0.1:8000 --output runs/whitebox

python -m train.opencode --server https://YOUR_SPACE.hf.space \
  --vllm-url http://127.0.0.1:8000 --output runs/opencode
```

Harbor additionally needs an inference URL reachable by the environment server. Its task paths and tool counts are read from the server, so the trainer does not need access to the server's filesystem. The default job launcher colocates the server with the trainer and handles this routing automatically.

Check [VALIDATION.md](../VALIDATION.md) for what has actually passed. A deployment recipe alone does not establish that a GPU training run or public Space rollout has passed.
