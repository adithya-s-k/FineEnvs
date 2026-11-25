# Jupyter Agent — OpenEnv

A multi-turn Jupyter notebook agent environment, packaged using **[OpenEnv](https://github.com/openenv/OpenEnv)** (Hugging Face's HTTP-server / MCP-protocol framework for RL environments). The environment is a real Jupyter kernel running inside an [E2B](https://e2b.dev) cloud sandbox, exposed over MCP.

**Deployed:** [`AdithyaSK/jupyter-agent-openenv`](https://huggingface.co/spaces/AdithyaSK/jupyter-agent-openenv)

## What this environment exposes

The OpenEnv server registers five MCP tools (auto-discoverable via `list_tools()`):

| Tool | Description |
|---|---|
| `add_and_execute_code_cell(code: str)` | Run Python in the persistent notebook. Variables, imports, side-effects persist between calls. |
| `edit_and_execute_current_cell(code: str)` | Replace the last cell with new code and re-run. Use to fix errors without polluting history. |
| `execute_shell_command(command: str)` | Run a shell command inside the sandbox (`pip install`, `ls`, `curl`, …). |
| `get_notebook_state(include_images: bool = False)` | Compact summary of the cell history for agent memory. |
| `final_answer(answer: str)` | Submit the final answer for the task. Call this when you're done. The server records it (`_submitted_answer`) so a downstream reward function can score it. |

Each session gets its own E2B sandbox (`SUPPORTS_CONCURRENT_SESSIONS = True`).

## How to consume it

You can either (a) talk to the deployed HF Space (zero setup, ~1 min cold start) or (b) **run the env server locally with `uv run python -m server.app`** (fast, full control, no shared-Space concurrency limits).

> The HF Space is mainly for convenience. For real work, **run the env locally** — that's the canonical path.

### Option A — Direct URL to the deployed HF Space (zero setup)

The HF Space at [`AdithyaSK/jupyter-agent-openenv`](https://huggingface.co/spaces/AdithyaSK/jupyter-agent-openenv) is a running OpenEnv server. Connect with the generic [`openenv`](https://pypi.org/project/openenv/) MCP client. No env-specific package install required:

```python
from openenv.core.mcp_client import MCPToolClient

with MCPToolClient(base_url="https://AdithyaSK-jupyter-agent-openenv.hf.space").sync() as env:
    env.reset()
    tools = env.list_tools()             # [Tool(name='add_and_execute_code_cell', ...), ...]
    out = env.call_tool("add_and_execute_code_cell", code="print(2 ** 10)")
    print(out)                           # '1024'
```

### Option B — Install the typed client from the HF Space

Every OpenEnv Space is also a pip-installable git repo. This gives you typed clients and any helper classes the env author shipped:

```bash
uv pip install "git+https://huggingface.co/spaces/AdithyaSK/jupyter-agent-openenv"
```

Then:

```python
from jupyter_agent_env.client import JupyterAgentEnv
with JupyterAgentEnv(base_url="https://AdithyaSK-jupyter-agent-openenv.hf.space").sync() as env:
    ...
```

### Option C — Run the env server locally with Python (verified working)

This is the recommended path for development. The HF Space is just convenience.

```bash
cd 00-environments-101/envs/jupyter/openenv
uv sync                                      # installs openenv, fastmcp, e2b
export E2B_API_KEY=e2b_...                   # or rely on the repo-root .env
uv run python -m server.app                  # serves on http://0.0.0.0:8000
# in another shell:
curl http://localhost:8000/health            # -> {"status":"healthy"}
```

Then point the rollout at it:

```bash
OPENENV_URL=http://localhost:8000 uv run python rollout.py
```

(Or hit the deployed Space by leaving `OPENENV_URL` unset.)

### Option D — Docker pull from the HF registry

```bash
docker pull registry.hf.space/adithyask-jupyter-agent-openenv:latest
docker run -it -p 8000:8000 -e E2B_API_KEY=$E2B_API_KEY \
    registry.hf.space/adithyask-jupyter-agent-openenv:latest
```

> Reference: [OpenEnv deployment tutorial](https://github.com/openenv/OpenEnv/blob/main/tutorial/02-deployment.md).

## Run the rollout

The `rollout.py` in this folder shows end-to-end consumption: connect to the deployed Space, auto-discover its MCP tools, and drive a multi-turn conversation with **Qwen3-Coder-480B** via Hugging Face Inference Providers using the standard `openai` client.

### Setup

A `.env` at the **repo root** (`RL_Envs_101/.env`) is required:

```bash
HF_TOKEN=hf_...        # for HF Inference Providers (Qwen)
# OPENAI_API_KEY=sk-...   # only if you swap the model in rollout.py to OpenAI
```

### Run

```bash
cd 00-environments-101/envs/jupyter/openenv
uv sync
uv run python rollout.py
```

### What it does

1. `MCPToolClient(base_url="https://AdithyaSK-jupyter-agent-openenv.hf.space").sync()` — connects to the deployed env over MCP.
2. `env.list_tools()` — auto-discovers the 5 MCP tools the server exposes (4 notebook tools + `final_answer`).
3. Converts the MCP tool schemas to OpenAI tool-call schemas.
4. Loops up to `MAX_TURNS=6` calling `chat.completions.create` against `Qwen/Qwen3-Coder-480B-A35B-Instruct:together` through `https://router.huggingface.co/v1`.
5. For each tool-call the model emits, it dispatches via `env.call_tool(name, **args)` and feeds the result back as a `tool` message.
6. Stops when the model produces a turn with no tool calls, then prints the full trajectory.

### Sample output

```
================================================================================
OpenEnv server: https://AdithyaSK-jupyter-agent-openenv.hf.space
Model:          Qwen/Qwen3-Coder-480B-A35B-Instruct:together
Task:           Compute the mean and standard deviation of the list ...
================================================================================

Discovered 5 tools: ['add_and_execute_code_cell', 'edit_and_execute_current_cell',
                     'execute_shell_command', 'get_notebook_state', 'final_answer']

──── turn 1 ────────────────────────────────────────
[tool-call] add_and_execute_code_cell({'code': 'import numpy as np\n...'})
[tool-result]
Mean: 17.9
Standard Deviation: 8.080222769206305

──── turn 2 ────────────────────────────────────────
[assistant] The mean of the list is 17.9, and the standard deviation is approximately 8.08.

[done] no tool calls, stopping.
```

### Configuration knobs (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `OPENENV_URL` | `https://AdithyaSK-jupyter-agent-openenv.hf.space` | Where to find the OpenEnv server (set to `http://localhost:8000` for local Docker). |
| `ROLLOUT_MODEL` | `Qwen/Qwen3-Coder-480B-A35B-Instruct:together` | Any HF Router model id, or swap the client for OpenAI native by editing `rollout.py`. |
| `MAX_TURNS` | `6` | Hard cap on tool-call turns. |

## What's in this folder

| Path | What it is |
|---|---|
| `rollout.py` | Self-contained consumer demo (this is the file you'd write yourself). |
| `server/` | The OpenEnv server source that's deployed to the HF Space. The MCP env definition (`@mcp.tool` registrations) lives in `server/jupyter_environment.py`. |
| `models.py` | Pydantic models for the env's internal state, used by the server. |
| `Dockerfile` | Used by `openenv push` / HF Spaces to build the deployable image. |
| `openenv.yaml` | OpenEnv manifest (env name, version). |
| `pyproject.toml` | Lists `openenv`, `fastmcp`, `e2b-code-interpreter`, plus rollout-side `openai`, `python-dotenv`. |
| `tests/` | Server-side tests (require a running server). |

## Redeploying the server (optional)

If you've cloned this repo and want to push the env to your own HF Space:

```bash
cd 00-environments-101/envs/jupyter/openenv
openenv push --repo-id <your-username>/jupyter-agent-openenv
```

Requires `E2B_API_KEY` set as a Space secret.

## References

- [OpenEnv getting started](https://github.com/openenv/OpenEnv/blob/main/tutorial/01-environments.md)
- [OpenEnv deployment guide](https://github.com/openenv/OpenEnv/blob/main/tutorial/02-deployment.md)
- [HF Inference Providers — Chat Completion API](https://huggingface.co/docs/inference-providers/en/tasks/chat-completion)
- [E2B Code Interpreter](https://e2b.dev)
