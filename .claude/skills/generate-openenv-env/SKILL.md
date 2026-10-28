---
name: generate-openenv-env
description: Builds an OpenEnv (Hugging Face) variant of an RL environment. Use whenever someone asks to scaffold an OpenEnv server, port an existing env to OpenEnv, add MCP tools to an env, or deploy an OpenEnv to HF Spaces. OpenEnv is the right framework when the user wants HTTP+MCP, structured tool calls discovered via `list_tools()`, an optional Gradio UI, sandbox-backed sessions, or deployment as a Docker container / HF Space. Output is a runnable `<env_dir>/openenv/` folder with `server/app.py`, `server/<env>_environment.py`, `pyproject.toml`, `Dockerfile`, and `rollout.py`. Use for prompts like "wrap my game in OpenEnv", "make an MCP env for X", or "add the openenv variant".
---

# generate-openenv-env

Build the OpenEnv variant of an env. Targets OpenEnv >= 0.4.1 (the `openenv` package — formerly `openenv-core[core]`).

## Concept

OpenEnv is an HTTP server exposing tools via the **MCP** (Model Context Protocol) shape. The runtime is FastAPI; tools are FastMCP-decorated functions. Clients discover tools via `list_tools()` (under the hood: a list-tools action on `/step`) and call them via `call_tool(name, **args)`.

When the user has a shared domain module (`<domain>.py`) and wants an OpenEnv variant, **never duplicate domain logic into the framework folder** — wrap it.

## Archetypes (pick the one matching the task)

| Archetype | Hallmarks |
|---|---|
| **Pure-Python game** | Deterministic, single `@mcp.tool`, text-only observations. Reward computed externally from the trajectory. |
| **Stateful sandbox** | E2B / browser / DB, multiple tools mutating session state, MCPEnvironment per session. |
| **Vision / computer-use** | Screenshots returned as MCP image content blocks (`fastmcp.utilities.types.Image`), 19-tool action surface modelled on Anthropic's `computer_20251124`, optional custom Gradio UI mounted at `/web`. |

## Recommended file layout

The user picks the actual paths. The canonical shape:

```
<env_dir>/openenv/
├── pyproject.toml      # openenv + e2b-* + fastmcp + uvicorn + gradio
├── __init__.py
├── models.py           # Pydantic State / typed action / observation models
├── Dockerfile          # multi-stage from ghcr.io/huggingface/openenv-base
├── openenv.yaml        # spec_version 1, name, runtime, app, port
├── server/
│   ├── __init__.py
│   ├── app.py          # create_app(EnvCls, CallToolAction, CallToolObservation, env_name=...)
│   └── <env>_environment.py    # MCPEnvironment subclass with @mcp.tool methods
├── rollout.py          # MCPToolClient drives the server; auto-discovers tools
└── README.md           # one-page; with HF frontmatter if deploying to Spaces
```

## Implementation order (one continuous pass)

### 1. Pydantic state model — `models.py`

Subclass `openenv.core.env_server.types.State`. Add per-episode fields you'll mutate (`step_count`, `last_output`, sandbox/session ids, anything you want to inspect later).

For visual envs, add `last_screenshot_b64`. Don't store huge blobs unless you need them in `state` — use `metadata` on observations instead.

### 2. The MCPEnvironment — `server/<name>_environment.py`

```python
class MyEnv(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True   # only if real session isolation
    def __init__(self):
        # ... env-side state init
        mcp = FastMCP("my_env")
        @mcp.tool
        def my_tool(arg: int) -> str: ...
        super().__init__(mcp)
```

Key contracts:
- **Dual-import pattern.** Inside `server/`, write `try: from ..models import X; except ImportError: from models import X`. Relative imports work inside the repo (`PYTHONPATH=src:envs`); flat imports work in Docker (`/app/env`). Same applies to sibling modules like `e2b_sandbox.py`.
- **Tool methods** are `@mcp.tool` decorated functions inside `__init__`. They close over `self` and read/write env state. Don't try to put `@mcp.tool` on instance methods — FastMCP introspects free functions.
- **For images**, use `fastmcp.utilities.types.Image`: `return Image(data=png_bytes, format="png")`. The model receives an MCP image content block. Returning a base64 string in text means the model is blind.
- **Lifecycle hooks**: `reset(seed=None, episode_id=None, **kwargs)` returns an `Observation`; `step(action, timeout_s=None, **kwargs)` is inherited from `MCPEnvironment` for tool dispatch — only override if you need pre/post hooks (e.g. step-counter increment, terminate signal handling).

### 3. The FastAPI app — `server/app.py`

```python
import os
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
try:
    from .my_environment import MyEnv
    from .gradio_ui import my_ui_builder      # only if you have a custom UI
except ImportError:
    from server.my_environment import MyEnv
    from server.gradio_ui import my_ui_builder

def _custom_gradio_builder(*args, **kwargs):
    return my_ui_builder(env_factory=MyEnv)

os.environ["ENABLE_WEB_INTERFACE"] = "true"
app = create_app(
    MyEnv, CallToolAction, CallToolObservation,
    env_name="my_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "4")),
    gradio_builder=_custom_gradio_builder,   # omit if no custom UI
)
```

Pass the **class** to `create_app`, not an instantiated env.

### 4. Custom Gradio UI (optional, computer-use-style envs benefit)

`server/gradio_ui.py` defines `my_ui_builder(env_factory)` that returns a `gr.Blocks`. Mounted at `/web` (set `base_path: /web` in the HF Space frontmatter). For computer-use envs, the canonical pattern includes an iframe panel showing the E2B stream URL alongside text controls — but any `gr.Blocks` layout works.

### 5. The rollout — `rollout.py`

Use `openenv.core.mcp_client.MCPToolClient`. **Discover** tools, don't hardcode:

```python
from openenv.core.mcp_client import MCPToolClient
with MCPToolClient(base_url=ENV_URL).sync() as env:
    env.reset()
    tools = env.list_tools()                 # list of ToolSpec
    # convert to OpenAI tool schemas, drive the LLM, call env.call_tool(name, **args)
```

Note: for **image-returning tools**, `env.call_tool` strips to `result.data` (which is `None` for image returns). Use `env.step(CallToolAction(tool_name="screenshot", arguments={}))` to get the full result dict, then read `obs.result["content"][0]["data"]` for the b64 image. Pattern:

```python
def _call(env, name, **kwargs):
    out = env.step(CallToolAction(tool_name=name, arguments=kwargs))
    return out.observation.result or {}

def _b64_screenshot(env):
    res = _call(env, "screenshot")
    for c in res.get("content", []) or []:
        if c.get("type") == "image" and c.get("data"):
            return c["data"]
    raise RuntimeError(f"screenshot returned no image: {res}")
```

For multimodal models (Qwen3-VL, GPT-4o), feed the latest screenshot as an image block in the user message every turn.

### 6. The Dockerfile

Use a multi-stage build:
- `FROM ghcr.io/huggingface/openenv-base:latest` (the official base — already has FastAPI, MCP, Gradio).
- `uv sync` twice (no-install-project, then with project) for cache friendliness.
- Healthcheck via `/health`.
- Expose port 8000.

For HF Spaces, the canonical `app_port` is **8000** (not 7860 — OpenEnv's pattern uses 8000). Set `base_path: /web` in the README frontmatter so Gradio mounts under that prefix.

### 7. The HF Space README frontmatter

```yaml
---
title: My Env Server
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags: [openenv, your-domain]
short_description: One-line summary
---
```

## Validation gates

Before declaring done, all four must pass:

1. **In-repo import** — `PYTHONPATH=envs uv run python -c "from envs.<name>.openenv.server.<name>_environment import <Cls>"`
2. **Local server** — `uv run uvicorn server.app:app --port 8000` then `curl /health` returns `{"status":"healthy"}` and `/list_environments` returns the env name.
3. **Tool discovery** — `MCPToolClient.list_tools()` returns the expected list.
4. **Rollout** — `MAX_TURNS=3 uv run python rollout.py` runs without errors.

## Common gotchas (from real-world OpenEnv work)

- **`KeyError: 'tools'` from POST /list_tools** — OpenEnv doesn't expose `/list_tools` directly; `MCPToolClient` uses `/step` with a list-tools action under the hood. Always discover via the client.
- **Screenshot returns `None`** — `env.call_tool("screenshot")` returns only the structured `data` field. Use `env.step(CallToolAction(...))` and read `obs.result["content"]`.
- **`address already in use`** — common during local-dev iteration. Just pick a different `--port`.
- **`ModuleNotFoundError` in Docker but works locally** — missing dual-import pattern in `server/app.py` or `server/<name>_environment.py`.

## Reference

- `references/architecture.md` — full architecture deep-dive (when needed)

## Official documentation

- [openenv/OpenEnv](https://github.com/openenv/OpenEnv) — source repo
- [OpenEnv docs](https://huggingface.co/docs/openenv) — environment-builder + Core API
- [Build your first environment](https://huggingface.co/docs/openenv/guides/first-environment)
- [HF org](https://huggingface.co/openenv) — example deployments
- Upstream ships a `generate-openenv-env` skill at `.claude/skills/generate-openenv-env/` in their repo — useful as a second opinion if behaviour is unclear.
