---
title: Desktop Environment Server
emoji: 🖥️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - computer-use
  - desktop
  - e2b
short_description: Cloud Linux desktop with computer-use tools, backed by E2B
---

# Desktop Environment (OpenEnv)

`desktop_env` exposes a full Linux desktop via the [OpenEnv](https://github.com/openenv/OpenEnv) MCP-tool protocol. Each episode spins up a fresh [E2B Desktop](https://e2b.dev) sandbox; the agent observes the screen as PNG image blocks and drives the mouse/keyboard with tool calls.

The action surface mirrors **Anthropic's `computer_20251124`** schema (the broadest superset across Claude / OpenAI Operator / Qwen3-VL ComputerUse), so a model's native computer-use output drives the env with minimal token-level adaptation.

## Tools (19)

| Group | Tools |
|---|---|
| Observation | `screenshot`, `cursor_position`, `get_screen_size` |
| Mouse — clicks | `left_click`, `right_click`, `middle_click`, `double_click`, `triple_click` |
| Mouse — motion | `mouse_move`, `left_click_drag`, `left_mouse_down`, `left_mouse_up`, `scroll` |
| Keyboard | `type`, `key`, `hold_key` |
| Control | `wait`, `terminate`, `run_command` |

All click tools accept `coordinate=[x, y]` in pixel space and an optional `text` modifier (`"shift"`, `"ctrl"`, `"ctrl+shift"`, etc.). `screenshot` returns an MCP **image content block** so vision models actually see the pixels — not a base64 string in text.

## Reset kwargs

| Key | Default | Notes |
|---|---|---|
| `app` | `"desktop"` | Preset (`firefox`, `libreoffice-calc`, `terminal`, `gimp`, `blender`, …) or a raw shell launch command |
| `resolution` | `[1024, 768]` | `[w, h]` in pixels |
| `timeout` | `600` | Sandbox timeout, seconds |
| `install_commands` | `[]` | Optional setup commands run before launching |

## Local run

```bash
cd 00-environments-101/envs/desktop/openenv
uv sync
E2B_API_KEY=e2b_... uv run uvicorn server.app:app --port 8000
# UI:    http://localhost:8000/web
# MCP:   http://localhost:8000/mcp
```

## Rollouts

- `rollout_openai.py` — drives the env with OpenAI **`computer-use-preview`** (Responses API). Adapter maps `click/double_click/move/drag/scroll/keypress/type/wait` → MCP tool calls.
- `rollout_qwen.py` — drives the env with **Qwen3-VL** via HuggingFace Inference Providers (router). The 19 MCP tools are exposed in OpenAI function-calling format; Qwen3-VL emits `[x, y]` coordinates natively.

Part of the [exp_rl](https://github.com/adithya-s-k/exp_rl) project — demonstrating RL environment frameworks with TRL.
