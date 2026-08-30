# Harness logos

Project marks for the agent harnesses shown in `banner-opaque.html`, used
nominatively to identify each project.

Each file is that project's GitHub organisation avatar, fetched from
`https://github.com/<org>.png` and normalised to 128×128 PNG:

| file | org |
| --- | --- |
| `claude-code.png` | anthropics |
| `codex.png` | openai |
| `gemini-cli.png` | google-gemini |
| `opencode.png` | sst |
| `goose.png` | block |
| `qwen-coder.png` | QwenLM |
| `swe-agent.png` | SWE-agent |
| `openhands-sdk.png` | All-Hands-AI |
| `hermes.png` | NousResearch |
| `kimi-cli.png` | MoonshotAI |
| `terminus-2.png` | laude-institute |
| `trae-agent.png` | bytedance |

Each mark remains the property of its project. To add a harness, drop
`<name>.png` here and add the name to `HARNESSES` in `banner-opaque.html` —
the grid resizes itself, and a missing file falls back to a coloured tile.
