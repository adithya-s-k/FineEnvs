# Harness logos

Project marks for the agent harnesses shown in `banner-opaque.html`, used
nominatively to identify each project. Each mark remains its project's property.

Where a project publishes a vector mark on its own site we use that; otherwise
the file is the project's GitHub organisation avatar, normalised to 128x128 PNG.

| file | source |
| --- | --- |
| `claude-code.svg` | claude.com |
| `opencode.svg` | opencode.ai |
| `openhands-sdk.svg` | all-hands.dev |
| `swe-agent.svg` | swe-agent.com |
| `codex.png` | github.com/openai |
| `gemini-cli.png` | github.com/google-gemini |
| `goose.png` | github.com/block |
| `qwen-coder.png` | github.com/QwenLM |
| `hermes.png` | github.com/NousResearch |
| `kimi-cli.png` | github.com/MoonshotAI |
| `trae-agent.png` | github.com/bytedance |
| `terminus-2.png` | github.com/harbor-framework — terminus-2 is a Harbor harness, so it carries Harbor's mark |

To add a harness: drop `<name>.svg` or `<name>.png` here, add the name to
`HARNESSES` in `banner-opaque.html`, and list it in `SVG_MARKS` if it is a
vector. The grid resizes itself; a missing file falls back to a coloured tile.
