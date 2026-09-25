# Wordle — OpenEnv

A multi-turn Wordle environment, packaged using **[OpenEnv](https://github.com/huggingface/OpenEnv)** (Hugging Face's HTTP-server / MCP-protocol framework). Pure Python game logic, no E2B sandbox needed — Wordle is the cleanest way to see how OpenEnv handles a multi-turn env.

**Deployed:** [`AdithyaSK/wordle-openenv`](https://huggingface.co/spaces/AdithyaSK/wordle-openenv)

## What this environment exposes

The OpenEnv server registers three MCP tools (auto-discoverable via `list_tools()`):

| Tool | Description |
|---|---|
| `guess(word: str)` | Submit a 5-letter word. Returns colored feedback (`🟩🟨⬛`) and remaining-guesses count. |
| `get_history()` | View all previous guesses with their feedback. |
| `reset_game()` | Start a new game with a fresh random word once the current game is over (it refuses while a game is in progress). |

A guess returns a string like `'⬛🟨⬛⬛🟨 — 5 guesses remaining.'`:
- 🟩 letter is in the correct position
- 🟨 letter is in the word but wrong position
- ⬛ letter is not in the word

Episode ends after 6 guesses or a correct word. Last response includes `'Game over! The word was '<answer>'.'`. The step that ends the game returns `done=True` and `reward` set to `WordleGame.reward` (earlier steps: `done=False`, `reward=None`), so RL clients can score episodes without parsing the text. The reward is reported once: any step after the end returns `done=True` with `reward=None`, so an episode's score is its last non-null reward, and summing per-step rewards gives the same number. For a remote client the observations are the only place it arrives: over the WebSocket, OpenEnv's `State` keeps only `episode_id` and `step_count`, so `state["reward"]` is available in-process only. `reset(answer=...)` pins the hidden word, e.g. for fixed evaluation sets.

## How to consume it

```python
from openenv.core.mcp_client import MCPToolClient

with MCPToolClient(base_url="https://AdithyaSK-wordle-openenv.hf.space").sync() as env:
    env.reset()
    out = env.call_tool("guess", word="crane")
    print(out)   # '⬛⬛🟨🟨⬛ — 5 guesses remaining.'
```

No env-specific install needed — `openenv`'s generic `MCPToolClient` discovers the tools at runtime.

## Run the rollout

`rollout.py` plays a full Wordle game with **Qwen3-Coder-480B** via Hugging Face Inference Providers (auto-falls-back to OpenAI if `ROLLOUT_MODEL` has no `:provider` suffix).

### Setup

Repo-root `.env`:

```bash
HF_TOKEN=hf_...        # for HF Inference Providers (Qwen)
OPENAI_API_KEY=sk-...  # only if you switch ROLLOUT_MODEL to an OpenAI model
```

### Run

```bash
cd 00-environments-101/envs/wordle/openenv
uv sync
uv run python rollout.py                 # talks to deployed HF Space
# or run the env locally first:
uv run python -m server.app              # serves on :8000
OPENENV_URL=http://localhost:8000 uv run python rollout.py
```

### Sample output

```
OpenEnv server: https://AdithyaSK-wordle-openenv.hf.space
Provider:       hf-router
Model:          Qwen/Qwen3-Coder-480B-A35B-Instruct:together

Discovered 3 tools: ['guess', 'get_history', 'reset_game']

──── turn 1 ────────────────────────────────────────
[tool-call] guess({'word': 'crane'})
[tool-result] ⬛⬛🟨🟨⬛ — 5 guesses remaining.
──── turn 2 ────────────────────────────────────────
[tool-call] guess({'word': 'slate'})
[tool-result] ⬛⬛🟨⬛⬛ — 4 guesses remaining.
...
[tool-result] 🟨⬛🟨⬛⬛ — Game over! The word was 'mango'.

[done] env reported game over (won=False).
```

### Configuration knobs

| Variable | Default | Purpose |
|---|---|---|
| `OPENENV_URL` | `https://AdithyaSK-wordle-openenv.hf.space` | OpenEnv server URL. |
| `ROLLOUT_MODEL` | `Qwen/Qwen3-Coder-480B-A35B-Instruct:together` | If contains `:` → HF Router. Else → OpenAI native. |
| `MAX_TURNS` | `8` | Hard cap (Wordle allows 6 guesses + a couple of slack turns for `reset_game` etc.). |

## What's in this folder

| Path | What it is |
|---|---|
| `rollout.py` | Self-contained consumer demo. |
| `server/` | The OpenEnv server source deployed to the HF Space (env definition with `@mcp.tool` registrations). |
| `Dockerfile`, `Dockerfile.spaces`, `README.spaces.md` | Build / deploy the server image to HF Spaces. |
| `pyproject.toml` | `openenv`, `fastmcp`, plus rollout-side `openai`, `python-dotenv`. |

## References

- [OpenEnv tutorial](https://github.com/huggingface/OpenEnv/blob/main/tutorial/01-environments.md)
- [HF Inference Providers — Chat Completion API](https://huggingface.co/docs/inference-providers/en/tasks/chat-completion)
