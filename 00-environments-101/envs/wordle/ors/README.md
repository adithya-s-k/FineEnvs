# Wordle — ORS

A multi-turn Wordle environment, packaged using **[ORS / OpenReward](https://openrewardstandard.io)** (REST + SSE protocol with bundled tasks).

**Deployed:** [`AdithyaSK/wordle-ors`](https://huggingface.co/spaces/AdithyaSK/wordle-ors)

## What this environment exposes

The ORS server registers two `@tool`-decorated methods:

| Tool | Description |
|---|---|
| `guess(word: str)` | Submit a 5-letter word. Returns colored feedback (`🟩🟨⬛`). Per-call `reward`. |
| `get_history()` | List all previous guesses with feedback. |

Tasks are bundled with the env: 50 hand-crafted answers in the `train` split, each shaped as `Task(task_spec={"task": "...", "answer": "<word>"})`. Reward arrives **per tool call** as `ToolOutput.reward`, with `finished=True` set on win/loss. The terminal reward is paid once; later calls after the game is over keep `finished=True` but return `reward=None`.

## How to consume it

```python
from openreward import EnvironmentsAPI

api = EnvironmentsAPI(base_url="https://AdithyaSK-wordle-ors.hf.space", api_key="")
env = api.get("wordleors")
tasks = env.list_tasks("train")              # 50 tasks
with env.session(task=tasks[0]) as session:
    print(session.get_prompt())               # rules + symbols legend
    out = session.call_tool("guess", {"word": "crane"})
    # ToolOutput(blocks=[TextBlock(text='⬛⬛🟨⬛🟩 — 5 guesses remaining.')], reward=0.0, finished=False)
```

## Run the rollout

```bash
cd 00-environments-101/envs/wordle/ors
uv sync
uv run python rollout.py                 # talks to deployed HF Space
# or local:
uv run python server.py                  # serves on :8080
ORS_URL=http://localhost:8080 uv run python rollout.py
```

### Sample output

```
ORS server: https://AdithyaSK-wordle-ors.hf.space
Env name:   wordleors    split=train  task_index=0
Provider:   hf-router    Model: Qwen/Qwen3-Coder-480B-A35B-Instruct:together

50 tasks in 'train'. Using task #0: answer=apple
Discovered 2 tools: ['get_history', 'guess']

[task] Play Wordle! Guess the hidden 5-letter word in 6 attempts...

──── turn 1 ────────────────────────────────────────
[tool-call] guess({'word': 'crane'})
[tool-result reward=0.0 finished=False] ⬛⬛🟨⬛🟩 — 5 guesses remaining.
...
```

### Configuration knobs

| Variable | Default | Purpose |
|---|---|---|
| `ORS_URL` | `https://AdithyaSK-wordle-ors.hf.space` | ORS server URL. |
| `ORS_ENV_NAME` | `wordleors` | Env name registered on the server. |
| `TASK_INDEX` | `0` | Which task in the train split. |
| `ROLLOUT_MODEL` | `Qwen/Qwen3-Coder-480B-A35B-Instruct:together` | If contains `:` → HF Router. Else → OpenAI native. |
| `MAX_TURNS` | `6` | One turn per allowed guess. |

## What's in this folder

| Path | What it is |
|---|---|
| `rollout.py` | Self-contained consumer demo. |
| `server.py` | The ORS `Environment` subclass deployed to the HF Space. |
| `Dockerfile`, `Dockerfile.spaces`, `README.spaces.md` | Deployment to HF Spaces. |
| `pyproject.toml` | `openreward` + rollout-side `openai`, `python-dotenv`. |

## References

- [Open Reward Standard](https://openrewardstandard.io)
- [openreward on PyPI](https://pypi.org/project/openreward/)
- [OpenReward docs](https://docs.openreward.ai/) · [python-sdk source](https://github.com/openrewardstandard/python-sdk)
