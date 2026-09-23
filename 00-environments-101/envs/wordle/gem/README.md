# Wordle — GEM

A multi-turn Wordle environment, packaged using **[GEM](https://github.com/axon-rl/gem)** (in-process Gymnasium-style framework).

In-process — no server. `step()` returns the classic Gymnasium 5-tuple `(obs, reward, terminated, truncated, info)`.

## What this environment exposes

`WordleGemEnv(gem.Env)` parses the model's free-text action (either `<guess>word</guess>` or a reply that is just the 5-letter word) and submits it to the shared `WordleGame`. Words are not taken from surrounding prose, so a reply like "Could you share your last guess?" is not played as `guess`; pass `lenient_parsing=True` to restore the earlier fallback (a "guess is X" phrase, then the last 5-letter word in the text).

| Method | Returns |
|---|---|
| `reset()` | `(instruction_string, info)` |
| `step(action)` | `(observation, reward, terminated, truncated, info)` |
| `close()` | — |

`terminated=True` on win/loss (natural end), `truncated=True` only if you exceed `max_turns` mid-game (won't typically fire since the underlying `WordleGame` already enforces 6 guesses).

## How to consume it

### Native pattern

```python
import gem
from env import WordleGemEnv      # auto-registers as "wordle:Wordle-v0" on import

env = gem.make("wordle:Wordle-v0", answer="apple")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step("<guess>crane</guess>")
```

Or instantiate directly:

```python
from env import WordleGemEnv
env = WordleGemEnv(max_turns=6)
obs, info = env.reset()
```

### Manual rollout (`rollout.py` — what we ship)

The rollout sends the LLM's raw response as the action, parses the 5-tuple, appends the obs as the next user message.

## Run the rollout

```bash
cd 00-environments-101/envs/wordle/gem
uv sync
uv run python rollout.py
```

### Sample output

```
GEM env:    in-process WordleGemEnv (pure Python)
Provider:   hf-router    Model: Qwen/Qwen3-Coder-480B-A35B-Instruct:together

[reset] info={'answer': 'olive', 'max_turns': 6}
[obs] Play Wordle! Guess the hidden 5-letter word in 6 attempts...

──── turn 1 ────────────────────────────────────────
[assistant] <guess>crane</guess>
[env-result reward=0.0 terminated=False truncated=False] ⬛⬛⬛🟨⬛ — 5 guesses remaining.
...
──── turn 6 ────────────────────────────────────────
[assistant] <guess>alive</guess>
[env-result reward=0.4 terminated=True truncated=False] ⬛🟩🟩🟩🟩 — Game over! The word was 'olive'.

FINAL  cumulative_reward=0.4  done=True
```

### Configuration knobs

| Variable | Default | Purpose |
|---|---|---|
| `ROLLOUT_MODEL` | `Qwen/Qwen3-Coder-480B-A35B-Instruct:together` | Provider auto-detect. |
| `MAX_TURNS` | `6` | Wordle's 6-guess limit. |

## What's in this folder

| Path | What it is |
|---|---|
| `rollout.py` | Self-contained consumer demo. |
| `env.py` | `WordleGemEnv(gem.Env)`. Auto-registers as `wordle:Wordle-v0`. |
| `pyproject.toml` | `gem-llm` + rollout-side `openai`, `python-dotenv`. |

## In-process means: no server to run

There's nothing to deploy.

## References

- [GEM GitHub](https://github.com/axon-rl/gem)
