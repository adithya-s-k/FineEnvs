# Wordle — SkyRL Gym

A multi-turn Wordle environment, packaged using **[SkyRL Gym](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym)** (in-process Gym-style framework with text-action dispatch).

In-process — no server, no external sandbox.

## What this environment exposes

`WordleSkyRLEnv(BaseTextEnv)` parses the model's free-text action for either:

- `<guess>word</guess>` tags, or
- a reply that is just the 5-letter word

Words are not taken from surrounding prose; `WordleSkyRLEnv(lenient_parsing=True)` restores the earlier fallback (a "guess is X" phrase, then the last 5-letter word in the text).

…then submits it to the shared `WordleGame` and returns `BaseTextEnvStepOutput(observations, reward, done, ...)`. The terminal reward is paid once on the step that ends the game; later calls after `done=True` return `reward=0.0`.

Reward (set by `WordleGame`):
- `1.0` win
- partial credit (0.1–0.5) for correct letters / positions
- `0.0` otherwise

`done` is True once the game is over (win, 6 guesses used, or invalid limit hit).

## How to consume it

### Native pattern

```python
import skyrl_gym
env = skyrl_gym.make("wordle:Wordle-v0")          # auto-registered on import of env.py
obs, info = env.init([{"role": "user", "content": "Play Wordle"}])
out = env.step("<guess>crane</guess>")
# out.observations -> [{"role": "user", "content": "⬛⬛⬛🟨🟩 — 5 guesses remaining."}]
```

Or instantiate directly:

```python
from env import WordleSkyRLEnv
env = WordleSkyRLEnv(max_turns=6)
obs, info = env.init([{"role": "user", "content": "..."}])
print(info["answer"])     # the hidden word, for diagnostics
```

### Manual rollout (`rollout.py` — what we ship)

The rollout sends the LLM's raw response as the action. `step()` extracts the guess.

## Run the rollout

```bash
cd 00-environments-101/envs/wordle/skyrl_gym
uv sync
uv run python rollout.py
```

### Sample output

```
SkyRL Gym env: in-process WordleSkyRLEnv (pure Python)
Provider:      hf-router    Model: Qwen/Qwen3-Coder-480B-A35B-Instruct:together

[init] info={'max_turns': 6, 'answer': 'noble'}

──── turn 1 ────────────────────────────────────────
[assistant] <guess>crane</guess>
[env-result reward=0.0 done=False] ⬛⬛⬛🟨🟩 — 5 guesses remaining.
...
──── turn 6 ────────────────────────────────────────
[assistant] <guess>metal</guess>
[env-result reward=0.1 done=True] ⬛🟨⬛⬛🟨 — Game over! The word was 'noble'.

FINAL  cumulative_reward=0.1  done=True  answer=noble
```

### Configuration knobs

| Variable | Default | Purpose |
|---|---|---|
| `ROLLOUT_MODEL` | `Qwen/Qwen3-Coder-480B-A35B-Instruct:together` | Provider auto-detect by `:` suffix. |
| `MAX_TURNS` | `6` | One per allowed Wordle guess. |

## What's in this folder

| Path | What it is |
|---|---|
| `rollout.py` | Self-contained consumer demo. |
| `env.py` | `WordleSkyRLEnv(BaseTextEnv)`. Registers as `wordle:Wordle-v0` on import. |
| `pyproject.toml` | `skyrl-gym` + rollout-side `openai`, `python-dotenv`. |

## In-process means: no server to run

There's nothing to deploy.

## References

- [SkyRL GitHub](https://github.com/NovaSky-AI/SkyRL)
