# Wordle — NeMo Gym

A multi-turn Wordle environment, packaged using **[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)** (NVIDIA's HTTP-server framework with REST endpoints + cookie-based sessions).

**Deployed:** [`AdithyaSK/wordle-nemo-gym`](https://huggingface.co/spaces/AdithyaSK/wordle-nemo-gym)

## What this environment exposes

| Endpoint | Body | Description |
|---|---|---|
| `POST /seed_session` | `{}` | Initialize a fresh game; sets a session cookie. |
| `POST /guess` | `{"word": "..."}` | Submit a 5-letter word. Returns `{"output": "<feedback>"}`. |
| `POST /get_history` | `{}` | Return the full guess history. |
| `POST /verify` | NeMoGym verify schema | Post-episode reward computation. |

Sessions are managed via cookies. Reward is **computed post-episode** by `/verify`, not per-call (Pattern 3 in the COMPARE doc). It uses the shared Wordle reward: a win gets the speed bonus from `WordleGame.reward`, and a loss gets `0.1 ×` the best green-letter count. Only exact `guess` tool outputs count; the model's own messages are not read, so writing "Correct" earns nothing.

## How to consume it

Plain `requests`, no SDK:

```python
import requests

s = requests.Session()
s.post("https://AdithyaSK-wordle-nemo-gym.hf.space/seed_session", json={}).raise_for_status()
r = s.post("https://AdithyaSK-wordle-nemo-gym.hf.space/guess", json={"word": "crane"})
print(r.json()["output"])   # '⬛⬛🟨⬛🟩 — 5 guesses remaining.'
```

## Run the rollout

```bash
cd 00-environments-101/envs/wordle/nemo_gym
uv sync                                  # needs Python 3.12
uv run python rollout.py                 # talks to deployed HF Space
```

### Sample output

```
NeMo Gym server: https://AdithyaSK-wordle-nemo-gym.hf.space
Provider: hf-router    Model: Qwen/Qwen3-Coder-480B-A35B-Instruct:together

[seed_session] cookies={'WordleResourcesServer___wordle': '...'}

──── turn 1 ────────────────────────────────────────
[tool-call] guess({'word': 'crane'})
[tool-result] ⬛⬛⬛⬛⬛ — 5 guesses remaining.
...
──── turn 5 ────────────────────────────────────────
[tool-call] guess({'word': 'vivid'})
[tool-result] 🟩🟩🟩🟩🟩 — Correct! The word was 'vivid'. Solved in 5 guesses.

[done] game ended, stopping.
```

> ⚠️ Same caveat as the Jupyter NeMo Gym env: **local server requires Ray**, which fails on shared HF / SLURM cluster nodes (`gcs_server` can't bind). The deployed Space is the recommended path. See `00-environments-101/envs/jupyter/nemo_gym/README.md` for the full story.

### Configuration knobs

| Variable | Default | Purpose |
|---|---|---|
| `NEMO_GYM_URL` | `https://AdithyaSK-wordle-nemo-gym.hf.space` | NeMo Gym server URL. |
| `ROLLOUT_MODEL` | `Qwen/Qwen3-Coder-480B-A35B-Instruct:together` | If contains `:` → HF Router. Else → OpenAI native. |
| `MAX_TURNS` | `6` | One turn per allowed guess. |

## What's in this folder

| Path | What it is |
|---|---|
| `rollout.py` | Self-contained consumer demo — raw HTTP, no SDK. |
| `server.py` | The `SimpleResourcesServer` deployed to the HF Space. |
| `configs/wordle.yaml` | NeMo Gym Hydra config. |
| `Dockerfile`, `Dockerfile.spaces`, `README.spaces.md` | Deployment to HF Spaces. |
| `pyproject.toml` | `nemo_gym` (git) + rollout-side `openai`, `python-dotenv`, `requests`. **Python 3.12 required.** |

## References

- [NeMo Gym GitHub](https://github.com/NVIDIA-NeMo/Gym)
