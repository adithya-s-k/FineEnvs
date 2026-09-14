---
title: LaTeX OCR Env
emoji: 📐
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
base_path: /web
---

# LaTeX OCR Environment

A dataset-backed, single-step (bandit) RL environment for **image → LaTeX**
transcription, served through OpenEnv.

- **Task**: the agent is shown an image of a math/text expression and must
  return its LaTeX source.
- **Dataset**: tasks are served from a Hugging Face dataset (default
  [`unsloth/LaTeX_OCR`](https://huggingface.co/datasets/unsloth/LaTeX_OCR):
  `image` + `text` columns, `train`/`test` splits) via the OpenEnv **Task API**.
- **Reward**: `[(1 - exact_weight) * (1 - CER) + exact_weight * exact_match] * length_factor`,
  where `CER` is the normalized character edit distance over whitespace-stripped
  LaTeX. With the default `exact_weight=0.4` (tunable via `LATEX_OCR_EXACT_WEIGHT`),
  partial answers score in `[0, 0.6]`; only an exact match reaches `1.0`.
  Computed **server-side** against the hidden ground truth — the agent never
  sees the target on `reset`. Dense and smooth, for stable RL training.
- **Length guard**: because the score is whitespace-insensitive, a policy could
  otherwise emit the correct answer followed by unlimited whitespace and still
  score `1.0` — a reward hack that, in RL, shows up as completions drifting to
  the generation-length cap. `length_factor` decays the reward once the *raw*
  prediction grows past `LATEX_OCR_OVERLONG_RATIO`× the target length (default
  `4.0`, floor `LATEX_OCR_OVERLONG_FLOOR=80` chars); normal LaTeX spacing is well
  within the allowance. Set `LATEX_OCR_OVERLONG_RATIO=0` to disable. The rubric
  also strips code fences / `$`-delimiters from raw completions itself.

## Episode

```
reset(split="test", index=0)  -> observation { image_base64, prompt, ... }   # target hidden
step(LatexOCRAction(latex=…)) -> reward, done=True { target_latex, exact_match, char_error_rate }
```

## Task API

| Endpoint | Purpose |
|---|---|
| `GET  /latex_ocr_env/splits` | list splits (`train`, `test`) |
| `POST /latex_ocr_env/num_tasks` | row count for a split |
| `POST /latex_ocr_env/tasks` | all task specs for a split (materialize mode) |
| `POST /latex_ocr_env/task` | one task by `{split, index}` |
| `POST /latex_ocr_env/task_range` | slice `{split, start, stop}` |

Client helpers: `env.list_splits()`, `env.num_tasks(split)`,
`env.get_task(split, index)`, `env.get_task_range(split, start, stop)`.
In stream mode, full task enumeration returns `501`; use `num_tasks` and
bounded `get_task_range` calls instead.

## Run locally

From the HuggingEnvs repository root:

```bash
LATEX_OCR_MAX_ROWS=50 uv run --project 01-latex-ocr/envs/latex_ocr latex-ocr-server
```

Open `http://localhost:8000/web` for the Gradio playground. `GET /healthz` checks process health;
`latex-ocr-smoke` additionally verifies dataset access and scoring. The package pins OpenEnv 0.4.2,
which includes the Task API, and installs the Gradio dependency used by the app.

```bash
uv run --project 01-latex-ocr/envs/latex_ocr --extra dev \
  python 01-latex-ocr/envs/latex_ocr/validate.py --base-url http://localhost:8000 --split test --num 3
```

The validator uses a VLM policy if `HF_TOKEN` is set; otherwise it submits empty predictions.
For deterministic assertions, use `latex-ocr-smoke` instead.

## Docker and Spaces

The environment directory is a standalone Docker build context, including this Space card:

```bash
docker build -t latex-ocr-env 01-latex-ocr/envs/latex_ocr
docker run --rm -p 8000:8000 -e LATEX_OCR_MAX_ROWS=50 latex-ocr-env
```

To deploy to your own Docker Space (replace the ID):

```bash
hf repo create your-name/latex-ocr-env --repo-type space --space-sdk docker
hf upload your-name/latex-ocr-env 01-latex-ocr/envs/latex_ocr . --repo-type space \
  --exclude '.venv/**' '__pycache__/**' '*.egg-info/**' '.pytest_cache/**' '.ruff_cache/**'
```

The existing deployment is [AdithyaSK/latex-ocr-env](https://huggingface.co/spaces/AdithyaSK/latex-ocr-env).
This migration does not redeploy that Space. Set runtime configuration in your Space's Settings.

## Access modes

`materialize` supports indexed reset. A configured row cap streams only that many rows and caches the bounded
result in memory; without a cap it loads the full split. Training needs materialize mode.

`stream` advances a cursor on every reset and does not download the whole split. Indices are zero-based;
`pct_done` counts consumed rows. Each session owns its cursor and starts its own pass: no-repeat applies within
a session, not across concurrent sessions. After an exhausted observation, the next reset starts a new pass.
Task discovery does not advance the cursor; stream task indices describe positions, and indexed reset is rejected.

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `LATEX_OCR_MODE` | `materialize` | `materialize` or `stream` |
| `LATEX_OCR_MAX_SESSIONS` | `16` | maximum simultaneous WebSocket environments |
| `LATEX_OCR_CONFIG` | — | optional dataset configuration |
| `LATEX_OCR_SHUFFLE_BUFFER` | `0` | optional stream shuffle buffer |
| `LATEX_OCR_SEED` | `0` | stream shuffle seed (same order across sessions) |
| `LATEX_OCR_STREAM_RANGE_CAP` | `100000` | maximum metadata tasks in one stream range request |
| `LATEX_OCR_DATASET` | `unsloth/LaTeX_OCR` | source dataset |
| `LATEX_OCR_IMAGE_COLUMN` | `image` | image column |
| `LATEX_OCR_TEXT_COLUMN` | `text` | ground-truth LaTeX column |
| `LATEX_OCR_SPLITS` | `train,test` | splits to expose |
| `LATEX_OCR_MAX_ROWS` | — | cap rows per split (dev) |
| `LATEX_OCR_EXACT_WEIGHT` | `0.4` | exact-match share of the reward |
| `LATEX_OCR_OVERLONG_RATIO` | `4.0` | raw length allowed as a multiple of the target before the reward decays (`0` disables the length guard) |
| `LATEX_OCR_OVERLONG_FLOOR` | `80` | minimum allowed raw length (chars) for short targets |

Swap in any `(image, latex)` dataset by pointing `LATEX_OCR_DATASET` at it (and
the column vars if they differ).

## Origin and license

Adapted from [OpenEnv #1003](https://github.com/huggingface/OpenEnv/pull/1003), commit
`9fc674b9922f22d2e62274b09476236bcf7984eb`. Imported source retains its
[BSD-3-Clause license](./LICENSE) and [attribution](./NOTICE).
