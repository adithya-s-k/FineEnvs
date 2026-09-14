# 01 · LaTeX OCR

Train **Qwen3-VL-2B** to transcribe formula images into LaTeX with GRPO and an OpenEnv reward server.
This project houses the environment, notebook, training and HF Jobs scripts, and smoke-test results.
It continues the environment originally proposed in [OpenEnv #1003](https://github.com/huggingface/OpenEnv/pull/1003).

| Part | Source |
|---|---|
| Environment, client, rubric, Docker Space, tests | [`envs/latex_ocr/`](./envs/latex_ocr/) |
| Interactive recipe | [`notebooks/01_latex_ocr_grpo.ipynb`](./notebooks/01_latex_ocr_grpo.ipynb) |
| Shared GRPO runner and HF Jobs entry point | [`train/`](./train/) |
| Curated Trackio dashboard source | [`dashboard/`](./dashboard/) |
| Verification evidence | [`results/`](./results/) |
| Hosted environment | [HuggingEnvs/latex-ocr-env](https://huggingface.co/spaces/HuggingEnvs/latex-ocr-env) |
| Collection | [LaTeX OCR — final assets](https://huggingface.co/collections/HuggingEnvs/latex-ocr-6aa7ed8498b3ffd222ad1c8c) |
| Published GRPO checkpoint | [HuggingEnvs/qwen3.5-latex-ocr-grpo](https://huggingface.co/HuggingEnvs/qwen3.5-latex-ocr-grpo) |
| Training dashboard | [HuggingEnvs/trackio-latex-ocr](https://huggingface.co/spaces/HuggingEnvs/trackio-latex-ocr) |
| Hub organization | [HuggingEnvs](https://huggingface.co/HuggingEnvs) |

The collection contains the final environment, published model, source dataset, and **one training/evaluation
dashboard**. Its backing bucket, `HuggingEnvs/latex-ocr-results`, holds the Qwen3.5 training history
through 3,000 optimizer steps, baseline/checkpoint evaluations, and sample predictions. Experiment dashboards
and trial buckets remain under `AdithyaSK` and are excluded from the collection.

The published Qwen3.5 checkpoint and historical dashboard are separate from the current Qwen3-VL-2B recipe
and its two-step HF Jobs smoke. See the [results notes](./results/README.md) for provenance and verification.

## Smoke test

From the repository root, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv run --project 01-latex-ocr/envs/latex_ocr --extra dev pytest 01-latex-ocr/envs/latex_ocr/tests -q
uv run --project 01-latex-ocr/envs/latex_ocr latex-ocr-smoke
uv run --project 01-latex-ocr/envs/latex_ocr latex-ocr-smoke --real-data
```

The default smoke uses tiny offline fixtures. `--real-data` loads two actual dataset rows per split.
Both start and stop their own server and test materialized and streaming access, task discovery, PNG observations,
isolated sessions, and server-side rewards. To check the existing Space:

```bash
uv run --project 01-latex-ocr/envs/latex_ocr latex-ocr-smoke \
  --url https://huggingenvs-latex-ocr-env.hf.space
```

## Train

The notebook and Jobs use the same Python runner. On a local CUDA GPU:

```bash
uv run --project 01-latex-ocr/envs/latex_ocr --extra train \
  python 01-latex-ocr/train/grpo_latex_ocr.py --smoke
```

Or on HF Jobs, from a pushed checkout:

```bash
hf auth login
hf jobs uv run --flavor a10g-small --timeout 30m --secrets HF_TOKEN \
  01-latex-ocr/train/hf_job.py --revision "$(git rev-parse HEAD)" --smoke
```

The GPU smoke performs two optimizer steps with two generations per group, evaluates two held-out images before
and after, and verifies an adapter weight update. It is a pipeline check, not evidence of quality improvement.
See [`train/README.md`](./train/README.md) for full runs, persistent output, and hosted environments.
For an interactive GPU notebook, the repository launcher remains available:

```bash
python3 tools/jupyter_launch.py --flavor a100-large
```

## Task and reward

`reset(split, index)` returns a formula image and instruction. `step(latex)` ends the episode, grades against
the hidden dataset reference, and reveals the reference for logging. The default score is:

```text
reward = [0.6 × (1 − normalized edit distance) + 0.4 × exact match] × length factor
```

The scorer strips formatting and whitespace for comparison but measures **raw completion length** to penalize
padding. This is a string-based OCR reward: it does not render LaTeX or prove mathematical equivalence.
Training uses indexed materialize mode so every completion in a group receives the same image. Streaming is
sequential per session; concurrent sessions each begin their own pass.
