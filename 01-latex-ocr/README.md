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
| Configurations, provenance, and reproduction commands | [`REPRODUCE.md`](./REPRODUCE.md) |
| Hosted environment | [HuggingEnvs/latex-ocr-env](https://huggingface.co/spaces/HuggingEnvs/latex-ocr-env) |
| Collection | [LaTeX OCR — final assets](https://huggingface.co/collections/HuggingEnvs/latex-ocr-6aa7ed8498b3ffd222ad1c8c) |
| Published GRPO checkpoint | [HuggingEnvs/qwen3.5-latex-ocr-grpo](https://huggingface.co/HuggingEnvs/qwen3.5-latex-ocr-grpo) |
| Training dashboard | [HuggingEnvs/trackio-latex-ocr](https://huggingface.co/spaces/HuggingEnvs/trackio-latex-ocr) |
| Hub organization | [HuggingEnvs](https://huggingface.co/HuggingEnvs) |

The collection contains the final environment, published model, source dataset, and **one combined results
dashboard**. Its main view overlays five runs across four models, including **both unstable and stabilized
Gemma4-E2B**. The same dashboard includes 12 earlier run segments and the original Qwen3.5 overnight run and sample images.
All views use the single `HuggingEnvs/latex-ocr-results` bucket. Original experiment assets remain under
`AdithyaSK`. See [the model comparison](./results/model-comparison.md) for the measured gains and run provenance.

The published Qwen3.5 checkpoint and historical dashboard are separate from the current Qwen3-VL-2B recipe
and its two-step HF Jobs smoke. See the [results notes](./results/README.md) for pipeline verification.

## Recorded experiments

The main Trackio view preserves the following historical evaluation rewards. Endpoints differ because the
unstable Gemma snapshot ends earlier; its curve is not extended beyond the recorded evaluation.

| Run | Baseline | Last evaluation | Evaluation step | Change |
|---|---:|---:|---:|---:|
| Qwen3-VL-2B / colocated | 0.6154 | 0.7231 | 3,500 | +0.1077 |
| Qwen3.5-2B / colocated | 0.6877 | 0.7101 | 3,500 | +0.0224 |
| GLM-OCR / colocated | 0.4494 | 0.6644 | 3,500 | +0.2150 |
| Gemma4-E2B / unstable | 0.3917 | 0.1322 | 1,500 | **−0.2595** |
| Gemma4-E2B / stabilized | 0.3917 | 0.5697 | 3,500 | +0.1780 |

At the same step **1,500**, stabilized Gemma scores **0.5487**, versus **0.1322** for the unstable run.
The unstable run briefly reaches **0.4395** at step 1,000 before regressing. Its first/last 50 training
records average **0.4046 → 0.2595** reward. Both outcomes belong in the comparison.

The stabilized configuration changes learning rate **1e-5 → 5e-6**, KL coefficient **0 → 0.03**,
temperature **0.9 → 0.8**, maximum gradient norm **1.0 → 0.5**, and vLLM GPU memory utilization
**0.30 → 0.18**. Both Gemma runs use Adafactor. These simultaneous changes do not identify which setting
caused the improvement. See [REPRODUCE.md](./REPRODUCE.md) for the complete recorded settings and sources.

The recovered launchers specify **full fine-tuning**, colocated vLLM, 5,000 training images, and greedy
evaluation on the first 200 test images every 500 steps. The [original harness and launchers](./train/historical/)
are included with source hashes. That recovered snapshot postdates the logged runs and contains a session
cleanup fix; it does not prove the exact code/dependency versions used for each historical score.

The available training snapshots end at **1,999 steps** for unstable Gemma and **3,999** for the other four
runs, against 5,000 configured steps. The logs do not prove completion. Evaluation sample counts and the
exact historical scoring revision are absent from the comparison configs. Final-window completion clipping
is **99.75%** for unstable Gemma and **100%** for stabilized Gemma and GLM; interpret reward with those
diagnostics. These are logged experiment outcomes, not a controlled leaderboard or a newly rerun benchmark.

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
