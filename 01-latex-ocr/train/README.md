# Training and HF Jobs

`grpo_latex_ocr.py` is the shared notebook/CLI recipe. It serves the environment locally by default, evaluates
Qwen3-VL-2B on the test split, trains LoRA with TRL GRPO, evaluates the same indices again, and saves the results.
`hf_job.py` is a standalone PEP 723 script accepted directly by `hf jobs uv run`. It downloads a **pushed commit
SHA** of HuggingEnvs and uses the environment's frozen `uv.lock`; no OpenEnv feature branch is needed.

Run the commands below from the repository root. HF Jobs needs an authenticated account with compute credits.
The public dataset/model can be read without a token; `--secrets HF_TOKEN` forwards your stored token securely.

## CPU checks on HF Jobs

```bash
hf jobs uv run --flavor cpu-basic --timeout 15m \
  01-latex-ocr/train/hf_job.py --revision "$(git rev-parse HEAD)" --mode env-smoke

# Same checks against two real dataset rows per split:
hf jobs uv run --flavor cpu-basic --timeout 15m --secrets HF_TOKEN \
  01-latex-ocr/train/hf_job.py --revision "$(git rev-parse HEAD)" --mode env-smoke --real-data
```

## GPU smoke

```bash
hf jobs uv run --flavor a10g-small --timeout 30m --secrets HF_TOKEN \
  01-latex-ocr/train/hf_job.py --revision "$(git rev-parse HEAD)" --smoke
```

This fixes the run to two optimizer steps, two generations per group, and two test images. It checks finite loss,
completed steps, and changed adapter weights. A reward increase is not required. Output is printed in the job logs;
use a bucket mount if you want the adapter to survive after the job ends.

## Full run with persistent output

Create a bucket with `hf buckets create your-name/latex-ocr-runs`, then:

```bash
hf jobs uv run --flavor a100-large --timeout 4h --secrets HF_TOKEN \
  -v hf://buckets/your-name/latex-ocr-runs:/outputs \
  01-latex-ocr/train/hf_job.py --revision "$(git rev-parse HEAD)" \
  --max-steps 30 --num-generations 8 --eval-samples 50 \
  --output-dir /outputs/run-001
```

Choose a new output directory per run. Jobs' unmounted filesystem is temporary. The runner saves `config.json`,
`baseline.json`, `summary.json`, the final training checkpoint, and `adapter/` (LoRA weights plus processor).
If the job stops early, the config and completed baseline remain in the bucket; this short recipe saves its
training checkpoint at the final step.

Optional arguments:

| Argument | Purpose |
|---|---|
| `--model` | VLM model ID (default `Qwen/Qwen3-VL-2B-Instruct`) |
| `--env-url` | Use your hosted materialize-mode server instead of starting a local one |
| `--trackio-space your-name/trackio-latex-ocr` | Enable Trackio publishing; requires a write token |
| `--push-repo-id your-name/qwen3-vl-2b-latex-ocr-grpo` | Merge LoRA and publish a standalone model after evaluation |
| `--max-completion-length` | Generation cap, default 256 |

Hosted example: `--env-url https://huggingenvs-latex-ocr-env.hf.space`. Set the server's
`LATEX_OCR_MAX_SESSIONS` to at least `num_generations + 4`. Each rollout owns a session; the runner closes all
sessions on success or failure. It fails early if indexed reset is unavailable. Stream mode is for sequential
consumers and cannot supply the same indexed task to every member of a GRPO group.

The recipe uses **one CUDA GPU**, bf16, SDPA, LoRA on `q_proj`/`v_proj`, and TRL 1.12.0. It sends raw completions
to the rubric so padding cannot evade the length guard. Optional publishing is off by default.

## Published runs

The [LaTeX OCR collection](https://huggingface.co/collections/HuggingEnvs/latex-ocr-6aa7ed8498b3ffd222ad1c8c) groups the environment, published Qwen3.5 GRPO checkpoint,
seven Trackio dashboards, and nine buckets of preserved experiment data. The current Qwen3-VL-2B recipe is
separate from that historical checkpoint. The main dashboard is
[HuggingEnvs/trackio-latex-ocr](https://huggingface.co/spaces/HuggingEnvs/trackio-latex-ocr).

To launch future Jobs under the organization, add `--namespace HuggingEnvs` to `hf jobs uv run`; this requires
organization compute permissions. Use a new run name when logging to an existing dashboard.

## Local and notebook

```bash
uv run --project 01-latex-ocr/envs/latex_ocr --extra train \
  python 01-latex-ocr/train/grpo_latex_ocr.py --smoke
```

Open [`../notebooks/01_latex_ocr_grpo.ipynb`](../notebooks/01_latex_ocr_grpo.ipynb) for installation, settings,
a CPU preflight, the shared training run, and before/after predictions. The notebook exports the same frozen
lockfile into its kernel environment.
