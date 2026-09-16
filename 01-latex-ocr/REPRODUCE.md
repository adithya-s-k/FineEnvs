# Reproducing LaTeX OCR experiments

This guide covers the five historical comparison runs, rebuilding their Trackio plots, and running the
maintained environment/notebook/HF Jobs pipeline. The [experiment README](./README.md) reports the outcomes;
[comparison-summary.json](./results/comparison-summary.json) contains exact values, run identities, source
hashes, and the recorded training settings for every selected run.

There are two training recipes. The historical comparison used **full fine-tuning with colocated vLLM**.
The maintained, GPU-smoke-tested recipe uses **LoRA with the migrated OpenEnv package**. Running the latter
checks the pipeline; it does not recreate the historical full-fine-tuning curves.

## 1. Recorded outcomes and source identities

| Run | Model ID | Baseline | Last evaluation | Eval step | Last logged train step |
|---|---|---:|---:|---:|---:|
| Qwen3-VL / colocated | `Qwen/Qwen3-VL-2B-Instruct` | 0.61538466 | 0.72310072 | 3,500 | 3,999 |
| Qwen3.5 / colocated | `Qwen/Qwen3.5-2B` | 0.687749705 | 0.710104455 | 3,500 | 3,999 |
| GLM / colocated | `zai-org/GLM-OCR` | 0.44944827 | 0.66441399 | 3,500 | 3,999 |
| Gemma / unstable | `google/gemma-4-E2B-it` | 0.391675955 | 0.132202275 | 1,500 | 1,999 |
| Gemma / stabilized | `google/gemma-4-E2B-it` | 0.391675955 | 0.569705410 | 3,500 | 3,999 |

The first four runs come from `AdithyaSK/latex-ocr-colocate-5k-bucket`, file
`trackio/latex-ocr-colocate-5k.db`. Stabilized Gemma comes from
`AdithyaSK/latex-ocr-colocate-5k-stable-bucket`, file `trackio/latex-ocr-colocate-5k-stable.db`.

| Run | Original Trackio run ID |
|---|---|
| Qwen3-VL | `7e9c3c987ef548c490f63d1434710abc` |
| Qwen3.5 | `613d17b2ee324f809092cd0679de8346` |
| GLM | `4d478bee2b6e4a02a7d4fb5840040f8b` |
| Gemma unstable | `caf259bcffb344f8bc2338454f4baacb` |
| Gemma stabilized | `36948599833f4825a3b4b44e8809bc3f` |

The unstable curve is intentionally included in the main comparison. Its evaluation rises briefly before
regressing; plotting only its best checkpoint would hide the outcome:

| Optimizer step | Gemma unstable | Gemma stabilized |
|---|---:|---:|
| 0 | 0.391675955 | 0.391675955 |
| 500 | 0.346191415 | 0.496046975 |
| 1,000 | 0.439515155 | 0.528709385 |
| 1,500 | 0.132202275 | 0.548734480 |
| 3,500 | — | 0.569705410 |

At the common step 1,500, the runs differ by **0.416532205** reward. Unstable Gemma's first/last 50
training records average **0.4046461 → 0.2595152**; stabilized Gemma averages **0.4092 → 0.5642**.
No value is extrapolated after unstable Gemma's last evaluation.

## 2. Historical configuration

The Trackio configs provide the settings below. The recovered launchers additionally specify full tuning,
5,000 training samples, 200 evaluation samples, and the cluster resource request.

| Setting | Qwen3-VL / Qwen3.5 / GLM | Gemma unstable | Gemma stabilized |
|---|---|---|---|
| Optimizer | `adamw_torch` | `adafactor` | `adafactor` |
| Learning rate | `1e-5` | `1e-5` | `5e-6` |
| KL coefficient `beta` | `0` | `0` | `0.03` |
| Temperature | `0.9` | `0.9` | `0.8` |
| Maximum gradient norm | `1.0` | `1.0` | `0.5` |
| vLLM GPU memory fraction | `0.30` | `0.30` | `0.18` |

Shared recorded settings:

- 5,000 configured optimizer steps; seed 42; batch size 8 per device; gradient accumulation 1;
  8 generations and 1 step per generation.
- `loss_type=dapo`, `scale_rewards=group`, `epsilon=0.2`, linear learning-rate schedule, no warmup,
  zero weight decay, bf16, gradient checkpointing with `use_reentrant=False`.
- Maximum completion length 512; `top_p=1`, `top_k=0`; **`mask_truncated_completions=False`**.
- vLLM `colocate`, tensor parallelism 1, context cap 16,384;
  importance-sampling correction `sequence_mask` with clip maximum 3.0.
- Save every 500 steps; scalar logging every step. The serialized model configs report
  `transformers_version=5.14.1`; that field is not a complete runtime package freeze.

The recovered launchers request one GPU and 24 CPUs per model on `hopper-prod`, with a 24-hour Slurm
limit and CUDA 13.0 module. The partition name alone does not establish the exact GPU SKU/memory or cost.
The original array indexes are Qwen3.5=0, Qwen3-VL=1, Gemma=2, GLM=3.

The recovered script uses `unsloth/LaTeX_OCR`, materializes the first 5,000 training rows, resizes images
to fit within 512×512 while preserving aspect ratio, and runs greedy evaluation on the first 200 test rows
at baseline and every 500 steps. Training and evaluation send raw completions to the environment's rubric.
The normalized edit/exact-match reward is length-guarded; it is not a rendering or mathematical-equivalence score.
The script disables thinking in Qwen chat templates and uses the instruction to output only LaTeX.

These are the **documented launcher protocol and recorded configs**. The Trackio scalar records do not store
the actual evaluation sample IDs/counts or scorer revision. The recovered script was committed after the
metric snapshots, so it cannot independently certify the protocol used for each historical evaluation.

Gemma's five settings changed together. The comparison does not isolate the effect of KL or learning rate.
Final 50-record mean completion clipping is 0.9975 for unstable Gemma, and 1.0 for stabilized Gemma and GLM.
Truncation was unmasked, so reward improvement should be read alongside those diagnostics.

## 3. Recovered training source and replay commands

The [historical source](./train/historical/) is included in this repository, with hashes and the recovered
dependency revisions in [reproduction-sources.json](./results/reproduction-sources.json).
It is from research commit `4d82cafc0090d480d856a46bdcca0865434ee932`, dated 2026-07-26.
That commit pins TRL `3389ccb27bf9621db31ba3496f006b86ed3bfaaa` and OpenEnv
`238b36e082dbefb829f3f6a446dae7ef09df1853` in their respective author forks.

Prepare the original relative directory layout, starting in this repository:

```bash
LATEX_REPLAY_ROOT="$(mktemp -d /tmp/latex-ocr-replay.XXXXXX)"
mkdir -p "$LATEX_REPLAY_ROOT/experiments/vlm_ocr/ablation"
cp 01-latex-ocr/train/historical/train_ablation.py \
  "$LATEX_REPLAY_ROOT/experiments/vlm_ocr/ablation/"
git clone https://github.com/adithya-s-k/trl.git "$LATEX_REPLAY_ROOT/trl"
git -C "$LATEX_REPLAY_ROOT/trl" checkout 3389ccb27bf9621db31ba3496f006b86ed3bfaaa
git clone https://github.com/adithya-s-k/OpenEnv.git "$LATEX_REPLAY_ROOT/OpenEnv"
git -C "$LATEX_REPLAY_ROOT/OpenEnv" checkout 238b36e082dbefb829f3f6a446dae7ef09df1853
cd "$LATEX_REPLAY_ROOT"
export PYTHONPATH="$LATEX_REPLAY_ROOT/trl:$LATEX_REPLAY_ROOT/OpenEnv/src:$LATEX_REPLAY_ROOT/OpenEnv/envs"
```

Use a CUDA Python environment compatible with those source revisions, including torch, torchvision,
transformers, vLLM, datasets, peft, Pillow, requests, uvicorn, websockets, and the legacy OpenEnv dependencies.
The original launchers reused `.venv312` and installed missing packages without pins. Its historical package
freeze was not recovered, so there is no honest fully locked install command for the old experiment.
The migrated, frozen recipe in section 5 is the validated installation path for new pipeline runs.
Gemma access also requires accepting its model terms on Hugging Face and authenticating your account.

After preparing that compatible environment, this is the recorded unstable Gemma invocation with a fresh
output directory and remote Trackio publishing disabled:

```bash
python experiments/vlm_ocr/ablation/train_ablation.py \
  --model google/gemma-4-E2B-it --run-name gemma4-e2b-unstable-replay \
  --mode env --tune full --optim adafactor --vllm colocate \
  --learning-rate 1e-5 --temperature 0.9 --beta 0 --max-grad-norm 1.0 \
  --vllm-gpu-mem 0.30 --vllm-max-len 16384 \
  --env-port 8092 --output-dir runs/gemma-unstable-replay \
  --max-steps 5000 --num-train-samples 5000 --num-generations 8 \
  --eval-samples 200 --eval-interval 500 --save-steps 500 \
  --max-completion-length 512 --mask-truncated false
```

For stabilized Gemma, change the run/output names and exactly these options:

```text
--learning-rate 5e-6 --temperature 0.8 --beta 0.03
--max-grad-norm 0.5 --vllm-gpu-mem 0.18
```

For Qwen3-VL, Qwen3.5, and GLM, use their model IDs from section 1, `--optim adamw_torch`, and the unstable
column's shared settings. Use separate ports/output paths for concurrent jobs. A short compatibility check
can override `--max-steps 2 --num-train-samples 8 --eval-samples 2 --eval-interval 1 --save-steps 2` before
committing to a full run; that is a new smoke protocol and is not evidence of the historical gains.
To log a new replay remotely, add `--trackio-space YOUR_NAME/YOUR_NEW_SPACE --trackio-project latex-ocr-replay`.

The harness saves `results.json`, `train_log.jsonl`, `eval_log.jsonl`, periodic trainer checkpoints, and final
`completions.json`. Preserve these with the resolved model/dataset commit IDs, package freeze, launcher
arguments, source hashes, accelerator details, and job ID. The old model/dataset IDs were not revision-pinned,
so using the current Hub heads cannot guarantee identical weights or sample ordering.

The recovered source closes evaluation clients and raises local session capacity to 64. Its comment describes
an earlier evaluation-session leak that reached capacity around step 4,000. This is consistent with four
3,999-step snapshots, but without their original job logs it is not independent proof of each termination cause.
The saved metrics do not establish why unstable Gemma stopped at 1,999. No full historical training rerun was
performed for this documentation update.

## 4. Rebuild the combined Trackio view

This reconstruction needs CPU only. Use a checkout containing this guide and record `git rev-parse HEAD`.
From the repository root, download the 11 source databases and verify their manifest hashes:

```bash
uv run --with huggingface_hub==1.31.0 python - <<'PY'
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi

root = Path('/tmp/latex-ocr-archives')
api = HfApi()
manifest = json.loads(Path('01-latex-ocr/dashboard/archive-sources.json').read_text())
for item in manifest:
    destination = root / item['bucket'].split('/', 1)[1] / Path(item['path']).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    api.download_bucket_files(item['bucket'], [(item['path'], destination)], raise_on_missing_files=True)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == item['sha256'], destination
print('All source hashes match')
PY
uv run python 01-latex-ocr/dashboard/build_comparison.py \
  --source-dir /tmp/latex-ocr-archives --output-dir /tmp/latex-ocr-combined
uv run --with-requirements 01-latex-ocr/dashboard/requirements.txt --with pytest \
  python -m pytest 01-latex-ocr/dashboard/ -q
```

Expected outputs:

- `latex-ocr-comparison.db`: **five runs**, four model families, **17,995 training records**,
  **36 evaluation points** (18,000 merged rows, including five baselines).
- `latex-ocr-history.db`: **12 other run segments**, 6,269 training records. Unstable Gemma is promoted
  into the main comparison, preserving its deterministic run ID and raw values.
- `comparison-summary.json`: complete source provenance, configs, exact evaluation points, window summaries,
  and excluded preflights/restart fragments.

The builder opens sources read-only, aligns scalar values by optimizer step, deduplicates identical baselines,
and rejects conflicting values. It orders original records by internal log step/id because upload timestamps
can arrive out of order. Optimizer-step resets split reused run IDs; ambiguous evaluation/attempt mappings
are omitted rather than guessed. Historical segments with fewer than 100 training records stay in the archives.

Preview the two derived projects locally:

```bash
TRACKIO_DIR=/tmp/latex-ocr-combined \
  uv run --with-requirements 01-latex-ocr/dashboard/requirements.txt \
  python 01-latex-ocr/dashboard/app.py
```

The production Space additionally retains `latex-ocr-2b-overnight`, `latex-ocr-eval`, and their media in
`HuggingEnvs/latex-ocr-results`. Upload only the two derived databases under `trackio/` and the summary under
`reports/`; preserve the original databases and media. Deploy the dashboard's `app.py`, `requirements.txt`,
and README to [HuggingEnvs/trackio-latex-ocr](https://huggingface.co/spaces/HuggingEnvs/trackio-latex-ocr).
Keep the existing bucket mounted at `/data`, `TRACKIO_DIR=/data/trackio`, and
`TRACKIO_BUCKET_ID=HuggingEnvs/latex-ocr-results`. Trackio is pinned to 0.37.1.

The overnight Qwen3.5 evaluation is a separate protocol: 1,000 examples per checkpoint, reward
0.744633157 → 0.829776173 and exact match 20.0% → 42.7% at checkpoint 3,000. Those points are not mixed
with the five-run comparison, and the archive does not prove that the published model weights equal that checkpoint.

## 5. Reproduce the maintained pipeline and HF Jobs smoke

Return to the HuggingEnvs repository root. The current package uses its frozen `uv.lock`, OpenEnv 0.4.2,
and TRL 1.12.0. The notebook and Jobs share `train/grpo_latex_ocr.py`.

```bash
uv run --frozen --project 01-latex-ocr/envs/latex_ocr --extra dev \
  pytest 01-latex-ocr/envs/latex_ocr/tests -q
uv run --frozen --project 01-latex-ocr/envs/latex_ocr latex-ocr-smoke
uv run --frozen --project 01-latex-ocr/envs/latex_ocr latex-ocr-smoke --real-data
uv run --frozen --project 01-latex-ocr/envs/latex_ocr latex-ocr-smoke \
  --url https://huggingenvs-latex-ocr-env.hf.space
```

On one local CUDA GPU:

```bash
uv run --frozen --project 01-latex-ocr/envs/latex_ocr --extra train \
  python 01-latex-ocr/train/grpo_latex_ocr.py --smoke
```

On HF Jobs, authenticate with a compute-enabled account and use a pushed full commit SHA:

```bash
hf auth login
hf jobs uv run --flavor cpu-basic --timeout 15m \
  01-latex-ocr/train/hf_job.py --revision "$(git rev-parse HEAD)" --mode env-smoke
hf jobs uv run --flavor a10g-small --timeout 30m --secrets HF_TOKEN \
  01-latex-ocr/train/hf_job.py --revision "$(git rev-parse HEAD)" --smoke
```

The GPU smoke uses Qwen3-VL-2B, two steps, two generations, and two held-out images. It verifies finite loss
and an adapter weight update. It does not require a reward increase. The previously completed jobs used
source revisions `f64d444c531ddedb170ee96e62551c58bb75e5b9` (CPU) and
`e954ec2164899366d7467938b1ac19b07f650ab3` (GPU); use those SHA values to select their original code.
Their IDs, outcomes, predictions, and runtime evidence are in [`results/`](./results/).

For persistent output, first create your own bucket, then submit a new run:

```bash
hf buckets create YOUR_NAME/latex-ocr-runs
hf jobs uv run --flavor a100-large --timeout 4h --secrets HF_TOKEN \
  -v hf://buckets/YOUR_NAME/latex-ocr-runs:/outputs \
  01-latex-ocr/train/hf_job.py --revision "$(git rev-parse HEAD)" \
  --max-steps 30 --num-generations 8 --eval-samples 50 \
  --output-dir /outputs/run-001
```

This maintained recipe uses LoRA rank 16, alpha 32, dropout 0.05 on `q_proj`/`v_proj`; bf16/SDPA;
learning rate 1e-5; temperature 0.9; seed 42; image cap 512; completion cap 256; and truncation masking enabled.
It evaluates the same indexed test images before and after training and saves config, baseline, summary,
checkpoint, adapter, and processor. A hosted environment must support indexed materialize mode and at least
`num_generations + 4` sessions. The default runner starts and cleans up its own local server.
The historical full-tuning Gemma/GLM recipes have not been GPU-validated on this maintained stack.
