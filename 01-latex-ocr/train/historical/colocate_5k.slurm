#!/usr/bin/env bash
#SBATCH --job-name=latex-colo-5k
#SBATCH --partition=hopper-prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --array=0-3
#SBATCH --output=/fsx/%u/logs/%x-%A_%a.out
#SBATCH --error=/fsx/%u/logs/%x-%A_%a.err
#SBATCH --time=1-00:00:00

# COLOCATE 5k ablation: 4 env-mode models, full-FT, vLLM colocate generation, 5000 steps,
# eval every 500 steps (200 examples) + checkpoint every 500. New length-guarded reward
# (local env). Logs LOCALLY (JSON, crash-safe) AND to Trackio (metrics only — media off).
module load cuda/13.0 || true
set -x
TRL_PROD=/fsx/adithyaskolavi/projects/trl_prod
cd "$TRL_PROD"; export VENV="${VENV:-.venv312}"; source "$VENV/bin/activate"
export PYTHONPATH="$TRL_PROD/trl:$TRL_PROD/OpenEnv/src:$TRL_PROD/OpenEnv/envs:${PYTHONPATH:-}"
export UV_LINK_MODE=copy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_AOT_COMPILE=0 HF_HUB_DISABLE_PROGRESS_BARS=1
export TRACKIO_STORAGE_MODE=sqlite
{ set +x; } 2>/dev/null
set -a; source "$TRL_PROD/experiments/.env" 2>/dev/null || true; set +a
export HF_TOKEN="${HF_TOKEN:-$HF_API_KEY}"
set -x
python -c "import vllm, uvicorn, websockets, requests, fastmcp, peft, gradio, trackio" 2>/dev/null \
    || uv pip install vllm uvicorn websockets requests fastmcp peft gradio trackio

MODELS=("Qwen/Qwen3.5-2B" "Qwen/Qwen3-VL-2B-Instruct" "google/gemma-4-E2B-it" "zai-org/GLM-OCR")
RUNS=("qwen3.5-2b" "qwen3-vl-2b" "gemma4-e2b" "glm-ocr")
OPTIMS=("adamw_torch" "adamw_torch" "adafactor" "adamw_torch")

i="$SLURM_ARRAY_TASK_ID"
MODEL="${MODELS[$i]}"; RUN="${RUNS[$i]}"; OPTIM="${OPTIMS[$i]}"; PORT=$((8090 + i))
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((24000 + (SLURM_JOB_ID % 1500) * 10 + i))

echo "[$(date)] COLO-5k task $i: $MODEL full-FT vLLM-colocate (optim=$OPTIM, port $PORT)"

python experiments/vlm_ocr/ablation/train_ablation.py \
    --model "$MODEL" \
    --run-name "$RUN" \
    --mode env \
    --tune full \
    --optim "$OPTIM" \
    --vllm colocate \
    --vllm-gpu-mem 0.3 \
    --output-dir "experiments/vlm_ocr/ablation/runs/colocate-5k/$RUN" \
    --env-port "$PORT" \
    --max-steps 5000 \
    --num-train-samples 5000 \
    --num-generations 8 \
    --eval-samples 200 \
    --eval-interval 500 \
    --save-steps 500 \
    --max-completion-length 512 \
    --mask-truncated false \
    --trackio-space "AdithyaSK/latex-ocr-colocate-5k" \
    --trackio-project "latex-ocr-colocate-5k"

echo "[$(date)] COLO-5k task $i ($RUN) done"
