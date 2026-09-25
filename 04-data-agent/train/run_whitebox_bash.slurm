#!/usr/bin/env bash
#SBATCH --job-name=wb-bash-run
#SBATCH --partition=hopper-extra,hopper-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --output=/fsx/%u/logs/%x-%j.out
#SBATCH --error=/fsx/%u/logs/%x-%j.err
#SBATCH --time=0-04:00:00
#
# A REAL (small) run: one pass over the 100-task generated train split, checkpoints saved, so the
# result can be evaluated against base on the DISJOINT 30-task test split.
#
# SPLIT DEFAULTS TO train:easy, AND THAT IS A GRADIENT DECISION, NOT A DIFFICULTY PREFERENCE.
# GRPO needs variance WITHIN a group: if every rollout of a prompt scores the same, the advantages are
# all zero and the step contributes nothing. Base pass@1 for this model by tier is easy 0.214,
# medium 0.062, hard 0.060, so P(all `num_generations` rollouts fail) is
#     train:easy    g=4 -> 0.38   g=8 -> 0.15   g=16 -> 0.02
#     train:medium  g=4 -> 0.77   g=8 -> 0.60   g=16 -> 0.36
# Measured: `train` at g=4 produced 0 reward and grad_norm 0 on 7 of 7 steps (job 76712). easy at
# g=8 is the smallest configuration that actually learns on one GPU.
#
# Same 1-GPU colocate shape as the smoke. Everything here that looks incidental is load-bearing --
# see smoke_whitebox_bash.slurm for why PATH, the ninja gate and the derived ports exist.
set -euo pipefail

TRL_PROD=/fsx/adithyaskolavi/projects/trl_prod
cd "$TRL_PROD"
PY="$TRL_PROD/.venv312/bin/python"
export PATH="$TRL_PROD/.venv312/bin:$PATH"
module load cuda/12.9 2>/dev/null || module load cuda/13.0 2>/dev/null || true
command -v ninja >/dev/null || { echo "FATAL: ninja not on PATH; flashinfer's JIT dies at step 0"; exit 1; }

set -a; source "$TRL_PROD/experiments/.env" 2>/dev/null || true; set +a
export HF_TOKEN="${HF_TOKEN:-${HF_API_KEY:-}}"
echo "E2B_API_KEY: ${E2B_API_KEY:+present}"
[ -n "${E2B_API_KEY:-}" ] || { echo "FATAL: E2B_API_KEY unset"; exit 1; }

export PYTHONPATH="$TRL_PROD/trl:$TRL_PROD/HuggingEnvs/04-data-agent/train:$TRL_PROD/HuggingEnvs/04-data-agent/train/_pypath:$TRL_PROD/OpenEnv/src"
export TRL_EXPERIMENTAL_SILENCE=1 HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PORT=$(( 8400 + ${SLURM_JOB_ID:-0} % 150 ))
export WHITE_BOX_BASH_URL="http://127.0.0.1:${PORT}"
LOGS="$TRL_PROD/HuggingEnvs/04-data-agent/envs/whitebox-bash/logs/job-${SLURM_JOB_ID:-local}"
mkdir -p "$LOGS"

echo "== env server on :$PORT =="
"$PY" -m uvicorn whitebox_bash.server.app:app --host 127.0.0.1 --port "$PORT" > "$LOGS/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "FATAL: env server died"; tail -30 "$LOGS/server.log"; exit 1; }
    sleep 2
done
SPLIT="${SPLIT:-train:easy}"
N=$(curl -sf -X POST "http://127.0.0.1:${PORT}/white_box_bash/num_tasks" -H 'content-type: application/json' \
      -d "{\"split\":\"$SPLIT\"}" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["num_tasks"])')
NT=$(curl -sf -X POST "http://127.0.0.1:${PORT}/white_box_bash/num_tasks" -H 'content-type: application/json' \
      -d '{"split":"test"}' | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["num_tasks"])')
echo "== server healthy: $SPLIT=$N test=$NT =="
[ "$N" -ge 100 ] || { echo "FATAL: split $SPLIT has only $N tasks"; exit 1; }
# Tasks come from HuggingEnvs/data-agent, staged per task out of an HF bucket into the shared
# E2B template. Both are required: the default E2B base has no huggingface_hub and no writable
# /workdir, so staging dies before the agent starts.
echo "== task source: ${WHITE_BOX_BASH_TASK_SOURCE:-data-agent}  template: ${E2B_TEMPLATE:-data-agent-opencode} =="

export MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.3}"

echo "== trainer: ${MAX_STEPS:-100} steps over $N tasks, saving every ${SAVE_STEPS:-50} =="
"$PY" "$TRL_PROD/HuggingEnvs/04-data-agent/train/train_whitebox_bash.py" \
    --model "$MODEL" --server "$WHITE_BOX_BASH_URL" --split "$SPLIT" \
    --toolsets "${TOOLSETS:-bash,seta}" --output-dir "$LOGS/run" \
    --max-steps "${MAX_STEPS:-100}" --save-steps "${SAVE_STEPS:-50}" \
    --num-generations "${NUM_GENERATIONS:-8}" --per-device-train-batch-size "${PER_DEVICE_BS:-2}" \
    --gradient-accumulation-steps "${GRAD_ACCUM:-4}" \
    --max-tool-calling-iterations "${MAX_TOOL_ITERS:-8}" --step-limit "${STEP_LIMIT:-12}" \
    --max-completion-length "${MAX_COMPLETION_LENGTH:-1024}" --report-to "${REPORT_TO:-none}"

echo "== run finished; checkpoints: =="
ls -d "$LOGS/run"/checkpoint-* 2>/dev/null || echo "  (none)"
