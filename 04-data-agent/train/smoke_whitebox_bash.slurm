#!/usr/bin/env bash
#SBATCH --job-name=wb-bash-smoke
#SBATCH --partition=hopper-extra,hopper-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --output=/fsx/%u/logs/%x-%j.out
#SBATCH --error=/fsx/%u/logs/%x-%j.err
#SBATCH --time=0-02:00:00
#
# SMOKE: sync GRPO on the WHITE-BOX bash/SETA environment. Prove the loop runs end to end.
#
# ONE GPU. vLLM runs COLOCATE -- in-process on the training card -- so there is no separate serve and
# no 2-GPU allocation. `vllm_gpu_memory_utilization` is what makes that fit: the trainer and the
# engine share one card and the default 0.9 would leave nothing for the optimizer states.
#
# The env server runs on THIS node, on a port derived from the job id. Fixed ports are how one job
# ends up talking to another job's server -- that happened on the async side and would have trained
# against the wrong weights.
set -euo pipefail

TRL_PROD=/fsx/adithyaskolavi/projects/trl_prod
cd "$TRL_PROD"
PY="$TRL_PROD/.venv312/bin/python"

# THE VENV'S bin/ MUST BE ON PATH, not just its python.
# Qwen3.5 is hybrid Gated-DeltaNet, and vLLM routes its prefill through flashinfer, which JIT-COMPILES
# a kernel at first generation. That compile shells out to `ninja` and `nvcc`. Calling
# .venv312/bin/python directly (rather than activating) leaves .venv312/bin off PATH, so `ninja` is
# not found and the run dies at step 0 with a bare
#   FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
# thirty frames below flashinfer, naming nothing that suggests PATH. The backend is chosen from vLLM's
# `additional_config["gdn_prefill_backend"]`, which colocate mode does not expose, so PATH is the fix.
export PATH="$TRL_PROD/.venv312/bin:$PATH"
# nvcc for the same JIT. Already present at /usr/local/cuda-12.9/bin on these nodes; the module load
# is belt-and-braces and must not fail the job when the module system has no such module.
module load cuda/12.9 2>/dev/null || module load cuda/13.0 2>/dev/null || true
command -v ninja >/dev/null || { echo "FATAL: ninja not on PATH; flashinfer's JIT will fail at step 0"; exit 1; }
command -v nvcc  >/dev/null || echo "WARN: nvcc not on PATH; flashinfer's JIT may fail"
echo "ninja=$(command -v ninja)  nvcc=$(command -v nvcc)"

set -a; source "$TRL_PROD/experiments/.env" 2>/dev/null || true; set +a
export HF_TOKEN="${HF_TOKEN:-${HF_API_KEY:-}}"
# Presence only -- never the value.
echo "E2B_API_KEY: ${E2B_API_KEY:+present}"
[ -n "${E2B_API_KEY:-}" ] || { echo "FATAL: E2B_API_KEY unset; every episode would fail at sandbox creation"; exit 1; }

export PYTHONPATH="$TRL_PROD/trl:$TRL_PROD/HuggingEnvs/04-data-agent/train:$TRL_PROD/HuggingEnvs/04-data-agent/train/_pypath:$TRL_PROD/OpenEnv/src"
export TRL_EXPERIMENTAL_SILENCE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

PORT=$(( 8400 + ${SLURM_JOB_ID:-0} % 150 ))
export WHITE_BOX_BASH_URL="http://127.0.0.1:${PORT}"
LOGS="$TRL_PROD/HuggingEnvs/04-data-agent/envs/whitebox-bash/logs/job-${SLURM_JOB_ID:-local}"
mkdir -p "$LOGS"

echo "== env server on :$PORT =="
"$PY" -m uvicorn whitebox_bash.server.app:app --host 127.0.0.1 --port "$PORT" \
    > "$LOGS/server.log" 2>&1 &
SERVER_PID=$!
# Kill the server on any exit, including failure: a survivor holds the port and the next job's
# server cannot bind, which presents as "stuck loading forever" rather than as a port clash.
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    kill -0 $SERVER_PID 2>/dev/null || { echo "FATAL: env server died during startup"; tail -30 "$LOGS/server.log"; exit 1; }
    sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "FATAL: env server never became healthy"; tail -30 "$LOGS/server.log"; exit 1; }

# GATE: the server must actually serve tasks. A healthy server with an empty split would train on
# nothing and report clean numbers doing it.
N=$(curl -sf -X POST "http://127.0.0.1:${PORT}/white_box_bash/num_tasks" \
      -H 'content-type: application/json' -d "{\"split\":\"${SPLIT:-train}\"}" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["num_tasks"])')
echo "== server healthy, split=${SPLIT:-train} has $N tasks =="
[ "$N" -gt 0 ] || { echo "FATAL: split has no tasks"; exit 1; }

export MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.3}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"

echo "== trainer (colocate vLLM, 1 GPU) =="
"$PY" "$TRL_PROD/HuggingEnvs/04-data-agent/train/train_whitebox_bash.py" \
    --model "$MODEL" \
    --server "$WHITE_BOX_BASH_URL" \
    --split "${SPLIT:-train}" \
    --toolsets "${TOOLSETS:-bash,seta}" \
    --output-dir "$LOGS/run" \
    --max-steps "${MAX_STEPS:-4}" \
    --num-generations "${NUM_GENERATIONS:-4}" \
    --per-device-train-batch-size "${PER_DEVICE_BS:-4}" \
    --max-tool-calling-iterations "${MAX_TOOL_ITERS:-8}" \
    --step-limit "${STEP_LIMIT:-12}" \
    --max-completion-length "${MAX_COMPLETION_LENGTH:-1024}" \
    --report-to "${REPORT_TO:-none}"

echo "== smoke finished =="
