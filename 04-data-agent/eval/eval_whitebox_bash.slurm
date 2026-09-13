#!/usr/bin/env bash
#SBATCH --job-name=wb-bash-eval
#SBATCH --partition=hopper-extra,hopper-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --output=/fsx/%u/logs/%x-%j.out
#SBATCH --error=/fsx/%u/logs/%x-%j.err
#SBATCH --time=0-03:00:00
#
# PAIRED base-vs-checkpoint eval for the white-box env. Both models are served CONCURRENTLY on ONE
# GPU (they are 2B, ~5 GB each at 0.4 utilisation) so the two arms see the same sandbox conditions AND
# the same hardware -- running them hours apart would let a change in E2B health masquerade as a
# change in the policy, which is exactly how two q3i evals produced a "collapse" that was 976
# sandbox 404s. One GPU also stays inside the QOSMaxGRESPerJob limit that refused a 2-GPU request.
set -euo pipefail

TRL_PROD=/fsx/adithyaskolavi/projects/trl_prod
cd "$TRL_PROD"
PY="$TRL_PROD/.venv312/bin/python"
export PATH="$TRL_PROD/.venv312/bin:$PATH"
module load cuda/12.9 2>/dev/null || module load cuda/13.0 2>/dev/null || true

set -a; source "$TRL_PROD/experiments/.env" 2>/dev/null || true; set +a
export HF_TOKEN="${HF_TOKEN:-${HF_API_KEY:-}}"
echo "E2B_API_KEY: ${E2B_API_KEY:+present}"
[ -n "${E2B_API_KEY:-}" ] || { echo "FATAL: E2B_API_KEY unset"; exit 1; }

export PYTHONPATH="$TRL_PROD/trl:$TRL_PROD/HuggingEnvs/04-data-agent/train/_pypath:$TRL_PROD/OpenEnv/src"
export TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=1

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-2B}"
CKPT="${CKPT:?set CKPT to a checkpoint directory}"
OFF=$(( ${SLURM_JOB_ID:-0} % 120 ))
ENV_PORT=$(( 8500 + OFF )); BASE_PORT=$(( 8700 + OFF )); CKPT_PORT=$(( 8900 + OFF ))
LOGS="$TRL_PROD/HuggingEnvs/04-data-agent/envs/whitebox-bash/logs/eval-${SLURM_JOB_ID:-local}"
mkdir -p "$LOGS"

# --max-num-seqs MUST be set for Qwen3.5 when two engines share a card.
# It is hybrid Gated-DeltaNet, so EVERY concurrent decode sequence needs its own Mamba cache block.
# At 0.40 utilisation there is room for ~417, but vLLM defaults max_num_seqs=1024 and refuses to
# start:  ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (417).
# The eval runs 8 concurrent episodes, so 64 is generous. The failure is confusing because the FIRST
# engine starts fine and only the second dies -- it reads as a bad checkpoint, not a memory setting.
serve () {  # $1=gpu $2=model $3=port $4=logname   (both share GPU 0)
    # qwen3_xml is REQUIRED: with the wrong parser vLLM returns no structured tool_calls, every
    # episode makes zero tool calls, and the eval reports a model that cannot use tools.
    CUDA_VISIBLE_DEVICES=$1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
    "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$2" --served-model-name "$2" --port "$3" --trust-remote-code \
        --max-model-len "${MAX_MODEL_LEN:-16384}" --gpu-memory-utilization "${VLLM_UTIL:-0.40}" \
        --max-num-seqs "${MAX_NUM_SEQS:-64}" \
        --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
        --gdn-prefill-backend triton \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        > "$LOGS/$4.log" 2>&1 &
}

echo "== env server :$ENV_PORT =="
"$PY" -m uvicorn whitebox_bash.server.app:app --host 127.0.0.1 --port "$ENV_PORT" > "$LOGS/env.log" 2>&1 &
ENVPID=$!
echo "== vLLM base :$BASE_PORT and ckpt :$CKPT_PORT, both on GPU 0 =="
serve 0 "$BASE_MODEL" "$BASE_PORT" vllm-base
serve 0 "$CKPT"       "$CKPT_PORT" vllm-ckpt
trap 'kill $ENVPID 2>/dev/null || true; pkill -P $$ 2>/dev/null || true' EXIT

# vLLM's /health returns 200 with an EMPTY body -- check the STATUS, parse nothing.
for url in "http://127.0.0.1:$ENV_PORT/health" "http://127.0.0.1:$BASE_PORT/health" "http://127.0.0.1:$CKPT_PORT/health"; do
    for _ in $(seq 1 180); do curl -sf -o /dev/null "$url" && break; sleep 5; done
    curl -sf -o /dev/null "$url" || { echo "FATAL: never healthy: $url"; tail -20 "$LOGS"/*.log; exit 1; }
    echo "  healthy: $url"
done

"$PY" "$TRL_PROD/HuggingEnvs/04-data-agent/eval/eval_whitebox_bash.py" \
    --server "http://127.0.0.1:$ENV_PORT" \
    --base-url "http://127.0.0.1:$BASE_PORT" --base-model "$BASE_MODEL" \
    --ckpt-url "http://127.0.0.1:$CKPT_PORT" --ckpt-model "$CKPT" \
    --split "${SPLIT:-test}" --n-tasks "${N_TASKS:-60}" --k "${EVAL_K:-4}" \
    --concurrency "${CONCURRENCY:-8}" --max-turns "${MAX_TURNS:-8}" \
    --out "$LOGS/result.json"
echo "== eval finished =="
