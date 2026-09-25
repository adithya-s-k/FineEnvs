#!/bin/bash
# Core launcher: vLLM OpenAI-compatible server + cloudflared tunnel.
# Invoked from per-model .slurm files that pre-set environment variables.
#
# Required env vars:
#   MODEL            HF model id (e.g. Qwen/Qwen3-4B)
#   TP_SIZE          tensor-parallel size (per replica)
#   DP_SIZE          data-parallel size (replicas, vLLM-managed)
#   MAX_MODEL_LEN    max prompt+output tokens
#   SHORT_NAME       short slug for log/url filenames (e.g. qwen3-4b)
#
# Optional env vars:
#   PORT                       default 8000
#   GPU_MEMORY_UTILIZATION     default 0.92
#   TOOL_CALL_PARSER           default hermes
#   REASONING_PARSER           default qwen3 (empty disables)
#   EXTRA_VLLM_ARGS            extra args appended verbatim
#   READY_TIMEOUT_SEC          default 1800 (30 min, larger models need longer)
#   TRL_PROD                   default current working directory
#
# Outputs:
#   VLLM_URL_FILE (default: $TRL_PROD/temp/vllm-url-<job>.txt)
#   VLLM_LOG (default: $TRL_PROD/temp/vllm-server-<job>.log)

set -e

: "${MODEL:?MODEL is required}"
: "${TP_SIZE:?TP_SIZE is required}"
: "${DP_SIZE:?DP_SIZE is required}"
: "${MAX_MODEL_LEN:?MAX_MODEL_LEN is required}"
: "${SHORT_NAME:?SHORT_NAME is required}"

# Derive a per-job port so co-located jobs (this cluster does not allocate
# nodes exclusively) don't all collide on 8000. Range 8000-8999.
PORT="${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
# Use `:-` only when var is UNSET (not when explicitly empty), so per-model
# slurm scripts can opt out of a reasoning parser with REASONING_PARSER="".
REASONING_PARSER="${REASONING_PARSER-qwen3}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1800}"
TRL_PROD="${TRL_PROD:-$PWD}"
# gradio by default: it needs no binary download and no ingress, and these endpoints are reached
# from off-cluster (evals, sandboxed agents). TUNNEL=cloudflared keeps the old path; TUNNEL=none
# serves locally only.
TUNNEL="${TUNNEL:-gradio}"
CLOUDFLARED="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"

cd "$TRL_PROD"
# shellcheck disable=SC1091
source "${VENV:-.venv312}/bin/activate"   # .venv312 = the current (cuda-13/vllm-0.25) env
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-0}"   # avoids vLLM distributed-startup errors (trl-internal #206)

# The API-server process waits VLLM_ENGINE_READY_TIMEOUT_S (default 600s) for
# the engine cores to come up. With full CUDA-graph capture at large context
# on a CPU-contended node, engine init can exceed 600s and the API server
# bails out *after* the engines were nearly ready. Give it the same generous
# budget we use for /health polling.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-$READY_TIMEOUT_SEC}"

NODE_HOSTNAME=$(hostname)
NODE_IP=$(hostname -I | awk '{print $1}')

TOTAL_GPUS=$((TP_SIZE * DP_SIZE))

echo "============================================================"
echo "vLLM + Cloudflare tunnel"
echo "  Model:           $MODEL"
echo "  Short name:      $SHORT_NAME"
echo "  Node:            $NODE_HOSTNAME ($NODE_IP)"
echo "  Port:            $PORT"
echo "  TP size:         $TP_SIZE"
echo "  DP size:         $DP_SIZE"
echo "  Total GPUs:      $TOTAL_GPUS"
echo "  Max model len:   $MAX_MODEL_LEN"
echo "  GPU mem util:    $GPU_MEMORY_UTILIZATION"
echo "  Tool parser:     $TOOL_CALL_PARSER"
echo "  Reasoning parser:$REASONING_PARSER"
echo "  Extra args:      $EXTRA_VLLM_ARGS"
echo "  vllm bin:        $(command -v vllm)"
echo "  python:          $(which python)"
echo "  START TIME:      $(date)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Install cloudflared if missing (skipped unless TUNNEL=cloudflared)
if [ "$TUNNEL" = cloudflared ] && [ ! -f "$CLOUDFLARED" ]; then
    echo ">>> Installing cloudflared to $CLOUDFLARED ..."
    mkdir -p "$(dirname "$CLOUDFLARED")"
    curl -sSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -o "$CLOUDFLARED"
    chmod +x "$CLOUDFLARED"
fi

VLLM_LOG="${VLLM_LOG:-$TRL_PROD/temp/vllm-server-${SLURM_JOB_ID:-$$}.log}"
mkdir -p "$(dirname "$VLLM_LOG")"
: > "$VLLM_LOG"
echo ">>> vLLM log: $VLLM_LOG"

# Build vllm args
VLLM_ARGS=(
    "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --trust-remote-code
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
)
if [ "$DP_SIZE" -gt 1 ]; then
    VLLM_ARGS+=(--data-parallel-size "$DP_SIZE")
fi
if [ -n "$REASONING_PARSER" ]; then
    VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
# ENFORCE_EAGER=1 skips torch.compile + CUDA-graph capture. Slower decode but
# much faster, crash-free startup (the Inductor autotune path can fault under
# memory pressure on shared nodes). Recommended for the large TP>=2 models.
if [ "${ENFORCE_EAGER:-0}" = "1" ]; then
    VLLM_ARGS+=(--enforce-eager)
fi
# ENABLE_THINKING=0 sets a SERVER-SIDE default of enable_thinking=false for the
# chat template, so every client (incl. opencode, which makes its own API calls
# and can't send per-request kwargs) gets thinking off. vLLM merges this with
# request-level chat_template_kwargs (request wins). NOTE the flag is
# --default-chat-template-kwargs (vLLM 0.18); plain --chat-template-kwargs does
# NOT exist and errors with "unrecognized arguments".
if [ "${ENABLE_THINKING:-1}" = "0" ]; then
    VLLM_ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
fi
# shellcheck disable=SC2206
EXTRA_ARR=($EXTRA_VLLM_ARGS)
VLLM_ARGS+=("${EXTRA_ARR[@]}")

echo ">>> Starting vLLM:"
echo "    vllm serve ${VLLM_ARGS[*]}"
vllm serve "${VLLM_ARGS[@]}" >> "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Stream the vllm log into job stdout so failures are immediately visible.
tail -f -n +1 --pid=$$ "$VLLM_LOG" &
TAIL_PID=$!

# Wait for /health
echo ">>> Waiting up to ${READY_TIMEOUT_SEC}s for vLLM /health ..."
READY=0
for i in $(seq 1 "$READY_TIMEOUT_SEC"); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo ">>> vLLM ready after ${i}s"
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited prematurely. Tail of log:"
        tail -80 "$VLLM_LOG"
        exit 1
    fi
    sleep 1
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: vLLM did not become ready within ${READY_TIMEOUT_SEC}s. Tail:"
    tail -80 "$VLLM_LOG"
    kill "$VLLM_PID" 2>/dev/null || true
    exit 1
fi

echo ">>> /v1/models:"
curl -s "http://localhost:$PORT/v1/models" | python -m json.tool || true

# Start the tunnel
TUNNEL_LOG="/tmp/tunnel_${SHORT_NAME}_${SLURM_JOB_ID:-$$}.log"
TUNNEL_URL=""
TUNNEL_PID=""
echo ""
echo "============================================================"
echo ">>> Starting tunnel: $TUNNEL (log: $TUNNEL_LOG)"
echo "============================================================"
case "$TUNNEL" in
  gradio)
    # The helper only prints TUNNEL_URL= after proving a request THROUGH the tunnel reaches this
    # server, so a URL here is a working endpoint rather than merely a resolving hostname.
    python "$(dirname "${BASH_SOURCE[0]}")/../gradio_tunnel.py" \
        --port "$PORT" --verify-path /health > "$TUNNEL_LOG" 2>>"$TUNNEL_LOG" &
    TUNNEL_PID=$!
    for i in $(seq 1 60); do
        TUNNEL_URL=$(grep -o 'TUNNEL_URL=https://[^ ]*' "$TUNNEL_LOG" 2>/dev/null | head -1 | cut -d= -f2-)
        [ -n "$TUNNEL_URL" ] && break
        if grep -q "TUNNEL_ERROR=" "$TUNNEL_LOG" 2>/dev/null; then
            echo "ERROR: gradio tunnel did not reach the server:"; cat "$TUNNEL_LOG"; break
        fi
        sleep 5
    done
    ;;
  cloudflared)
    "$CLOUDFLARED" tunnel --url "http://localhost:$PORT" 2>&1 | tee "$TUNNEL_LOG" &
    TUNNEL_PID=$!
    for i in $(seq 1 60); do
        TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1)
        [ -n "$TUNNEL_URL" ] && break
        sleep 1
    done
    ;;
  none)
    echo ">>> TUNNEL=none, serving locally only"
    ;;
esac

URL_FILE="${VLLM_URL_FILE:-$TRL_PROD/temp/vllm-url-${SLURM_JOB_ID:-$$}.txt}"
mkdir -p "$(dirname "$URL_FILE")"
echo "$TUNNEL_URL" > "$URL_FILE"

echo ""
echo "============================================================"
echo ">>> TUNNEL URL: ${TUNNEL_URL:-<not-yet-available>}"
echo ""
echo "  OpenAI endpoint: ${TUNNEL_URL}/v1"
echo "  Models:          ${TUNNEL_URL}/v1/models"
echo "  Health:          ${TUNNEL_URL}/health"
echo "  Local:           http://${NODE_HOSTNAME}:${PORT}/v1"
echo "  URL file:        $URL_FILE"
echo "============================================================"
echo ""

# Quick self-test against the tunnel (optional, ignore failures)
if [ -n "$TUNNEL_URL" ]; then
    echo ">>> Self-test (chat completion via tunnel) ..."
    curl -sS --max-time 60 -X POST "${TUNNEL_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word PONG.\"}],\"max_tokens\":16,\"temperature\":0}" \
        | python -m json.tool 2>/dev/null | head -40 || echo "  (self-test failed or no JSON returned — server may still be warming up)"
fi

# Keep running until vllm exits (or job time elapses).
wait "$VLLM_PID"
