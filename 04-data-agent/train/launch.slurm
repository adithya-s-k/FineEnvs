#!/bin/bash
#SBATCH --job-name=da-blackbox-2b
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --partition=hopper-extra
#SBATCH --output=/fsx/%u/logs/%x-%j.out
#SBATCH --error=/fsx/%u/logs/%x-%j.err
#SBATCH --time=0-12:00:00
set -euo pipefail

# One job, three processes, in this order and for these reasons:
#
#   GPU 0  vLLM, with the capture flags. The TRAINER's engine -- the agent calls the same weights the
#          optimizer updates, which is what makes the rollouts on-policy.
#   GPU 1  the trainer.
#   cpu    the env server, which starts the capture proxy and publishes it.
#
# The sandboxes must reach CAPTURE, not vLLM: opencode is pointed at the proxy and its api key is the
# capture session id, so vLLM is never exposed and never sees a sandbox. Capture is published with a
# gradio tunnel because E2B runs off-cluster; `direct` only works for a sandbox on this host and fails
# silently -- opencode cannot reach the engine, makes zero model calls, and the rollout returns an
# empty answer that grades exactly like a model that could not do the task.

REPO=/fsx/$USER/projects/trl_prod
ENVDIR=$REPO/HuggingEnvs/04-data-agent/envs/blackbox-opencode
TRAINDIR=$REPO/HuggingEnvs/04-data-agent/train

# TWO INTERPRETERS, ON PURPOSE.
#   .venv312          torch / vLLM / TRL. The heavy stack, shared and reproducible via install.sh.
#   $ENVDIR/.venv     the environment server only. Deliberately light -- no torch, no vLLM.
# The trainer needs two LIGHT packages that .venv312 does not carry: the local `openenv` (the capture
# stack is in no release) and `data_agent_env`. Both go on PYTHONPATH rather than being pip-installed,
# because install.sh owns what is in .venv312 and an ad hoc install there stops being reproducible.
# `data_agent_env` needs the symlink in _pypath/: the directory is named `blackbox-opencode`, which is
# not a legal Python identifier, so PYTHONPATH alone cannot reach it.
PY312=$REPO/.venv312/bin/python
export PYTHONPATH="$TRAINDIR/_pypath:$REPO/OpenEnv/src${PYTHONPATH:+:$PYTHONPATH}"
export TRL_EXPERIMENTAL_SILENCE=1

cd "$ENVDIR"

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
VLLM_PORT="${VLLM_PORT:-18100}"
ENV_PORT="${ENV_PORT:-18200}"
CAP_PORT="${CAP_PORT:-18300}"
VLLM_URL="http://127.0.0.1:$VLLM_PORT"
LOGS="$ENVDIR/logs/job-${SLURM_JOB_ID:-local}"
mkdir -p "$LOGS"

set -a; . "$REPO/experiments/.env"; set +a
. "$ENVDIR/../blackbox-harbor/logs/hf_token.sh"

cleanup() { pkill -P $$ 2>/dev/null || true; }
trap cleanup EXIT

echo "== vLLM on GPU 0 =="
CUDA_VISIBLE_DEVICES=0 VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
  "$PY312" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$VLLM_PORT" --served-model-name "$MODEL" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" --trust-remote-code \
  --return-tokens-as-token-ids --logprobs-mode processed_logprobs \
  > "$LOGS/vllm.log" 2>&1 &

# vLLM's /health returns 200 with an EMPTY body, so check the STATUS and parse nothing. A probe that
# json.loads() it throws, and a caller treating that as "not ready" waits out the whole budget on a
# server that came up minutes ago.
echo "waiting for vLLM (up to 30 min: weights + torch.compile)"
for _ in $(seq 1 900); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "$VLLM_URL/health" || true)
  [ "$code" = "200" ] && break
  sleep 2
done
[ "$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$VLLM_URL/health")" = "200" ] || {
  echo "FATAL: vLLM never became healthy; its log follows"; tail -60 "$LOGS/vllm.log"; exit 1; }
echo "vLLM healthy"

echo "== env server + capture =="
PORT="$ENV_PORT" CAPTURE_PORT="$CAP_PORT" EXPOSE=gradio \
  SPLITS="${SPLITS:-train}" SANDBOX="${SANDBOX:-e2b}" \
  MAX_CONCURRENT="${MAX_CONCURRENT:-48}" \
  LLM_URL="$VLLM_URL/v1" MODEL="$MODEL" \
  ./serve.sh > "$LOGS/env.log" 2>&1 &

for _ in $(seq 1 180); do
  curl -s -o /dev/null --max-time 3 "http://127.0.0.1:$ENV_PORT/health" && break
  sleep 2
done
curl -s -o /dev/null --max-time 5 "http://127.0.0.1:$ENV_PORT/health" || {
  echo "FATAL: env server never became healthy; its log follows"; tail -60 "$LOGS/env.log"; exit 1; }
grep -a "capture proxy ready" "$LOGS/env.log" || echo "WARNING: capture did not warm at boot"
echo "env server healthy"

echo "== trainer on GPU 1 =="
# NOT from $REPO: the `trl/` submodule DIRECTORY shadows the `trl` package as a namespace package
# there, and the import dies with "cannot import name '__version__' from trl (unknown location)".
cd "$TRAINDIR"
CUDA_VISIBLE_DEVICES=1 TRACKIO_STORAGE_MODE=sqlite \
  "$PY312" -u train_blackbox_opencode.py \
  --server "http://127.0.0.1:$ENV_PORT" \
  --vllm-url "$VLLM_URL" \
  --model "$MODEL" \
  --split "${SPLITS:-train}" \
  --sandbox "${SANDBOX:-e2b}" \
  --learning-rate "${LR:-3e-6}" \
  --num-generations "${NUM_GENERATIONS:-8}" \
  --max-inflight "${MAX_INFLIGHT:-32}" \
  --grad-accum "${GRAD_ACCUM:-4}" \
  --max-steps "${MAX_STEPS:-400}" \
  --output-dir "$LOGS/run"
