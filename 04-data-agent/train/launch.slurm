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

# PORTS DERIVED PER JOB. Fixed ports are a trap on a shared node and it is not a theoretical one:
# two of these jobs landed on ip-10-53-93-25 together, the second one's /server_info probe answered
# 200 because it was talking to the FIRST job's vLLM, and the NCCL weight-transfer group then tried to
# attach to another job's engine and died with "NCCL error: unhandled cuda error". Every preflight
# check passed while pointing at the wrong process.
#
# Worse, had the models matched, it would have trained one job against the other's weights and looked
# perfectly healthy. The reference launcher derives ports the same way and says so.
_OFF=$(( ${SLURM_JOB_ID:-0} % 200 * 4 ))
VLLM_PORT="${VLLM_PORT:-$(( 18000 + _OFF ))}"
ENV_PORT="${ENV_PORT:-$(( 18001 + _OFF ))}"
CAP_PORT="${CAP_PORT:-$(( 18002 + _OFF ))}"
VLLM_URL="http://127.0.0.1:$VLLM_PORT"
LOGS="$ENVDIR/logs/job-${SLURM_JOB_ID:-local}"
mkdir -p "$LOGS/run"

# BRIDGE THE CHECKPOINTS TO THE EVAL WATCHER.
#
# eval_watcher.py polls experiments/temp_asyncgrpo_code/logs/ckpt-<train_job>/checkpoint-* . Without
# this link it never sees a checkpoint from this run, and the 400-step job would finish with nothing
# evaluated -- silently, because "no checkpoints yet" and "wrong directory" look identical to a poller.
#
# The link, rather than writing there directly: the run's artifacts belong with the environment, and
# the eval stack is the one that produced the +0.2028 reference, so reusing it keeps the comparison
# apples-to-apples instead of introducing a second evaluator.
WATCHDIR=/fsx/$USER/projects/trl_prod/experiments/temp_asyncgrpo_code/logs/ckpt-${SLURM_JOB_ID:-local}
ln -sfn "$LOGS/run" "$WATCHDIR"
echo "checkpoints visible to eval_watcher at $WATCHDIR -> $LOGS/run"

set -a; . "$REPO/experiments/.env"; set +a
. "$ENVDIR/../blackbox-harbor/tools/hf_token.sh"

cleanup() { pkill -P $$ 2>/dev/null || true; }
trap cleanup EXIT

# SAMPLING, FOR THE ENGINE AND THE TRAINER, FROM ONE VARIABLE SO THEY CANNOT DRIFT.
# opencode sends NO sampling parameters, so whatever the engine defaults to is what actually
# generated the actions -- and Qwen3.5-2B ships no generation_config.json, so that default is
# vLLM's own 1.0/1.0, NOT the 0.8 the trainer divides logits by when it recomputes logprobs for
# the importance ratio. Leaving the engine unpinned means the gradient is computed against a
# distribution that never produced the samples. Measured, unpinned: entropy climbed 0.229 -> 0.587
# over 24 steps while reward fell 0.592 -> 0.216, with turns/mean 10.4 against the reference's 6.1
# (job 76577). The reference pins both from one variable (run_cluster.py:165) and holds entropy
# flat at 0.17-0.21. top_k=-1 because top_k truncation is likewise unmodelled by the recomputation.
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"

echo "== vLLM on GPU 0 ==  (ports vllm=$VLLM_PORT env=$ENV_PORT capture=$CAP_PORT, derived from job ${SLURM_JOB_ID:-0})"
# EVERY ONE OF THESE FLAGS IS LOad-BEARING. They are the set the +0.2028 run served with; a
# hand-rolled `vllm serve` that omits any of them fails silently rather than loudly.
#
#   --enable-auto-tool-choice / --tool-call-parser qwen3_xml
#       Without them vLLM never parses tool calls, so `has_tool_call` is False for EVERY turn and the
#       whole rollout is discarded with no error anywhere. The agent path is entirely tool calls.
#   --default-chat-template-kwargs {"enable_thinking": false}
#       Pins thinking OFF server-side. Qwen3.5-4B's template opens <think> by default and -2B's
#       closes it -- INVERTED defaults between two models of the same family. Leaving it to the
#       default is what made a re-rendered prompt match the 2B and diverge on the 4B, forking 100% of
#       turn transitions. This is what `think_template_patch` used to monkeypatch; it belongs here.
#   --return-tokens-as-token-ids / --logprobs-mode processed_logprobs
#       The capture tier. Without them capture degrades to text and every rollout is untrainable --
#       and looks completely normal while being so.
#   --gdn-prefill-backend triton
#       Qwen3.5 is hybrid Gated-DeltaNet; this is the working prefill backend for it.
#   VLLM_USE_DEEP_GEMM=0 / VLLM_DEEP_GEMM_WARMUP=skip
#       On Hopper a stale importable deep_gemm kills startup with "DeepGEMM backend is not available
#       or outdated".
#   VLLM_SERVER_DEV_MODE=1
#       Gates /server_info, /pause and /init_weight_transfer_engine. TRL's weight transfer reads
#       /server_info for the server's dtype, so WITHOUT this the trainer dies at on_train_begin with
#       "404 Client Error: Not Found for url: .../server_info?config_format=json" -- after vLLM is
#       healthy, after the env server is up, after the dataset loads. Job 76488 died exactly there.
#   VLLM_USE_FLASHINFER_SAMPLER=0
#       flashinfer JIT-compiles its sampling kernel and needs nvcc; the PyTorch sampler does not.
#   --weight-transfer-config {"backend":"nccl"}
#       The trainer syncs weights into this engine over NCCL. Without it there is no transfer engine
#       to initialise.
CUDA_VISIBLE_DEVICES=0 \
  VLLM_SERVER_DEV_MODE=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
  "$PY312" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$VLLM_PORT" --served-model-name "$MODEL" \
  --trust-remote-code \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --gdn-prefill-backend triton \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --return-tokens-as-token-ids --logprobs-mode processed_logprobs \
  --override-generation-config "{\"temperature\": $TEMPERATURE, \"top_p\": $TOP_P, \"top_k\": -1}" \
  --weight-transfer-config '{"backend":"nccl"}' \
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
# CHECK THE ENDPOINT THE TRAINER ACTUALLY NEEDS, not just /health. /health passing says nothing about
# whether VLLM_SERVER_DEV_MODE took effect, and the trainer only finds out ~5 minutes later at
# on_train_begin, after the env server and the whole dataset have loaded.
si=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$VLLM_URL/server_info?config_format=json" || true)
[ "$si" = "200" ] || { echo "FATAL: /server_info -> $si; VLLM_SERVER_DEV_MODE=1 did not take effect."; exit 1; }
# Confirm the engine on this port is OURS. Reachability is not identity: a neighbouring job's vLLM
# answers /health and /server_info exactly the same way, and the first sign of talking to the wrong
# one is an NCCL failure minutes later -- or, if the models happen to match, no sign at all.
served=$(curl -s --max-time 10 "$VLLM_URL/v1/models" | tr -d ' \n' | grep -o "\"id\":\"[^\"]*\"" | head -1)
echo "vLLM /server_info ok; serving $served on :$VLLM_PORT"

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

# max_inflight 32 is the REFERENCE value, and it matters beyond throughput: at 16 this run
# measured samples/step 7.2 against the reference's 16.1, i.e. less than half the training
# signal per optimizer step, which over 400 steps is a different experiment rather than a
# slower one. 16 was chosen after an EVAL collapsed at 64 concurrent MCP clients -- but that
# eval ran ~16-turn rollouts (64 x 16 = ~1024 concurrent turn-slots) where training runs ~8.5
# (32 x 8.5 = ~272), about a quarter of the load that broke it.
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
  --temperature "$TEMPERATURE" \
  --num-generations "${NUM_GENERATIONS:-8}" \
  --max-inflight "${MAX_INFLIGHT:-32}" \
  --grad-accum "${GRAD_ACCUM:-4}" \
  --curriculum "${CURRICULUM:-warmup:125}" \
  --agent-step-limit "${AGENT_STEP_LIMIT:-17}" \
  --token-budget "${TOKEN_BUDGET:-40960}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH:-16384}" \
  --heartbeat-stale-after-s "${HEARTBEAT_STALE_AFTER_S:-900}" \
  --dtype "${TRAIN_DTYPE:-bfloat16}" \
  --max-steps "${MAX_STEPS:-400}" \
  --save-steps "${SAVE_STEPS:-100}" \
  --output-dir "$LOGS/run"
