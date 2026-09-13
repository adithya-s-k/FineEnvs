#!/usr/bin/env bash
# Serve the data-agent Harbor catalog through OpenEnv. Credentials by NAME from experiments/.env.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ../../../../experiments/.env; set +a
. "$(dirname "$0")/hf_token.sh"
# `harbor serve` defaults MAX_CONCURRENT_ENVS to 4 -- a demo setting. A trainer or an eval
# holds one WebSocket SESSION per in-flight rollout, so anything above four is refused with
# "Server at capacity: 4/4 sessions active (CAPACITY_REACHED)" -- which arrives as a failed
# rollout, not as a queue, so an eval burns its whole split in seconds scoring nothing.
# Separate from the EXECUTION ceiling: the capture proxy is a single uvicorn process that
# starved /health at ~200 concurrent and crashed at 320, so stay well under that.
export MAX_CONCURRENT_ENVS="${MAX_CONCURRENT_ENVS:-128}"

exec .venv/bin/python -m openenv.cli harbor serve \
  --dataset "${DATASET:-HuggingEnvs/data-agent-harbor-train}" \
  --llm-url "${LLM_URL:?set LLM_URL}" \
  --model "${MODEL:-Qwen/Qwen3.5-2B}" \
  --port "${PORT:-8210}" \
  --capture-port "${CAPTURE_PORT:-8311}" \
  --expose "${EXPOSE:-gradio}"
