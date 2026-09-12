#!/usr/bin/env bash
# Serve the data-agent Harbor catalog through OpenEnv. Credentials by NAME from experiments/.env.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ../../../../experiments/.env; set +a
. "$(dirname "$0")/hf_token.sh"
exec .venv/bin/python -m openenv.cli harbor serve \
  --dataset "${DATASET:-HuggingEnvs/data-agent-harbor-train}" \
  --llm-url "${LLM_URL:?set LLM_URL}" \
  --model "${MODEL:-Qwen/Qwen3.5-2B}" \
  --port "${PORT:-8210}" \
  --capture-port "${CAPTURE_PORT:-8311}" \
  --expose "${EXPOSE:-gradio}"
