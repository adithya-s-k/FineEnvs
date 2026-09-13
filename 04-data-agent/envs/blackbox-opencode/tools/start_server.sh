#!/usr/bin/env bash
# Start blackbox-opencode against the live engine. Credentials come from experiments/.env by NAME.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ../../../../experiments/.env; set +a
. "$(dirname "$0")/../../blackbox-harbor/tools/hf_token.sh"
export PORT="${PORT:-8200}"
export SPLITS="${SPLITS:-train:medium}"
export SANDBOX="${SANDBOX:-e2b}"
export EXPOSE="${EXPOSE:-gradio}"
export CAPTURE_PORT="${CAPTURE_PORT:-8301}"
export LLM_URL="${LLM_URL:?set LLM_URL}"
export MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
exec ./serve.sh
