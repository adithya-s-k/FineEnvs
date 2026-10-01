#!/usr/bin/env bash
set -euo pipefail
cd /app
python prepare.py --output "${SMOLDATA_DATA:-prepared}"
exec uvicorn "envs.${ENV_MODE:-harbor}.server:app" --host 0.0.0.0 --port "${PORT:-7860}" --ws-ping-timeout 1800
