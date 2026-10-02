#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python prepare.py --output "${SMOLDATA_DATA:-prepared}"
exec uvicorn smoldataenv_whitebox.server:app --host 0.0.0.0 --port "${PORT:-7860}" --ws-ping-timeout 1800
