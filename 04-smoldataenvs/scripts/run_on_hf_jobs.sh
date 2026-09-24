#!/usr/bin/env bash
# Launch the SmolDataEnvs SFT run on a Hugging Face Jobs GPU.
#
# Jobs run in an isolated container with no access to this checkout, so the
# script is passed by URL rather than by path. That is also why HUB_MODEL_ID
# matters: the container is deleted when the job ends, and anything not pushed
# to the Hub goes with it.
#
#   ./run_on_hf_jobs.sh you/smoldataenvs-sft-360m
#   MODEL=HuggingFaceTB/SmolLM3-3B FLAVOR=a10g-large ./run_on_hf_jobs.sh you/smoldataenvs-sft-3b
#
# Requires a Hugging Face account with Jobs enabled, and `hf auth login`.
set -euo pipefail

HUB_MODEL_ID="${1:?usage: $0 <hub-model-id>   e.g. you/smoldataenvs-sft-360m}"

SCRIPT_URL="${SCRIPT_URL:-https://raw.githubusercontent.com/adithya-s-k/FineEnvs/main/04-smoldataenvs/scripts/train_sft.py}"
MODEL="${MODEL:-HuggingFaceTB/SmolLM2-360M-Instruct}"
FLAVOR="${FLAVOR:-a10g-large}"   # 360M fits t4-medium; 3B wants a10g-large
TIMEOUT="${TIMEOUT:-2h}"         # the 30m default is shorter than any real run
TRACKIO_SPACE="${TRACKIO_SPACE:-}"

echo "model      $MODEL"
echo "dataset    FineEnvs/SmolDataEnvs-sft"
echo "pushing to https://huggingface.co/$HUB_MODEL_ID"
echo "flavor     $FLAVOR   timeout $TIMEOUT"
echo

# Flags go BEFORE the script URL; anything after it is ignored.
hf jobs uv run \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --secrets HF_TOKEN \
  -e "MODEL=$MODEL" \
  -e "HUB_MODEL_ID=$HUB_MODEL_ID" \
  -e "TRACKIO_SPACE=$TRACKIO_SPACE" \
  "$SCRIPT_URL"

echo
echo "watch it:   hf jobs ps"
echo "logs:       hf jobs logs <job-id>"
