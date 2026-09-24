#!/usr/bin/env bash
# Launch SmolDataEnvs training and evaluation on Hugging Face Jobs.
#
#   ./run_on_hf_jobs.sh eval   Qwen/Qwen3.5-2B                     # baseline first
#   ./run_on_hf_jobs.sh sft    you/smoldataenvs-sft-2b             # imitate correct trajectories
#   ./run_on_hf_jobs.sh grpo   you/smoldataenvs-grpo-2b            # RL against the grader
#   ./run_on_hf_jobs.sh eval   you/smoldataenvs-grpo-2b            # the same measurement, after
#
# Everything runs on the `huggingface/trl` image, so the CUDA toolchain is the
# one TRL is built against; the script's own PEP 723 header still pins the
# Python deps on top of it.
#
# The Jobs container is deleted when the job ends, so a training run that does
# not push to the Hub is a run you cannot keep. Hence the required model id.
set -euo pipefail

MODE="${1:?usage: $0 <eval|sft|grpo> <model-or-hub-id>}"
TARGET="${2:?usage: $0 <eval|sft|grpo> <model-or-hub-id>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FLAVOR="${FLAVOR:-a10g-large}"
IMAGE="${IMAGE:-huggingface/trl}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
TRACKIO_SPACE="${TRACKIO_SPACE:-}"

case "$MODE" in
  eval)
    SCRIPT="$HERE/eval_pass1.py"
    TIMEOUT="${TIMEOUT:-2h}"
    # for eval, the positional argument IS the model under test
    ENVS=(-e "MODEL=$TARGET" -e "SPLIT=${SPLIT:-eval}" -e "TRACKIO_SPACE=$TRACKIO_SPACE")
    ;;
  sft)
    SCRIPT="$HERE/train_sft.py"
    TIMEOUT="${TIMEOUT:-3h}"
    ENVS=(-e "MODEL=$MODEL" -e "HUB_MODEL_ID=$TARGET" -e "TRACKIO_SPACE=$TRACKIO_SPACE")
    ;;
  grpo)
    SCRIPT="$HERE/train_grpo.py"
    TIMEOUT="${TIMEOUT:-6h}"
    # Rollouts run in Hugging Face Sandboxes, which are themselves Jobs, so the
    # token has to reach the training job: --secrets HF_TOKEN below.
    ENVS=(
      -e "MODEL=$MODEL" -e "HUB_MODEL_ID=$TARGET" -e "TRACKIO_SPACE=$TRACKIO_SPACE"
      -e "NUM_TASKS=${NUM_TASKS:-256}" -e "MAX_STEPS=${MAX_STEPS:-200}"
      -e "NUM_GENERATIONS=${NUM_GENERATIONS:-8}"
    )
    ;;
  *)
    echo "unknown mode: $MODE (expected eval, sft or grpo)" >&2
    exit 2
    ;;
esac

echo "mode     $MODE"
echo "script   $SCRIPT"
echo "model    $MODEL"
echo "target   $TARGET"
echo "image    $IMAGE   flavor $FLAVOR   timeout $TIMEOUT"
echo

# Flags go BEFORE the script path; anything after it is passed to the script.
hf jobs uv run \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --image "$IMAGE" \
  --secrets HF_TOKEN \
  "${ENVS[@]}" \
  "$SCRIPT"

echo
echo "watch it:  hf jobs ps"
echo "logs:      hf jobs logs <job-id>"
