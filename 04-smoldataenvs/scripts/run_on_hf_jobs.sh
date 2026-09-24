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

# Jobs fetch the script over HTTP, and raw.githubusercontent.com caches a branch
# path for 300s -- long enough to run the *previous* version of a script you just
# fixed, which is a confusing way to lose a GPU-minute. Pin to this commit
# instead: immutable URL, no cache to be stale, and the job is reproducible.
SHA="${SHA:-$(git -C "$HERE" rev-parse HEAD)}"
RAW="${RAW:-https://raw.githubusercontent.com/adithya-s-k/FineEnvs/$SHA/04-smoldataenvs/scripts}"

FLAVOR="${FLAVOR:-a10g-large}"   # grpo on a 2B needs a100-large: full fine-tune + colocated vLLM
IMAGE="${IMAGE:-huggingface/trl}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
TRACKIO_SPACE="${TRACKIO_SPACE:-}"

case "$MODE" in
  eval)
    SCRIPT="$RAW/eval_pass1.py"
    TIMEOUT="${TIMEOUT:-2h}"
    # for eval, the positional argument IS the model under test
    ENVS=(
      -e "MODEL=$TARGET" -e "SPLIT=${SPLIT:-eval}" -e "NUM_TASKS=${NUM_TASKS:-0}"
      -e "ROLLOUT_URL=$RAW/rollout.py" -e "TRACKIO_SPACE=$TRACKIO_SPACE"
    )
    ;;
  sft)
    SCRIPT="$RAW/train_sft.py"
    TIMEOUT="${TIMEOUT:-3h}"
    ENVS=(
      -e "MODEL=$MODEL" -e "HUB_MODEL_ID=$TARGET" -e "MAX_STEPS=${MAX_STEPS:-0}"
      -e "TRACKIO_SPACE_ID=$TRACKIO_SPACE"
    )
    ;;
  grpo)
    SCRIPT="$RAW/train_grpo.py"
    TIMEOUT="${TIMEOUT:-6h}"
    # Rollouts run in Hugging Face Sandboxes, which are themselves Jobs, so the
    # token has to reach the training job: --secrets HF_TOKEN below.
    ENVS=(
      -e "MODEL=$MODEL" -e "HUB_MODEL_ID=$TARGET" -e "TRACKIO_SPACE_ID=$TRACKIO_SPACE"
      -e "ROLLOUT_URL=$RAW/rollout.py"
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
echo "commit   $SHA"
echo

# Flags go BEFORE the script path; anything after it is passed to the script.
hf jobs uv run --detach \
  --flavor "$FLAVOR" \
  --timeout "$TIMEOUT" \
  --image "$IMAGE" \
  --secrets HF_TOKEN \
  "${ENVS[@]}" \
  "$SCRIPT"

echo
echo "watch it:  hf jobs ps"
echo "logs:      hf jobs logs <job-id>"
