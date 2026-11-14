#!/usr/bin/env bash
# The HF Job's command: copy the source, install, prepare tasks, run, then copy the outputs to the bucket.
#
# Outputs go to the Job's local disk and reach the bucket through `hf buckets sync`: every QED_SYNC_EVERY_SEC
# while the run goes (logs, Trackio, trials and checkpoints show up as it trains), then once more at the end.
# Writing them straight into the bucket mount lost every checkpoint's weights in a smoke run: the mount uploads
# a file after it is closed, in the background, and drops whatever is still pending when the container stops.
set -euo pipefail
mkdir -p /workspace/fineproof
cp -a /fineproof-source/. /workspace/fineproof/
cd /workspace/fineproof/training
uv venv --python 3.12 --seed .venv
export PATH="$PWD/.venv/bin:$PATH"
bash jobs/install.sh
python prepare.py ${FINEPROOF_PREPARE_ARGS:-}

outputs=/workspace/outputs
mkdir -p "$outputs"
sync_outputs() { hf buckets sync "$outputs" "hf://buckets/${FINEPROOF_OUTPUT_BUCKET}" --format quiet; }
(while sleep "${QED_SYNC_EVERY_SEC:-300}"; do
    sync_outputs || echo "periodic sync to the bucket failed; the next one retries" >&2
done) &
syncer=$!
# A Job that is stopped or times out gets SIGTERM first: copy what is there before it goes.
trap 'sync_outputs || true; exit 143' TERM

action=$1
shift
status=0
if [[ "$action" == "smoke" ]]; then
    python jobs/smoke.py "$@" || status=$?
else
    python jobs/run.py "$action" "$@" || status=$?
fi
pkill -P "$syncer" 2>/dev/null || true
kill "$syncer" 2>/dev/null || true
wait "$syncer" 2>/dev/null || true
for attempt in 1 2 3; do
    sync_outputs && break
    if [[ $attempt == 3 ]]; then
        echo "final sync to the bucket failed: the outputs are lost when the Job ends" >&2
        exit 1
    fi
    sleep 30
done
echo "outputs synced to hf://buckets/${FINEPROOF_OUTPUT_BUCKET}"
exit "$status"
