#!/usr/bin/env bash
# HF Job (radixark/miles:latest, one node): Miles GRPO on DesignGym GUI episodes.
# DesignGym runs here (common/job_common.sh); the model is served by SGLang and trained by Megatron-Bridge.
# Env: see common/job_common.sh, plus TRAIN_ARGS (train_designgym.py options). Secret: HF_TOKEN.
set -euo pipefail
python3 - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
tarfile.open(hf_hub_download(os.environ["BUNDLE_REPO"], os.environ["BUNDLE_PATH"], repo_type="dataset")).extractall("/work")
PY
source /work/rl/common/job_common.sh
trap 'rc=$?; log "exit $rc"; grep -E "step [0-9]+:|perf [0-9]+:" $OUT/train.log > $OUT/steps.log 2>/dev/null || true; nvidia-smi > $OUT/nvidia-smi.txt 2>&1 || true; upload_results' EXIT
gpu_watch
nvidia-smi -L

make_tasks
python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("/work/data/tasks.jsonl")]
with open("/work/data/miles.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps({"prompt": "designgym", "metadata": r}) + "\n")
PY

start_env > $OUT/start-env.log 2>&1 &
ENV_PID=$!
download_model
wait $ENV_PID || { cat $OUT/start-env.log; exit 1; }
cat $OUT/start-env.log
DESIGNGYM_URL=${ENV_URL:-http://127.0.0.1:8027}

# our package for every Python process of the run (Ray workers included)
echo /work > "$(python3 -c 'import site; print(site.getsitepackages()[0])')/designgym_rl.pth"
python3 -c "import websockets" 2>/dev/null || pip install -q websockets
python3 -c "import rl.common.episode, rl.miles.designgym_generate; print('designgym adapter imports ok')"

log "smoke-checking the env from this Python"
python3 - <<PY
import asyncio
from rl.common.env_client import EnvSession
async def main():
    async with EnvSession("$DESIGNGYM_URL") as s:
        m = await s.reset(split="rl", index=0, surface="gui", mode="reference", skin="figma", coordinates="normalized", max_steps=5)
        shot = await s.call("screenshot", {})
        print("env ok:", m["task_id"], len(shot.images), "image(s)")
asyncio.run(main())
PY

log "training"
cd /root/miles
set +e
python3 /work/rl/miles/train_designgym.py --model "$MODEL" --hf-dir "$HF_DIR" --data /work/data/miles.jsonl \
  --save /work/ckpt --env-url "$DESIGNGYM_URL" $TRAIN_ARGS 2>&1 | tee $OUT/train.log
RC=${PIPESTATUS[0]}
set -e
log "trainer exited $RC"
# LoRA adapters (small) and the checkpoint layout; full Megatron states stay on the node
if [ -d /work/ckpt ]; then
  find /work/ckpt -maxdepth 3 | head -100 > $OUT/ckpt-tree.txt
  du -sh /work/ckpt/* >> $OUT/ckpt-tree.txt 2>/dev/null || true
  for d in /work/ckpt/iter_*/adapter; do [ -d "$d" ] && mkdir -p $OUT/ckpt/$(basename $(dirname $d)) && cp -r $d $OUT/ckpt/$(basename $(dirname $d))/; done
fi
grep -q "step 1:" $OUT/train.log || { log "no training step logged"; exit 1; }
exit $RC
