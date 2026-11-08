#!/usr/bin/env bash
# HF Job (novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron, one node): SkyRL GRPO on DesignGym GUI episodes.
# DesignGym runs here (common/job_common.sh); vLLM serves the policy (merged LoRA weights), Megatron trains it.
#
# The image has Ray and uv but not SkyRL's `megatron` extra (vLLM, Megatron-Bridge, TE, ...): `uv sync` builds it at
# job start. With CACHE_DIR (an HF bucket mounted by the launcher's --cache-bucket) the built venv is kept as one tar
# there and unpacked on later runs (sequential reads are fast on bucket mounts; many small files are not).
# Env: see common/job_common.sh, plus SKYRL_REF (git commit), TRAIN_ARGS (extra SkyRL overrides), GPUS. Secret: HF_TOKEN.
set -euo pipefail
python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub 2>/dev/null \
  || uv pip install --system -q huggingface_hub 2>/dev/null || uv pip install --system --break-system-packages -q huggingface_hub
export W=$( (mkdir -p /work && touch /work/.w) 2>/dev/null && echo /work || echo $HOME/work)
mkdir -p $W
python3 - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
tarfile.open(hf_hub_download(os.environ["BUNDLE_REPO"], os.environ["BUNDLE_PATH"], repo_type="dataset")).extractall(os.environ["W"])
PY
source $W/rl/common/job_common.sh
trap 'rc=$?; log "exit $rc"; grep -E "policy/|rollout_train_logprobs|Finished step|global_step|designgym/reward" $OUT/train.log > $OUT/steps.log 2>/dev/null || true; nvidia-smi > $OUT/nvidia-smi.txt 2>&1 || true; upload_results' EXIT
gpu_watch
nvidia-smi -L
GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
SKYRL_REF=${SKYRL_REF:-3d0ae22}

make_tasks
start_env > $OUT/start-env.log 2>&1 &
ENV_PID=$!

log "SkyRL $SKYRL_REF"
git clone -q https://github.com/NovaSky-AI/SkyRL.git $W/skyrl && git -C $W/skyrl checkout -q "$SKYRL_REF"
for p in $W/rl/skyrl/patches/*.patch; do [ -f "$p" ] && { log "applying $(basename $p)"; git -C $W/skyrl apply "$p"; }; done
LOCK_HASH=$(cat $W/skyrl/uv.lock $W/skyrl/pyproject.toml | sha1sum | cut -c1-12)
VENV_TAR=${CACHE_DIR:+$CACHE_DIR/skyrl-venv-megatron-$LOCK_HASH.tar}
cd $W/skyrl
T0=$(date +%s)
if [ -n "$VENV_TAR" ] && [ -f "$VENV_TAR" ]; then
  log "unpacking the cached venv $(basename $VENV_TAR) ($(du -sh $VENV_TAR | cut -f1))"
  tar -xf "$VENV_TAR" -C $W/skyrl
  CACHED=1
else
  log "building the megatron venv (uv sync --extra megatron)"
  uv sync --frozen --extra megatron > $OUT/uv-sync.log 2>&1 || { tail -40 $OUT/uv-sync.log; exit 1; }
  CACHED=0
fi
log "venv ready in $(( $(date +%s) - T0 )) s ($(du -sh .venv | cut -f1))"
PYV=$W/skyrl/.venv/bin/python
uv pip install --python $PYV -q websockets > /dev/null
echo $W > "$($PYV -c 'import site; print(site.getsitepackages()[0])')/designgym_rl.pth"
if [ -n "$VENV_TAR" ] && [ "$CACHED" = 0 ]; then
  (tar -cf "$VENV_TAR.tmp" .venv && mv "$VENV_TAR.tmp" "$VENV_TAR" && echo "[cache] saved $VENV_TAR") > $OUT/venv-cache.log 2>&1 &
  CACHE_PID=$!
fi
PY=$PYV
$PY -c "import rl.skyrl.designgym_generator, rl.skyrl.entrypoint; print('designgym adapter imports ok')"

download_model
wait $ENV_PID || { cat $OUT/start-env.log; exit 1; }
cat $OUT/start-env.log
export DESIGNGYM_URL=${ENV_URL:-http://127.0.0.1:8027}

$PY - <<'PY'
import json
import pandas as pd
import os
rows = [json.loads(l) for l in open(os.environ["W"] + "/data/tasks.jsonl")]
pd.DataFrame([{"prompt": [{"role": "user", "content": "designgym"}], "env_class": "designgym",
               "task": json.dumps(r)} for r in rows]).to_parquet(os.environ["W"] + "/data/train.parquet")
print(len(rows), "rows -> train.parquet")
PY

log "training"
set +e
$PY -m rl.skyrl.entrypoint \
  data.train_data="['$W/data/train.parquet']" \
  data.val_data="['$W/data/train.parquet']" \
  trainer.policy.model.path="$HF_DIR" \
  trainer.strategy=megatron \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$GPUS \
  trainer.policy.language_model_only=false \
  generator.inference_engine.language_model_only=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.engine_init_kwargs='{"gdn_prefill_backend": "triton", "disable_custom_all_reduce": true, "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}' \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  trainer.remove_microbatch_padding=false \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.off_policy_correction.tis_ratio_type=token \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_tokens_per_microbatch=-1 \
  trainer.policy.optimizer_config.lr=1e-5 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_path=$W/ckpt \
  trainer.export_path=$W/exports \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=none \
  trainer.logger=console \
  trainer.project_name=designgym \
  trainer.run_name="$RUN_NAME" \
  generator.batched=false \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=false \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.logprobs=1 \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  environment.env_class=designgym \
  $TRAIN_ARGS 2>&1 | tee $OUT/train.log
RC=${PIPESTATUS[0]}
set -e
log "trainer exited $RC"
if [ -d $W/ckpt ]; then
  find $W/ckpt -maxdepth 4 | head -200 > $OUT/ckpt-tree.txt
  du -sh $W/ckpt/* >> $OUT/ckpt-tree.txt 2>/dev/null || true
fi
[ -n "${CACHE_PID:-}" ] && wait $CACHE_PID || true  # never a bare `wait`: gpu_watch and the env server run forever
exit $RC
