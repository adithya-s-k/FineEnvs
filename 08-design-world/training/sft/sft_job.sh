# HF Job: TRL LoRA SFT of a Qwen VLM on DesignGym (run by launch.py; an optional data bucket is mounted at /data).
# env: BUNDLE_REPO BUNDLE_PATH RUN_NAME MODEL TRAIN_ARGS; secret: HF_TOKEN
set -euo pipefail
cd /work 2>/dev/null || { mkdir -p /work && cd /work; }
echo "[job] $(date -u +%FT%TZ) run $RUN_NAME"
uv venv -q /opt/venv --python 3.12
. /opt/venv/bin/activate
uv pip install -q torch==2.14.1 torchvision==0.29.1 transformers==5.19.0 trl==1.15.0 peft==0.21.2 accelerate==1.15.0 \
    datasets==5.1.0 pyarrow==26.0.0 pillow==12.3.0 safetensors==0.8.0 "huggingface_hub[hf_xet]" flash-linear-attention
python -c "import torch, fla; print('[job] torch', torch.__version__, 'cuda', torch.version.cuda, 'fla', fla.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
[ "$NGPU" -ge 1 ] || { echo "[job] no GPU visible on this node"; exit 3; }

hf download "$BUNDLE_REPO" "$BUNDLE_PATH" --repo-type dataset --local-dir /work/in >/dev/null
tar xzf "/work/in/$BUNDLE_PATH" -C /work
ls /data /data/data 2>/dev/null | head -20
t0=$(date +%s); hf download "$MODEL" --quiet >/dev/null; echo "[job] base weights in $(( $(date +%s) - t0 ))s"

cd /work/sft
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node "$NGPU" train.py --model "$MODEL" --data /data --output /work/out $TRAIN_ARGS
echo "[job] done $(date -u +%FT%TZ)"
