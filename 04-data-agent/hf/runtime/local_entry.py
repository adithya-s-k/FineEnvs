"""Use the same audited HF runner with local GPUs and a local native environment."""
import json
import os
from pathlib import Path
import sys

from dotenv import dotenv_values

values = dotenv_values(os.environ["LOCAL_ENV_FILE"])
os.environ["HF_TOKEN"] = values.get("HF_API_KEY") or values["HF_TOKEN"]
for key in ["DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET", "E2B_API_KEY"]:
    if values.get(key):
        os.environ[key] = values[key]
root = Path(os.environ["REPRO_ROOT"])
config = json.loads((root / "hf/configs/deployment.json").read_text())
gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
if len(gpu_ids) != 2:
    raise RuntimeError("The local recipe requires two allocated GPUs")
arm = sys.argv[sys.argv.index("--arm") + 1]
role = sys.argv[sys.argv.index("--role") + 1]
job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
port_seed = int(job) % 1000
os.environ.update(LOCAL_RUNTIME="1", RUN_OWNER=f"local-{role}-{arm}-{job}",
    RUN_ID=config["run_id"], COMPARISON_ARM=arm,
    BUNDLE_SHA256=json.loads((root / "local_manifest.json").read_text())["sha256"],
    ARTIFACT_BUCKET=config["resources"]["artifacts_bucket"], JOB_FLAVOR="hopper-prod-2h100",
    INFERENCE_GPU=gpu_ids[0], TRAIN_GPU=gpu_ids[1],
    LOCAL_INFERENCE_PORT=str(12000 + port_seed), LOCAL_ENV_PORT=str(14000 + port_seed),
    DATA_AGENT_CAPTURE_PORT=str(16000 + port_seed), VLLM_DP_RPC_PORT=str(26000 + port_seed),
    EVAL_BACKENDS="daytona", EVAL_CONCURRENCY="50", EVAL_DAYTONA_CONCURRENCY="50", EVAL_NO_RAMP="1",
    SANDBOX_CAPACITY="50" if role == "eval" else "16", TRAIN_RESERVED_SANDBOXES="8",
    HF_HOME=os.environ.get("HF_HOME", str(root / "cache/huggingface")))
from job import main
try:
    main()
finally:
    import subprocess
    subprocess.run([str(root / 'OpenEnv/.venv/bin/python'), str(root / 'hf/runtime/cleanup_local.py')],
                   timeout=240, check=True)
