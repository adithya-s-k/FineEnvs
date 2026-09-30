"""Exercise the actual frozen import and CLI boundaries in a fresh HF Job."""
from common import ROOT, RUN, TOOLS, TRAIN_PY, ENV_PY, configure, verify_bundle, write_json
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    argparse.ArgumentParser().parse_known_args()
    configure()
    count = verify_bundle()
    subprocess.run([str(ENV_PY), str(ROOT / "hf/runtime/ui_smoke.py"), "--help"], check=True, stdout=subprocess.DEVNULL)
    scripts = [RUN / "source/HuggingEnvs/04-data-agent/train/train_harbor_multi.py",
               TOOLS / "train_whitebox_daytona.py", TOOLS / "eval_whitebox_native.py",
               RUN / "eval-source/eval_concurrent.py"]
    for script in scripts:
        subprocess.run([str(TRAIN_PY), str(script), "--help"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([str(ENV_PY), "-c", "from openenv.harbor.serving import HarborService; from whitebox_bash.server.environment import WhiteBoxBashEnvironment; from daytona_whitebox_backend import load_frozen_tasks; assert len(load_frozen_tasks('train'))==1000; assert len(load_frozen_tasks('test'))==250"], check=True)
    subprocess.run([str(ENV_PY), str(ROOT / "hf/runtime/check_task_schedule.py")], check=True)
    result = {"passed": True, "verified_files": count, "checked_at": time.time(),
              "bundle_sha256": os.environ["BUNDLE_SHA256"], "role": "preflight"}
    out = ROOT / "outputs/preflight"
    write_json(out / "result.json", result)
    from huggingface_hub import HfApi
    dest = "hf://buckets/" + os.environ["ARTIFACT_BUCKET"] + "/" + os.environ["RUN_ID"] + "/preflight"
    HfApi().sync_bucket(str(out), dest, quiet=True)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
