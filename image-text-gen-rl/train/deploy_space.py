"""Deploy the environment as a private Docker Space built on the published task dataset.

  Space    AdithyaSK/image-text-gen-rl-env      (private: it spends the owner's inference credits)
  Dataset  AdithyaSK/image-text-gen-rl-prompts  mounted read-only at /dataset (tasks served)
  Bucket   AdithyaSK/image-text-gen-rl          mounted read-write at /data
           /data/audit/        grading records + sampled images (server/audit.py)
           /data/calibration/  verifier calibration reports

The verifier token is stored as a Space secret; model routes and limits as variables.

  ../../launch image-text-gen-rl --exec python image-text-gen-rl/train/deploy_space.py
"""

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token
from image_text_gen.data.catalog import PUBLISHED_DATASET, PUBLISHED_REVISION

HERE = Path(__file__).resolve().parents[1]
ENV_DIR = HERE / "envs" / "image_text_gen"
IGNORE = [".venv/*", "**/__pycache__/*", ".pytest_cache/*", ".ruff_cache/*", "tests/*", "*.egg-info/*"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", default="AdithyaSK/image-text-gen-rl-env")
    parser.add_argument("--bucket", default="AdithyaSK/image-text-gen-rl")
    parser.add_argument("--dataset", default=PUBLISHED_DATASET)
    parser.add_argument("--hardware", default="cpu-basic")
    parser.add_argument("--concurrency", default="32", help="verifier images in flight")
    parser.add_argument("--sessions", default="64", help="concurrent OpenEnv sessions")
    parser.add_argument("--image-rate", default="0.05", help="fraction of graded images kept")
    parser.add_argument("--models", default=None, help="override verifier routes (model:provider,...)")
    args = parser.parse_args()

    api, token = HfApi(), get_token()
    if not token:
        raise SystemExit("Log in with `hf auth login` first")

    if not PUBLISHED_REVISION:
        raise SystemExit("Run train/publish_dataset.py first: no pinned dataset revision")
    api.create_bucket(args.bucket, private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        (staging / "calibration").mkdir()
        for path in (HERE / "results").glob("verifier-calibration*"):
            (staging / "calibration" / path.name).write_bytes(path.read_bytes())
        full = HERE / "artifacts" / "calibration-40.json"
        if full.is_file():
            (staging / "calibration" / full.name).write_bytes(full.read_bytes())
        api.sync_bucket(str(staging), f"hf://buckets/{args.bucket}")
    print(f"bucket  https://huggingface.co/buckets/{args.bucket}")

    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
    api.add_space_secret(args.space, "IMAGE_TEXT_GEN_VERIFIER_TOKEN", token,
                         description="HF token paying for verifier calls on Inference Providers")
    variables = {
        "IMAGE_TEXT_GEN_DATA_DIR": "/dataset/data",
        "IMAGE_TEXT_GEN_AUDIT_DIR": "/data/audit",
        "IMAGE_TEXT_GEN_AUDIT_IMAGE_RATE": args.image_rate,
        "IMAGE_TEXT_GEN_VERIFIER_CONCURRENCY": args.concurrency,
        "IMAGE_TEXT_GEN_MAX_SESSIONS": args.sessions,
    }
    if args.models:
        variables["IMAGE_TEXT_GEN_VERIFIER_MODELS"] = args.models
    for key, value in variables.items():
        api.add_space_variable(args.space, key, value)
    api.set_space_volumes(
        args.space,
        [
            # Pinned to the commit the code was validated against, like the local default.
            Volume(type="dataset", source=args.dataset, mount_path="/dataset",
                   revision=PUBLISHED_REVISION, read_only=True),
            Volume(type="bucket", source=args.bucket, mount_path="/data", read_only=False),
        ],
    )
    api.upload_folder(
        repo_id=args.space, repo_type="space", folder_path=ENV_DIR, ignore_patterns=IGNORE,
        commit_message="Deploy image text generation RL environment",
    )
    api.request_space_hardware(args.space, args.hardware)
    subdomain = args.space.lower().replace("/", "-").replace("_", "-").replace(".", "-")
    print(f"space   https://huggingface.co/spaces/{args.space}")
    print(f"url     https://{subdomain}.hf.space  (private: send an HF token)")


if __name__ == "__main__":
    main()
