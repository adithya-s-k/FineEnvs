"""Upload one environment to a CPU Space, using the same source as local jobs."""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["whitebox", "opencode", "harbor"], required=True
    )
    parser.add_argument("--repo", required=True, help="Your HF owner/space-name")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(
        args.repo,
        repo_type="space",
        space_sdk="docker",
        private=not args.public,
        exist_ok=True,
    )
    api.add_space_variable(args.repo, "ENV_MODE", args.mode)
    api.add_space_variable(args.repo, "MAX_CONCURRENT_ENVS", str(args.concurrency))
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary)
        for folder in ("envs", "data"):
            for source in (ROOT / folder).rglob("*"):
                if source.is_file() and source.suffix in {
                    ".py",
                    ".sh",
                    ".json",
                    ".txt",
                    ".yaml",
                }:
                    target = destination / source.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        shutil.copyfile(ROOT / "prepare.py", destination / "prepare.py")
        shutil.copyfile(ROOT / "envs/Dockerfile", destination / "Dockerfile")
        title = {
            "whitebox": "SETA whitebox",
            "opencode": "native OpenCode",
            "harbor": "Harbor multi-harness",
        }[args.mode]
        oauth = (
            "hf_oauth: true\nhf_oauth_scopes:\n  - inference-api\n"
            if args.mode == "harbor"
            else ""
        )
        (destination / "README.md").write_text(
            f"---\ntitle: SmolDataEnv {title}\nsdk: docker\napp_port: 7860\n{oauth}---\n\n"
            "Open `/web` for the environment UI and `/docs` for the API. "
            "Set DAYTONA_API_KEY and HF_TOKEN in Space secrets before running tasks.\n"
        )
        api.upload_folder(
            repo_id=args.repo,
            repo_type="space",
            folder_path=temporary,
            commit_message=f"Deploy SmolDataEnv {title}",
        )
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
