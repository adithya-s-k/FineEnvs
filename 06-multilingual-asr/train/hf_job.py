# /// script
# requires-python = ">=3.11"
# dependencies = ["uv>=0.8,<1", "huggingface-hub==1.31.0"]
# ///
"""HF Jobs launcher: fetch a pushed source commit and use its frozen lockfile."""

import argparse
import io
import re
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", required=True, help="Pushed 40-character Git commit"
    )
    parser.add_argument("--repo", default="adithya-s-k/FineEnvs")
    parser.add_argument("--mode", choices=("env-smoke", "train"), default="env-smoke")
    parser.add_argument("--languages", nargs="+", default=["en_us", "hi_in"])
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--per-split", type=int, default=8)
    parser.add_argument("--source-root", help="Attached fleurs bucket mount")
    parser.add_argument("--artifact-repo")
    args, extra = parser.parse_known_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full commit SHA")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        parser.error("--repo must be owner/repository")

    with tempfile.TemporaryDirectory(prefix="asr-job-") as directory:
        directory = Path(directory)
        url = f"https://api.github.com/repos/{args.repo}/zipball/{args.revision}"
        with urllib.request.urlopen(url, timeout=120) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                source = directory / "source"
                for name in archive.namelist():
                    if not (source / name).resolve().is_relative_to(source.resolve()):
                        raise ValueError("Unsafe path in source archive")
                archive.extractall(source)
        root = next(source.iterdir()) / "06-multilingual-asr"
        project = root / "envs" / "multilingual_asr"
        run = directory / "run"
        run.mkdir()
        (run / "source.json").write_text(
            f'{{"repo":"{args.repo}","revision":"{args.revision}"}}\n'
        )
        base = ["uv", "run", "--frozen", "--project", str(project)]
        print(f"Source: {args.repo}@{args.revision}; mode={args.mode}", flush=True)
        try:
            if args.mode == "env-smoke":
                subprocess.run(
                    base + ["--extra", "dev", "pytest", str(project / "tests"), "-q"],
                    check=True,
                )
                subprocess.run(
                    base + ["asr-smoke", "--output", str(run / "smoke.json")] + extra,
                    check=True,
                )
                return
            # Training needs a snapshot; build one inside the job from the pinned copy.
            snapshot = directory / "snapshot"
            prepare = base + [
                "asr-prepare",
                "--output",
                str(snapshot),
                "--languages",
                *args.languages,
                "--splits",
                *args.splits,
                "--per-split",
                str(args.per_split),
                "--revision",
                args.revision,
            ]
            if args.source_root:
                prepare += ["--source-root", args.source_root]
            subprocess.run(prepare, check=True)
            (run / "snapshot-manifest.json").write_bytes(
                (snapshot / "manifest.json").read_bytes()
            )
            subprocess.run(
                base
                + [
                    "--extra",
                    "train",
                    "python",
                    str(root / "train" / "grpo_asr.py"),
                    "--snapshot",
                    str(snapshot),
                    "--output-dir",
                    str(run),
                ]
                + extra,
                check=True,
            )
        finally:
            if args.artifact_repo:
                from huggingface_hub import HfApi

                api = HfApi()
                api.create_repo(
                    args.artifact_repo, repo_type="dataset", private=True, exist_ok=True
                )
                api.upload_folder(
                    repo_id=args.artifact_repo,
                    repo_type="dataset",
                    folder_path=run,
                    path_in_repo=args.revision,
                    commit_message=f"ASR {args.mode}: {args.revision}",
                )


if __name__ == "__main__":
    main()
