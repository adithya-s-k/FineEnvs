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
    parser.add_argument(
        "--mode",
        choices=("env-smoke", "real-smoke", "train", "eval-vllm"),
        default="env-smoke",
    )
    parser.add_argument(
        "--corpus-manifest",
        help="Published full-corpus manifest HF URI, local path, or 'repo' for the "
        "manifest committed at 05-multilingual-ocr/data/",
    )
    parser.add_argument(
        "--evalset",
        help="Frozen evaluation set committed under 05-multilingual-ocr/data/, by file "
        "name; the job resolves it inside its own checkout",
    )
    parser.add_argument("--prepare-pages", type=int, default=256)
    parser.add_argument(
        "--prepare-languages", nargs="+", default=["en", "kn", "hi", "ar"]
    )
    parser.add_argument("--prepare-max-bytes", type=int, default=8_000_000_000)
    parser.add_argument(
        "--artifact-repo",
        help="Optional personal dataset repo for run outputs; no publication by default",
    )
    args, extra = parser.parse_known_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full commit SHA")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        parser.error("--repo must be owner/repository")
    with tempfile.TemporaryDirectory(prefix="nayana-job-") as directory:
        directory = Path(directory)
        url = f"https://api.github.com/repos/{args.repo}/zipball/{args.revision}"
        with urllib.request.urlopen(url, timeout=120) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                source = directory / "source"
                for name in archive.namelist():
                    if not (source / name).resolve().is_relative_to(source.resolve()):
                        raise ValueError("Unsafe path in source archive")
                archive.extractall(source)
        root = next(source.iterdir()) / "05-multilingual-ocr"
        # Paths inside the job's checkout are not knowable when the job is submitted,
        # so repo-relative data is named by file and resolved here.
        if args.corpus_manifest == "repo":
            args.corpus_manifest = str(root / "data" / "corpus-manifest.json")
        if args.evalset:
            evalset_path = root / "data" / args.evalset
            if not evalset_path.is_file():
                parser.error(f"No evaluation set {args.evalset!r} in this revision")
            extra += ["--evalset", str(evalset_path)]
        project = root / "envs" / "nayana_ocr"
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
                    base
                    + ["nayana-smoke", "--output", str(run / "smoke.json")]
                    + extra,
                    check=True,
                )
            else:
                # A hosted env already owns its prepared window.
                remote = "--env-url" in extra or any(
                    arg.startswith("--env-url=") for arg in extra
                )
                if args.corpus_manifest:
                    if remote or args.mode not in ("train", "eval-vllm"):
                        parser.error(
                            "--corpus-manifest requires local --mode train or "
                            "--mode eval-vllm without --env-url"
                        )
                    snapshot = args.corpus_manifest
                elif not remote:
                    snapshot = directory / "snapshot"
                    subprocess.run(
                        base
                        + ["nayana-prepare", "--output", str(snapshot), "--languages"]
                        + args.prepare_languages
                        + [
                            "--pages-per-language",
                            str(2 if args.mode == "real-smoke" else args.prepare_pages),
                            "--max-media-bytes",
                            str(args.prepare_max_bytes),
                        ],
                        check=True,
                    )
                    (run / "manifest.json").write_bytes(
                        (snapshot / "manifest.json").read_bytes()
                    )
                if args.mode == "real-smoke":
                    if remote:
                        parser.error(
                            "Use --mode env-smoke --url for remote smoke checks"
                        )
                    subprocess.run(
                        base
                        + [
                            "nayana-smoke",
                            "--snapshot",
                            str(snapshot),
                            "--output",
                            str(run / "smoke.json"),
                        ]
                        + extra,
                        check=True,
                    )
                else:
                    # Evaluation drives vLLM in an environment of its own, so it needs
                    # the same corpus and the frozen set but none of the training deps.
                    script = (
                        "eval_vllm.py" if args.mode == "eval-vllm" else "grpo_nayana.py"
                    )
                    command = base + [
                        "--extra",
                        "train",
                        "python",
                        str(root / "train" / script),
                        "--output-dir",
                        str(run),
                    ]
                    if args.mode == "eval-vllm":
                        if not args.evalset:
                            parser.error("--mode eval-vllm needs --evalset")
                        command += ["--evalset", str(root / "data" / args.evalset)]
                    if not remote:
                        command += [
                            "--corpus" if args.mode == "eval-vllm" else "--snapshot",
                            str(snapshot),
                        ]
                    subprocess.run(command + extra, check=True)
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
                    commit_message=f"Nayana {args.mode}: {args.revision}",
                )


if __name__ == "__main__":
    main()
