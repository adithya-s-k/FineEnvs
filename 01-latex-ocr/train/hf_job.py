# /// script
# requires-python = ">=3.11"
# dependencies = ["uv>=0.8,<1"]
# ///
"""Run this repository at an exact revision on HF Jobs (or locally).

hf jobs uv run --flavor cpu-basic --timeout 15m hf_job.py --revision <commit> --mode env-smoke
hf jobs uv run --flavor a10g-small --timeout 30m --secrets HF_TOKEN hf_job.py --revision <commit> --smoke
"""

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
        "--revision", required=True, help="Pushed 40-character Git commit SHA"
    )
    parser.add_argument("--repo", default="adithya-s-k/HuggingEnvs")
    parser.add_argument("--mode", choices=("train", "env-smoke"), default="train")
    args, extra = parser.parse_known_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full commit SHA so the run is reproducible")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        parser.error("--repo must be owner/repository")
    with tempfile.TemporaryDirectory(prefix="huggingenvs-") as directory:
        url = f"https://api.github.com/repos/{args.repo}/zipball/{args.revision}"
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            zipfile.ZipFile(io.BytesIO(response.read())) as archive,
        ):
            archive.extractall(directory)
        root = next(Path(directory).iterdir()) / "01-latex-ocr"
        project = root / "envs" / "latex_ocr"
        base = ["uv", "run", "--frozen", "--project", str(project), "--extra"]
        print(f"Source: {args.repo}@{args.revision}", flush=True)
        if args.mode == "env-smoke":
            subprocess.run(
                base + ["dev", "pytest", str(project / "tests"), "-q"], check=True
            )
            subprocess.run(base + ["dev", "latex-ocr-smoke"] + extra, check=True)
        else:
            subprocess.run(
                base
                + ["train", "python", str(root / "train" / "grpo_latex_ocr.py")]
                + extra,
                check=True,
            )


if __name__ == "__main__":
    main()
