# /// script
# requires-python = ">=3.11"
# dependencies = ["uv>=0.8,<1"]
# ///
"""Run the CPU data-path benchmark at a pushed commit, returning results in job logs."""

import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import tempfile
import urllib.request
import zipfile
import zlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repo", default="adithya-s-k/HuggingEnvs")
    parser.add_argument("--source-root", default="/corpus")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("Use a pushed 40-character commit SHA")
    if not Path(args.source_root).is_dir():
        parser.error("Attach the corpus bucket before starting this job")
    with tempfile.TemporaryDirectory(prefix="nayana-speed-job-") as temporary:
        folder = Path(temporary)
        url = f"https://codeload.github.com/{args.repo}/zip/{args.revision}"
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for member in archive.infolist():
            if (
                not (folder / member.filename)
                .resolve()
                .is_relative_to(folder.resolve())
            ):
                raise ValueError("Archive path escapes the checkout directory")
        archive.extractall(folder)
        root = (
            folder
            / f"{args.repo.split('/')[-1]}-{args.revision}"
            / "05-multilingual-ocr"
        )
        output = folder / "speed-job.json"
        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "--project",
                str(root / "envs/nayana_ocr"),
                "python",
                str(root / "train/benchmark_corpus.py"),
                "--manifest",
                str(root / "data/corpus-manifest.json"),
                "--source-root",
                args.source_root,
                "--label",
                "hf-job-colocated-mount",
                "--output",
                str(output),
                "--warm-workers",
                "4",
                "--warm-seconds",
                "30",
            ],
            cwd=root,
            check=True,
        )
        result = json.loads(output.read_text())
        result["source_commit"] = args.revision
        raw = json.dumps(result, separators=(",", ":")).encode()
        encoded = base64.b64encode(zlib.compress(raw)).decode()
        print(
            "NAYANA_SPEED_REPORT_BEGIN " + hashlib.sha256(raw).hexdigest(), flush=True
        )
        for index, start in enumerate(range(0, len(encoded), 6000)):
            print(
                f"NAYANA_SPEED_REPORT_PART {index} {encoded[start : start + 6000]}",
                flush=True,
            )
        print("NAYANA_SPEED_REPORT_END", flush=True)


if __name__ == "__main__":
    main()
