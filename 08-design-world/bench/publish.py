"""Publish a DesignWorldBench v2 run to the Hub, in the cua-v1 format the DesignWorld-SFT data uses.

    python bench/publish.py results/v2/<run>                 # converts and uploads to the private results repo
    python bench/publish.py results/v2/<run> --dry-run       # counts only
    python bench/publish.py results/v2/<run> --repo <org>/<name> --config <name>

Each run becomes one config (`data/<run>/test-<model>-*.parquet`: one row per episode with the full conversation,
every screenshot, the grade, the submitted document and the episode's workspace), plus `meta/<run>.json` (counts and
the run's summary) and `runs/<run>/` (the raw trace, settings and summary). The repo is created private if it does
not exist. Needs HF_TOKEN with write access to the org.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = os.environ.get("DESIGNWORLD_RESULTS_REPO", "perpetual-research/designworldbench-runs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--config", default=None, help="config name (default: the run directory's name)")
    ap.add_argument("--evalset", type=Path, default=HERE / "evalsets" / "recreation-cua-v2.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-incomplete", action="store_true")
    a = ap.parse_args()
    cmd = [sys.executable, str(HERE / "harness" / "to_cua.py"), *map(str, a.runs), "--evalset", str(a.evalset)]
    if a.config:
        cmd += ["--config", a.config]
    if a.keep_incomplete:
        cmd += ["--keep-incomplete"]
    if a.dry_run:
        return subprocess.call(cmd + ["--dry-run"])
    with tempfile.TemporaryDirectory() as out:
        return subprocess.call(cmd + ["--out", out, "--push", a.repo])


if __name__ == "__main__":
    raise SystemExit(main())
