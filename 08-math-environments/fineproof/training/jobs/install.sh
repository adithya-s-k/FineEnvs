#!/usr/bin/env bash
# Run from fineproof/training in a Python 3.12 virtual environment. Adapted from 05-multi-harness-rl/jobs/install.sh,
# with TRL and OpenEnv pinned to the commits 05 validated instead of main.
set -euo pipefail
TRL_COMMIT=52fb144e7ece97e26aabd3c025740022838e46b5
# OpenEnv main at the merge of openenv/OpenEnv#1573 (fc2d57a8): `--harness module:Class` (allowed per server with
# OPENENV_HARBOR_ALLOW_HARNESS, which jobs/run.py sets), the capture proxy's loopback URL for host-side agents (the
# FineProof harness), and the harness preflight. The published run used the PR branch: 0bd1a89d until 2026-11-09
# evening, then a0068bfd (smoke-tested as fp-smoke-oe1573); main adds only optional hooks and allow-list trimming.
OPENENV_REPO=${OPENENV_REPO:-https://github.com/openenv/OpenEnv.git}
OPENENV_COMMIT=${OPENENV_COMMIT:-fc2d57a86749a3f4beddc1e8046d3b2db100e7c2}
python -m pip install -r requirements.txt
python -m pip install "trl @ git+https://github.com/huggingface/trl.git@${TRL_COMMIT}"
mkdir -p .deps
if [[ ! -d .deps/OpenEnv/.git ]]; then
    git clone --filter=blob:none "${OPENENV_REPO}" .deps/OpenEnv
fi
git -C .deps/OpenEnv fetch --quiet origin "${OPENENV_COMMIT}" && git -C .deps/OpenEnv checkout --quiet "${OPENENV_COMMIT}"
python -m pip install -e .deps/OpenEnv
# harbor_env (the Harbor server app and HarborSession) is a source package under OpenEnv's envs/.
python - <<'PY'
from importlib.metadata import distribution
from pathlib import Path
import json, site, subprocess
Path(site.getsitepackages()[0], "openenv_examples.pth").write_text(str(Path(".deps/OpenEnv/envs").resolve()) + "\n")
versions = {}
for name in ("openenv", "trl", "transformers", "vllm", "harbor"):
    installed = distribution(name)
    source = json.loads(installed.read_text("direct_url.json") or "{}")
    versions[name] = {"version": installed.version, "commit": source.get("vcs_info", {}).get("commit_id")}
versions["openenv"]["commit"] = subprocess.check_output(["git", "-C", ".deps/OpenEnv", "rev-parse", "HEAD"], text=True).strip()
Path(".deps/revisions.json").write_text(json.dumps(versions, indent=2) + "\n")
PY
# Our harness and the task generator, importable by the Harbor server and by prepare.py.
python -m pip install --no-deps -e harness
python check_setup.py
