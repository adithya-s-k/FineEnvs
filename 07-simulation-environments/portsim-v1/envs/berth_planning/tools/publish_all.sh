#!/usr/bin/env bash
# Publish PortSimEnv v1 and its article, all public: bucket -> dataset -> environment Space -> collection -> article.
# Needs `hf auth login` with write access to FineEnvs. Every step is idempotent, so a failed run can be re-run.
#
#   bash 07-simulation-environments/portsim-v1/envs/berth_planning/tools/publish_all.sh        # all steps
#   bash 07-simulation-environments/portsim-v1/envs/berth_planning/tools/publish_all.sh 3      # resume from step 3
set -euo pipefail
FROM="${1:-1}"
step() { [ "$1" -ge "$FROM" ]; }
ENV="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$ENV/../../../.." && pwd)"
PY="$ENV/openenv/.venv/bin/python"
DS="$(mktemp -d)/PortSimEnv"

step 1 && echo "== 1/6 bucket FineEnvs/PortSimEnv: 3D twin data, gzipped train pack, eval rollouts"
step 1 && "$PY" "$ENV/tools/publish_bucket.py"

if step 2; then
echo "== 2/6 dataset FineEnvs/PortSimEnv"
(cd "$ENV" && uv run -q --no-project --with pyarrow --with pandas --with-editable ./core python tools/build_dataset.py "$DS")
"$PY" - "$DS" <<'PY'
import sys
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("FineEnvs/PortSimEnv", repo_type="dataset", private=False, exist_ok=True)
api.upload_folder(repo_id="FineEnvs/PortSimEnv", repo_type="dataset", folder_path=sys.argv[1],
                  commit_message="PortSimEnv v1: tasks, source calls, eval rollouts")
print("https://huggingface.co/datasets/FineEnvs/PortSimEnv")
PY
fi

step 3 && echo "== 3/6 environment Space FineEnvs/PortSimEnv (openenv validate + openenv push, bucket mounted at /data)"
step 3 && "$PY" "$ENV/tools/deploy_space.py"

step 4 && echo "== 4/6 collection: Space + dataset, and link it from the article"
step 4 && "$PY" "$ENV/tools/publish_collection.py" --patch-article

step 5 && echo "== 5/6 article Space FineEnvs/simulation-rl-environments"
step 5 && "$PY" -c 'from huggingface_hub import HfApi; HfApi().create_repo("FineEnvs/simulation-rl-environments", repo_type="space", space_sdk="docker", private=False, exist_ok=True)'
step 5 && (cd "$ROOT" && python3 tools/deploy.py content/articles/simulation-rl-environments FineEnvs/simulation-rl-environments)

step 6 && echo "== 6/6 collection: add the article"
step 6 && "$PY" "$ENV/tools/publish_collection.py"
