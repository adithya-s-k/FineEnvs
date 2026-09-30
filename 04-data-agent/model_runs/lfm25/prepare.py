"""Stage pinned source and the original Qwen task manifests without changing them."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
OUT=REPO/'experiments/lfm25-multi4-20260921'
BUNDLE=REPO/'experiments/async_grpo_harbor_data_agent/logs/multi4-hard500-2epochs-from500-20260917/hf-jobs-v11/bundle.tar.gz'
EXPECTED='cc4e3801f9db9958fb52a24ef1c460391a896b2c6cb1737721769f03b23f021a'
if hashlib.sha256(BUNDLE.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Frozen source bundle changed')
OUT.mkdir(parents=True,exist_ok=True)
root=OUT/'runtime'
if not root.exists():
    root.mkdir()
    with tarfile.open(BUNDLE) as archive:archive.extractall(root,filter='data')
for target,source in [(root/'.venv312',REPO/'.venv312'),(root/'OpenEnv/.venv',REPO/'OpenEnv/.venv')]:
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source)
old=REPO/'experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-20260915'
for name in ['harness_schedule.json','indices.txt','manifest.json']:
    target=OUT/name
    if target.exists() and target.read_bytes()!=(old/name).read_bytes():raise ValueError('Staged input differs')
    if not target.exists():shutil.copy2(old/name,target)
versions={k:importlib.metadata.version(k) for k in ['torch','transformers','vllm','huggingface_hub','causal-conv1d','trackio']}
(OUT/'provenance.json').write_text(json.dumps({'bundle_sha256':EXPECTED,'versions':versions,
    'inputs':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['manifest.json','harness_schedule.json','indices.txt']}},indent=2)+'\n')
print(root)
