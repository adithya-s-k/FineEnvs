"""Release only sandboxes labelled with this unique local Slurm run owner."""
import concurrent.futures
import json
import os
from pathlib import Path
import time

from daytona import Daytona, ListSandboxesQuery

owner = os.environ['RUN_OWNER']
if not owner.startswith('local-') or not owner.endswith('-' + os.environ.get('SLURM_JOB_ID', owner.rsplit('-', 1)[-1])):
    raise ValueError('Expected the unique owner of this local Slurm run')
arm = os.environ['COMPARISON_ARM']
labels = ({'openenv_component': 'blackbox-opencode', 'openenv_owner': owner} if arm == 'opencode'
          else {'experiment': 'daytona-harness-comparison', 'run': '20260915', 'arm': arm, 'owner': owner})
api = Daytona()
selected = list(api.list(ListSandboxesQuery(labels=labels), request_timeout=30))
def delete(sandbox):
    try:
        api.delete(sandbox, timeout=60, wait=True)
        return {'id': sandbox.id, 'deleted': True}
    except Exception as exc:
        return {'id': sandbox.id, 'deleted': False, 'error_type': type(exc).__name__}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(delete, selected))
deadline = time.monotonic() + 30
while True:
    remaining = list(api.list(ListSandboxesQuery(labels=labels), request_timeout=15))
    if not remaining or time.monotonic() >= deadline:
        break
    time.sleep(3)
report = {'owner': owner, 'labels': labels, 'results': results, 'remaining': len(remaining)}
path = Path(os.environ['REPRO_ROOT']) / 'outputs' / owner / 'cleanup.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'owner': owner, 'selected': len(selected), 'remaining': len(remaining)}), flush=True)
if remaining:
    raise RuntimeError('Local owned sandbox cleanup did not complete')
