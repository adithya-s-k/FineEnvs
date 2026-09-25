"""Delete only this comparison's owned Daytona sandboxes after the owning job exits."""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from daytona import Daytona, ListSandboxesQuery


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--legacy-baseline',action='store_true')
    group.add_argument('--owner')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[3]
    load_dotenv(repo/'experiments/.env')
    client=Daytona()
    labels={'experiment':'daytona-harness-comparison','run':'20260915'}
    if args.owner: labels['owner']=args.owner
    sandboxes=list(client.list(ListSandboxesQuery(labels=labels),request_timeout=30))
    if args.legacy_baseline:
        sandboxes=[s for s in sandboxes if 'owner' not in s.labels]
    def remove(sandbox):
        try:
            client.delete(sandbox,timeout=90,wait=True)
            return {'sandbox_id':sandbox.id,'deleted':True}
        except Exception as exc:
            return {'sandbox_id':sandbox.id,'deleted':False,'error_type':type(exc).__name__}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(remove,sandboxes))
    # A successful wait=True delete can precede the list index update. Verify
    # disappearance with a bounded poll before reporting a leaked sandbox.
    deadline=time.monotonic()+60
    while True:
        remaining=list(client.list(ListSandboxesQuery(labels=labels),request_timeout=30))
        if args.legacy_baseline: remaining=[s for s in remaining if 'owner' not in s.labels]
        if not remaining or time.monotonic()>=deadline: break
        time.sleep(5)
    report={'labels':labels,'legacy_baseline':args.legacy_baseline,'results':results,'remaining':len(remaining)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'selected':len(sandboxes),'remaining':len(remaining)}))
    return 2 if remaining else 0


if __name__=='__main__': raise SystemExit(main())
