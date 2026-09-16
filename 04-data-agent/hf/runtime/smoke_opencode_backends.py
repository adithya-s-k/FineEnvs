"""Exercise the standalone sandbox protocol, deleting only sandboxes created by this invocation."""
import argparse
import concurrent.futures
import json
import time
from common import configure, write_json
configure()
from data_agent_env.sandbox import build_backend, DEFAULT_IMAGE


def check(name):
    start=time.monotonic();sandbox=None;result={"backend":name}
    try:
        sandbox=build_backend(name,image=DEFAULT_IMAGE).create(timeout_s=600,metadata={"purpose":"protocol-smoke"})
        result["sandbox_id"]=sandbox.sandbox_id
        content='literal `text` $(echo example)\nhello'
        sandbox.write_text('/tmp/protocol/test file.txt',content)
        assert sandbox.exists('/tmp/protocol/test file.txt')
        assert sandbox.read_text('/tmp/protocol/test file.txt')==content
        command=sandbox.exec('printf "%s" "$PROTOCOL_TRANSIENT"',envs={'PROTOCOL_TRANSIENT':'fixture-value'})
        assert command.exit_code==0 and command.stdout=='fixture-value'
        assert sandbox.exec('test -z "$PROTOCOL_TRANSIENT"').exit_code==0
        assert sandbox.start_bg('sleep 1; exit 7').wait(timeout=30)==7
        result['passed']=True
    except Exception as exc:
        result.update(passed=False,error_type=type(exc).__name__)
    finally:
        if sandbox is not None:
            try:sandbox.kill();result['deleted']=True
            except Exception as exc:result.update(passed=False,cleanup_error_type=type(exc).__name__)
    result['elapsed_s']=time.monotonic()-start
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backends',default='daytona,hf');parser.add_argument('--out',required=True)
    args=parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(check,args.backends.split(',')))
    write_json(args.out,records);print(json.dumps(records))
    if not all(r['passed'] and r.get('deleted') for r in records):raise SystemExit(1)
