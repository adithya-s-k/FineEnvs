"""Real HTTP/MCP bash/SETA contract smoke; oracle answers are never sent to a model."""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path


def one(server, index, correct, run):
    from harbor.models.task.task import Task
    from whitebox_bash import white_box_bash_env
    manifest = json.loads((run / 'test_manifest.json').read_text())
    native = Task(run / 'datasets/test/tasks' / manifest['tasks'][index]['name'])
    env = white_box_bash_env(server, toolsets='bash,seta', step_limit=30)()
    checks = {}
    start = time.monotonic()
    try:
        prompt = env.reset(split='test', index=index)
        checks['exact_task_instruction'] = prompt == (native.paths.task_dir / 'instruction.md').read_text()
        checks['write'] = '[error]' not in env.write(path='contract.txt', content='alpha\nbeta\n')
        checks['read'] = 'alpha\nbeta' in env.read(path='contract.txt')
        checks['edit'] = '[error]' not in env.edit(path='contract.txt', old='beta', new='gamma')
        checks['bash_same_filesystem'] = 'gamma' in env.bash(command='cat contract.txt')
        checks['grep'] = 'gamma' in env.grep(pattern='gamma', path='contract.txt')
        checks['glob'] = 'contract.txt' in env.glob(pattern='contract.*')
        checks['ls'] = 'contract.txt' in env.ls(path='.')
        checks['nonzero_command_preserved'] = '7' in env.bash(command='exit 7')
        answer = native.config.verifier.env['EXPECTED_ANSWER'] if correct else '__known_wrong_contract_answer__'
        env.submit_solution(answer=answer)
        reward = env.get_reward()
        checks['frozen_verifier'] = reward == float(correct)
        return {'index':index, 'expected_correct':correct, 'checks':checks,
                'passed':all(checks.values()), 'elapsed_s':time.monotonic()-start}
    except Exception as exc:
        return {'index':index, 'passed':False, 'checks':checks, 'error_type':type(exc).__name__}
    finally:
        if env._session is not None:
            try:
                env.get_reward()
            except Exception:
                pass
        env._mcp.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--server',required=True)
    p.add_argument('--concurrency',type=int,default=4)
    args=p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results=list(pool.map(lambda x: one(args.server,x//2,bool(x%2),args.run),range(8)))
    report={'passed':all(r['passed'] for r in results),'expected':8,'results':results}
    (args.run/'whitebox_tools_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
