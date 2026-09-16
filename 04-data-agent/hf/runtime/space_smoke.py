"""Real private-Space task discovery and Daytona tool/verifier checks, without a model."""
from common import RUN, ROOT, ENV_PY, configure, ready, start, write_json
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["blackbox", "whitebox"], required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    configure()
    a.out.mkdir(parents=True, exist_ok=True)
    process = start([ENV_PY, ROOT / "hf/runtime/auth_bridge.py"], a.out / "bridge.log")
    try:
        ready("http://127.0.0.1:8100/health", process, seconds=60)
        if a.arm == "whitebox":
            from native_tool_smoke import one
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda i: one("http://127.0.0.1:8100", i // 2, bool(i % 2), RUN), range(8)))
            report = {"passed": all(r["passed"] for r in results), "cases": results}
        else:
            from openenv.harbor.client import HarborEnv
            from openenv.harbor.tasks import read_instruction
            client = HarborEnv(base_url="http://127.0.0.1:8100")
            checks = []
            try:
                for split, expected in [("train", 1000), ("test", 250)]:
                    spec = "/workspace/repro/experiments/daytona_harness_comparison/logs/20260915/datasets/" + split
                    assert client.num_tasks(spec) == expected
                    manifest = json.loads((RUN / f"{split}_manifest.json").read_text())
                    catalog = sorted(manifest["tasks"], key=lambda row: row["name"])
                    for index in [0, expected - 1]:
                        remote = client.get_task(spec, index)
                        actual = remote.model_dump()
                        local = RUN / "datasets" / split / "tasks" / catalog[index]["name"]
                        assert actual["task_name"] == catalog[index]["name"]
                        assert actual["instruction"] == read_instruction(local)
                    checks.append({"split": split, "tasks": expected, "edge_instructions_match": True})
            finally:
                client.close()
            report = {"passed": True, "cases": checks}
        write_json(a.out / "result.json", report)
        print(json.dumps(report), flush=True)
        if not report["passed"]:
            raise SystemExit(2)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
