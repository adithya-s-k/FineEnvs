#!/usr/bin/env python
"""Baseline launch checks, using Harbor's existing task, sandbox and TiTO code."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def verify_dataset(run):
    manifest = json.loads((run / "manifest.json").read_text())
    split = Path(manifest["split"])
    names = sorted(p.name for p in (split / "tasks").iterdir() if (p / "task.toml").is_file())
    assert names == [t["name"] for t in manifest["tasks"]] and len(names) == 250
    for task in manifest["tasks"]:
        for rel, expected in task["file_hashes"].items():
            if rel.endswith("/task.toml"):
                expected = task["effective_task_toml_sha256"]
            assert hashlib.sha256((split / rel).read_bytes()).hexdigest() == expected, rel
    assert sorted(map(int, re.split(r"[,\s]+", (run / "indices.txt").read_text().strip()))) == list(range(250))
    return manifest


def ready(url, pid, timeout=1800):
    import httpx
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        os.kill(pid, 0)
        try:
            response = httpx.get(url, timeout=3)
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            time.sleep(2)
    raise TimeoutError(f"startup timeout: {url}")


def preflight(run, logs):
    import httpx
    from openenv.harbor.client import HarborEnv
    from openenv.harbor.tasks import read_instruction

    manifest = verify_dataset(run)
    engine = os.environ["VLLM_URL"]
    server = os.environ["SERVER_URL"]
    with httpx.Client(timeout=120) as client:
        models = client.get(engine + "/models").raise_for_status().json()
        assert manifest["model"] in [m["id"] for m in models["data"]], models
        response = client.post(engine + "/chat/completions", json={
            "model": manifest["model"], "messages": [{"role": "user", "content": "Reply PONG."}],
            "max_tokens": 16, "temperature": 0.8, "top_p": 1.0, "top_k": -1,
            "logprobs": True, "top_logprobs": 1, "return_token_ids": True,
        }).raise_for_status().json()
        choice = response["choices"][0]
        assert response.get("prompt_token_ids") and choice.get("token_ids"), response
        assert len(choice["token_ids"]) == len(choice["logprobs"]["content"])
        capture = f"http://127.0.0.1:{os.environ['CAPTURE_PORT']}"
        local = client.get(capture + "/health").raise_for_status().json()
        match = re.search(r"capture\s+:\d+\s+->\s+(https://\S+)", (logs / "openenv.log").read_text())
        assert match, "public capture URL missing"
        public = match[1]
        remote = client.get(public + "/health").raise_for_status().json()
        assert local.get("instance") and local["instance"] == remote.get("instance")
    env = HarborEnv(base_url=server)
    try:
        assert env.num_tasks(manifest["split"]) == 250
        refs = env.get_task_range(manifest["split"], 0, 250)
        assert len(refs) == 250
        for ref, task in zip(refs, manifest["tasks"], strict=True):
            task_dir = Path(manifest["split"]) / "tasks" / task["name"]
            assert ref.task_id == str(task_dir) and ref.instruction == read_instruction(task_dir)
    finally:
        env.close()
    vllm_log = Path(os.environ["VLLM_LOG"]).read_text()
    assert "data_parallel_size=2" in vllm_log or "'data_parallel_size': 2" in vllm_log
    assert "EngineCore_DP0" in vllm_log and "EngineCore_DP1" in vllm_log
    save(logs / "preflight.json", {"task_count": 250, "file_hashes_verified": True,
         "vllm_url": engine, "server_url": server, "capture_url": public,
         "capture_instance": local["instance"], "dp_ranks": [0, 1], "tp": 1,
         "models": models, "token_probe": response})
    print("Preflight passed: fixed 250 tasks, two DP ranks, token IDs/logprobs, capture tunnel", flush=True)


async def warm(run, logs):
    from harbor.environments.e2b import E2BEnvironment
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths

    manifest = verify_dataset(run)
    task = Task(Path(manifest["split"]) / "tasks" / manifest["tasks"][0]["name"])
    directory = logs / "trials" / f"warm-template-{logs.name}"
    directory.mkdir(parents=True, exist_ok=True)
    env = E2BEnvironment(environment_dir=task.paths.environment_dir,
        environment_name=os.environ["HARBOR_SHARED_ENV_NAME"], session_id=directory.name + "__env",
        trial_paths=TrialPaths(trial_dir=directory), task_env_config=task.config.environment)
    start = time.monotonic()
    try:
        await env.start(force_build=False)
        save(logs / "template.json", {"alias": env._template_name, "elapsed_s": time.monotonic() - start})
        print("Shared sandbox template ready:", env._template_name, flush=True)
    finally:
        await env.stop(delete=True)


def audit_captures(run, logs, phase):
    from openenv.harbor.models import HarborRolloutResult
    from smoke_multiharness_tito import audit

    manifest = verify_dataset(run)
    selected = {}
    for path in sorted((logs / "traces").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("reward") is not None and row.get("n_turns", 0) > 0:
                selected.setdefault((row["harness"], row["index"], row["rep"]), row)
    reports = []
    for key, row in selected.items():
        try:
            result = HarborRolloutResult.model_validate_json(Path(row["capture_file"]).read_text())
            report, _ = audit(result, token_budget=131072)
            reports.append({"harness": key[0], "index": key[1], "reward": row["reward"], **report})
        except Exception as exc:
            reports.append({"harness": key[0], "index": key[1], "tito_pass": False,
                            "error": f"{type(exc).__name__}: {exc}"})
    counts = {h: sum(r["harness"] == h for r in reports) for h in manifest["harnesses"]}
    passed = all(r["tito_pass"] for r in reports)
    save(logs / f"{phase}_tito.json", {"counts": counts, "tito_pass": passed, "reports": reports})
    if phase == "resume":
        assert all(n >= 2 for n in counts.values()), counts
    else:
        expected = 2 if phase == "smoke" else 250
        assert all(n == expected for n in counts.values()), counts
    assert passed, "TiTO audit failed; inspect per-capture reports"
    print(f"{phase}: {len(reports)} graded evaluations, all exact-token TiTO audits pass", flush=True)


def import_results(run, logs):
    manifest = verify_dataset(run)
    source = Path(os.environ["BASELINE_RESUME_FROM"])
    config = json.loads((source / "traces/eval_config.json").read_text())
    assert config["split"] == manifest["split"]
    assert config["temperature"] == manifest["temperature"] and config["repeat"] == 1
    assert config["agent_step_limit"] == manifest["agent_step_limit"]
    assert config["agent_timeout"] == manifest["agent_timeout_s"]
    assert [arm["harness"] for arm in config["arms"]] == manifest["harnesses"]
    assert all(arm["model"] == manifest["model"] for arm in config["arms"])
    assert sorted(config["indices"]) == list(range(250))
    original = json.loads(json.dumps(config))
    config["server"] = os.environ["SERVER_URL"]
    for arm in config["arms"]:
        arm["base_url"] = os.environ["VLLM_URL"]
    shutil.copytree(source / "traces", logs / "traces")
    save(logs / "traces/eval_config.json", config)
    save(logs / "resume_transport_migration.json", {"source": str(source), "previous": original,
         "current": config, "identity_check": "preflight.json and fixed manifest/model revision"})


def cleanup(logs):
    from e2b import Sandbox, SandboxQuery
    own = {p.name + "__env" for p in (logs / "trials").glob("*") if p.is_dir()}
    query = SandboxQuery(metadata={"environment_name": os.environ["HARBOR_SHARED_ENV_NAME"]})
    def found():
        pager = Sandbox.list(query=query)
        out = []
        while pager.has_next:
            out.extend(s for s in pager.next_items() if s.metadata.get("session_id") in own)
        return out
    removed = []
    for sandbox in found():
        Sandbox.kill(sandbox.sandbox_id)
        removed.append(sandbox.sandbox_id)
    remaining = [s.sandbox_id for s in found()]
    save(logs / "cleanup.json", {"removed": removed, "remaining_owned_sandboxes": remaining})
    assert not remaining, remaining
    print(f"Cleanup: {len(removed)} removed; zero owned sandboxes remain", flush=True)


def metrics(logs):
    import httpx
    with httpx.Client(timeout=5) as client, (logs / "metrics.jsonl").open("a", buffering=1) as file:
        while True:
            start = time.monotonic()
            try:
                response = client.get(os.environ["VLLM_URL"].removesuffix("/v1") + "/metrics")
                response.raise_for_status()
                value = {"metrics": response.text}
            except httpx.HTTPError as exc:
                value = {"error": type(exc).__name__}
            file.write(json.dumps({"time": time.time(), "scrape_s": time.monotonic() - start, **value}) + "\n")
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["dataset", "ready", "preflight", "warm", "smoke", "resume", "import", "final", "cleanup", "metrics"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--url")
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    if args.mode == "ready":
        ready(args.url, args.pid)
    elif args.mode == "dataset":
        verify_dataset(args.run)
    elif args.mode == "preflight":
        preflight(args.run, args.logs)
    elif args.mode == "warm":
        asyncio.run(warm(args.run, args.logs))
    elif args.mode in ("smoke", "resume", "final"):
        audit_captures(args.run, args.logs, args.mode)
    elif args.mode == "import":
        import_results(args.run, args.logs)
    elif args.mode == "cleanup":
        cleanup(args.logs)
    elif args.mode == "metrics":
        metrics(args.logs)
