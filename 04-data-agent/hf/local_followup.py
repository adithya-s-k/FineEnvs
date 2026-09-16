"""Local checkpoint evaluations using the qualified trainer's immutable runtime.

The training arm identifies checkpoint provenance. The evaluation suite is separate:
native OpenCode-trained weights run through all four Harbor adapters; SETA-trained
weights use the native bash/SETA evaluator. Neither operation contacts the trainer.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
EVAL_ADMISSION = {"SANDBOX_CAPACITY": "58", "TRAIN_RESERVED_SANDBOXES": "8"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_runtime(plan):
    root = Path(plan["root"])
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan["controller_sha256"]:
        raise ValueError("Checkpoint controller source changed after preparation")
    os.environ["REPRO_ROOT"] = str(root)
    sys.path.insert(0, str(root / "hf/runtime"))
    from common import configure, verify_bundle
    configure()
    verify_bundle()
    manifest = (root / "bundle_manifest.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() != plan["bundle_sha256"]:
        raise ValueError("Qualified runtime identity changed")
    from dotenv import dotenv_values
    values = dotenv_values(plan["env_file"])
    os.environ["HF_TOKEN"] = values.get("HF_API_KEY") or values["HF_TOKEN"]
    for key in ("DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET", "E2B_API_KEY"):
        if values.get(key):
            os.environ[key] = values[key]
    config = read(root / "hf/configs/deployment.json")
    os.environ.update(LOCAL_RUNTIME="1", BUNDLE_SHA256=plan["bundle_sha256"],
        ARTIFACT_BUCKET=config["resources"]["artifacts_bucket"], RUN_ID=config["run_id"],
        COMPARISON_ARM=plan["arm"], HF_HOME=os.environ.get("HF_HOME", str(root / "cache/huggingface")),
        JOB_FLAVOR="hopper-prod-2h100", EVAL_CONCURRENCY="50",
        **EVAL_ADMISSION,
        OPENENV_HARBOR_AGENT_VERSIONS=json.dumps(config["harness_pins"]))
    return root, config


def environment(plan):
    root, _ = load_runtime(plan)
    # Validate the same admission policy used by actual MCP rollouts at startup.
    from service_policy import admission
    if admission.total - admission.train_reserve != 50:
        raise ValueError("Local checkpoint service must admit exactly 50 eval sandboxes")
    from common import RUN
    trials = root / "outputs" / os.environ["RUN_OWNER"] / "trials"
    os.environ.update(ENABLE_WEB_INTERFACE="false", MAX_CONCURRENT_ENVS="128")
    if plan["arm"] in {"blackbox", "opencode"}:
        os.environ.update(COMPARISON_ARM="blackbox", OPENENV_HARBOR_TRIALS_DIR=str(trials),
            OPENENV_DATASETS=str(RUN / "datasets/test"), OPENENV_MODEL="Qwen/Qwen3.5-2B",
            OPENENV_CAPTURE_PORT=os.environ["DATA_AGENT_CAPTURE_PORT"], OPENENV_EXPOSE="gradio",
            OPENENV_MAX_OUTPUT_TOKENS="4096", OPENENV_CAPTURE_TRANSPORT="tunnel")
        from harbor_service import install
        install()
        from harbor_env.server.app import app
        from fastapi import HTTPException

        @app.get("/trial/{name}/result")
        def trial_result(name: str):
            if Path(name).name != name or name in {".", ".."}:
                raise HTTPException(400)
            target = trials / name / "result.json"
            if not target.is_file():
                raise HTTPException(404)
            return read(target)
    else:
        os.environ.update(WHITE_BOX_BASH_TASK_SOURCE="harbor-frozen",
            DAYTONA_WHITEBOX_TRIALS=str(trials), WHITE_BOX_BASH_MAX_SESSIONS="50",
            WHITE_BOX_BASH_MAX_CONCURRENT_ENVS="128")
        from whitebox_bash.server.app import app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["LOCAL_ENV_PORT"]),
                ws_ping_interval=20, ws_ping_timeout=None, timeout_keep_alive=120)


def cleanup(plan, output):
    from concurrent.futures import ThreadPoolExecutor
    from daytona import Daytona, ListSandboxesQuery
    labels = {"experiment": "daytona-harness-comparison", "run": "20260915",
              "arm": "blackbox" if plan["arm"] in {"blackbox", "opencode"} else "whitebox",
              "owner": os.environ["RUN_OWNER"]}
    api = Daytona()
    sandboxes = list(api.list(ListSandboxesQuery(labels=labels), request_timeout=30))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda sandbox: api.delete(sandbox, timeout=60, wait=True), sandboxes))
    remaining = sandboxes
    for _ in range(10):
        remaining = list(api.list(ListSandboxesQuery(labels=labels), request_timeout=15))
        if not remaining:
            break
        time.sleep(3)
    write(output / "cleanup.json", {"labels": labels, "deleted": len(sandboxes), "remaining": len(remaining)})
    if remaining:
        raise RuntimeError("Evaluation-owned sandbox cleanup incomplete")


def validate_evaluation(plan, output, record):
    """Keep a two-task execution qualification distinct from a full pass@1 eval."""
    scores = read(output / "canonical_scores.json")
    qualification = plan.get("checkpoint_eval_qualification", False)
    if qualification:
        suite = "blackbox" if plan["arm"] in {"blackbox", "opencode"} else "whitebox"
        if suite == "blackbox":
            config = read(Path(plan["root"]) / "hf/configs/deployment.json")
            pins = config["harness_pins"]
            audit = read(output / "final_tito.json")
            indices = [int(i) for i in (Path(plan["root"]) /
                "experiments/daytona_harness_comparison/logs/20260915/test_indices.txt").read_text().replace(",", " ").split()[:2]]
            expected = {(h, i) for h in pins for i in indices}
            reports = audit.get("reports", [])
            if ({(r["harness"], r["index"]) for r in reports} != expected or
                    len(reports) != len(expected) or not all(r["tito_pass"] for r in reports)):
                raise ValueError("Checkpoint smoke must audit both tasks through all four harnesses")
            if scores.get("harness_versions") != {h: {v: 2} for h, v in pins.items()}:
                raise ValueError("Checkpoint smoke harness versions differ")
            cells = 8
        else:
            selected = {}
            for line in (output / "attempts.jsonl").read_text().splitlines():
                row = json.loads(line)
                if row.get("reward") in (0, 1) and row.get("tito_pass"):
                    selected.setdefault((row["harness"], row["index"]), row)
            if len(selected) != 2 or any(h != "whitebox_seta" for h, _ in selected):
                raise ValueError("Checkpoint smoke must grade two native SETA trajectories")
            cells = 2
        if scores.get("graded_cells") != cells:
            raise ValueError("Checkpoint smoke coverage mismatch")
    else:
        cells = 1000 if plan["arm"] in {"blackbox", "opencode"} else 250
        if not scores.get("comparison_ready") or scores.get("graded_cells") != cells:
            raise ValueError("Checkpoint evaluation lacks complete graded/TiTO/version evidence")
    proof = {"passed": True, "qualification_only": qualification, "graded_cells": cells,
        "training_arm": plan["arm"], "step": record["step"],
        "manifest_sha256": record["manifest_sha256"], "bundle_sha256": plan["bundle_sha256"],
        "controller_sha256": plan["controller_sha256"]}
    write(output / "checkpoint_eval_verified.json", proof)
    return proof


def evaluate(plan_path, record_path):
    plan, record = read(plan_path), read(record_path)
    root, _ = load_runtime(plan)
    job_id = os.environ["SLURM_JOB_ID"]
    seed = int(job_id) % 1000
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 2:
        raise ValueError("Checkpoint eval requires two separately allocated GPUs")
    owner = f"local-eval-{plan['arm']}-{job_id}"
    os.environ.update(RUN_OWNER=owner, LOCAL_INFERENCE_PORT=str(12000 + seed),
        LOCAL_ENV_PORT=str(14000 + seed), DATA_AGENT_CAPTURE_PORT=str(16000 + seed),
        VLLM_DP_RPC_PORT=str(26000 + seed))
    output = root / "outputs" / owner
    output.mkdir(parents=True, exist_ok=False)
    from common import ENV_PY, ready, start
    from artifacts import Publisher
    from checkpoint_store import restore_model
    from telemetry import Telemetry
    import job
    processes = []
    publisher = Publisher(output)
    publisher.start()
    telemetry = Telemetry(output, job.inference_url())
    telemetry.start()
    status = {"passed": False, "training_arm": plan["arm"], "step": record["step"], "started_at": time.time()}
    try:
        model = output / "inference-model"
        manifest = restore_model(record["checkpoint_prefix"], model, arm=plan["arm"],
            bundle_sha256=plan["bundle_sha256"], manifest_sha256=record["manifest_sha256"])
        if manifest["step"] != record["step"]:
            raise ValueError("Checkpoint step changed after dispatch")
        os.environ["CHECKPOINT_MODEL"] = str(model)
        write(output / "checkpoint_evaluation.json", {**record, "training_arm": plan["arm"],
            "bundle_sha256": plan["bundle_sha256"], "controller_sha256": plan["controller_sha256"]})
        server = "http://127.0.0.1:" + os.environ["LOCAL_ENV_PORT"]
        proc = start([ENV_PY, Path(__file__), "environment", "--plan", plan_path], output / "environment.log")
        processes.append(proc)
        ready(server + "/health", proc, seconds=300)
        suite = "blackbox" if plan["arm"] in {"blackbox", "opencode"} else "whitebox"
        args = SimpleNamespace(role="eval", arm=suite,
            phase="smoke" if plan.get("checkpoint_eval_qualification") else "checkpoint", dp=2, limit=0)
        _, public = job.serving(args, output, processes)
        write(output / "services.json", {"server": server, "inference": public, "tp": 1, "dp": 2,
            "concurrency": 50, "training_job": plan["training_job"], "evaluation_suite": suite})
        job.evaluate(args, output, server, public)
        validate_evaluation(plan, output, record)
        status["passed"] = True
    finally:
        status["finished_at"] = time.time()
        write(output / "status.json", status)
        telemetry.finish()
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        # Use the environment interpreter for Daytona's pinned dependencies.
        try:
            subprocess.run([str(ENV_PY), str(Path(__file__)), "cleanup", "--plan", str(plan_path),
                            "--output", str(output)], check=True, timeout=240)
        finally:
            publisher.finish()


def state(job):
    value = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", str(job),
                                    "--format=JobID,State"], text=True)
    for line in value.splitlines():
        fields = line.split("|")
        if fields[0] == str(job):
            return fields[1].split()[0].rstrip("+")
    return "UNKNOWN"


def eligible_steps(manifests, finished, qualification=False):
    if qualification:
        # The two-phase optimizer smoke saves at2 and4; eval4 exercises the
        # same save/eval2:1 cadence as the main run's50/100 settings.
        return {4} if any(m["step"] == 4 for m in manifests) else set()
    maximum = max((m["step"] for m in manifests), default=0)
    return {m["step"] for m in manifests if m["step"] > 0 and
            (m["step"] % 100 == 0 or m.get("final", False) or finished and m["step"] == maximum)}


def admit_eval(plan):
    # Serialize comparison evals across both arms. Training retains its 24-slot reserve.
    jobs = subprocess.check_output(["squeue", "--noheader", "--user", os.environ["USER"],
                                    "--format=%j"], text=True).splitlines()
    if any(name.strip().startswith("cmp-eval-") for name in jobs):
        return False
    root = Path(plan["root"])
    code = "from daytona import Daytona; print(len(list(Daytona().list())))"
    occupied = int(subprocess.check_output([str(root / "OpenEnv/.venv/bin/python"), "-c", code], text=True).strip())
    return occupied + 50 + 24 <= 125


def watch(plan_path, submit):
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    root, config = load_runtime(plan)
    training = root / "outputs" / f"local-train-{plan['arm']}-{plan['training_job']}"
    out = plan_path.parent / "checkpoint-evals"
    out.mkdir(exist_ok=True)
    lock = (out / "watch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tracking = read(out / "state.json") if (out / "state.json").exists() else {}
    while True:
        training_state = state(plan["training_job"])
        published = read(training / "upload_status.json") if (training / "upload_status.json").exists() else {}
        manifests = []
        for name in published.get("published_checkpoints", []):
            marker = training / "run" / name / "checkpoint.hf.ready.json"
            manifest = read(marker)
            if manifest["arm"] != plan["arm"] or manifest["bundle_sha256"] != plan["bundle_sha256"]:
                raise ValueError("Published checkpoint provenance mismatch")
            manifests.append(manifest)
        eligible = eligible_steps(manifests, training_state in TERMINAL,
                                  qualification=plan.get("checkpoint_eval_qualification", False))
        active, alerts = [], []
        for key, record in tracking.items():
            if not record.get("job_id"):
                raise RuntimeError("Ambiguous submission intent; reconcile Slurm before another submission")
            record["state"] = state(record["job_id"])
            if record["state"] not in TERMINAL:
                active.append(record["job_id"])
            elif record["state"] != "COMPLETED":
                alerts.append(record)
            else:
                eval_output = root / "outputs" / f"local-eval-{plan['arm']}-{record['job_id']}"
                scores_path = eval_output / "canonical_scores.json"
                scores = read(scores_path)
                proof = read(eval_output / "checkpoint_eval_verified.json")
                if (not proof.get("passed") or proof.get("manifest_sha256") != record["manifest_sha256"] or
                        proof.get("controller_sha256") != plan["controller_sha256"] or
                        bool(proof.get("qualification_only")) != bool(plan.get("checkpoint_eval_qualification"))):
                    raise ValueError("Completed evaluation proof differs from its dispatch")
                write(out / f"scores-{record['step']:06d}.json", {"evaluation": record, "scores": scores, "proof": proof})
        pending = sorted(eligible - {r["step"] for r in tracking.values()})
        capacity_lock = (Path(plan["coordination_dir"]) / "eval-admission.lock").open("a")
        fcntl.flock(capacity_lock, fcntl.LOCK_EX)
        if submit and pending and not active and not alerts and admit_eval(plan):
            step = pending[0]
            marker = training / "run" / f"checkpoint-{step}" / "checkpoint.hf.ready.json"
            sha = hashlib.sha256(marker.read_bytes()).hexdigest()
            key = hashlib.sha256(f"{plan['training_job']}:{sha}:{plan['controller_sha256']}".encode()).hexdigest()
            record = {"step": step, "manifest_sha256": sha,
                "checkpoint_prefix": published["destination"] + f"/run/checkpoint-{step}",
                "status": "submitting"}
            record_path = out / f"checkpoint-{step}.json"
            write(record_path, record)
            command = [str(root / ".venv312/bin/python"), "-u", str(Path(__file__).resolve()), "eval",
                       "--plan", str(plan_path), "--record", str(record_path)]
            script = out / f"checkpoint-{step}.slurm"
            script.write_text("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n")
            tracking[key] = record
            write(out / "state.json", tracking)
            # Slurm owns separate GPU devices; the shared admission lock spans inspection and submission.
            result = subprocess.check_output(["sbatch", "--parsable", "--partition=" + plan.get("gpu_partition", "hopper-prod"),
                "--gres=gpu:2", "--cpus-per-task=8", "--mem=96G", "--time=04:00:00",
                f"--job-name=cmp-eval-{plan['arm']}-{step}", f"--output={out}/slurm-%j.out",
                f"--error={out}/slurm-%j.err", str(script)], text=True).strip().split(";")[0]
            if not result.isdigit():
                raise RuntimeError("Ambiguous Slurm submission response")
            record.update(job_id=result, status="submitted")
            active.append(result)
            pending.remove(step)
        capacity_lock.close()
        write(out / "state.json", tracking)
        write(out / "monitor.json", {"checked_at": time.time(), "training_state": training_state,
            "active_eval_jobs": active, "pending_steps": pending, "alerts": alerts})
        if alerts:
            raise RuntimeError("Checkpoint evaluation failed; inspect the recorded job before retrying")
        if not submit or training_state in TERMINAL and not active and not pending:
            return
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["watch", "eval", "environment", "cleanup"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan["controller_sha256"]:
        raise ValueError("Checkpoint controller changed after planning")
    if args.action == "watch":
        watch(args.plan, args.submit)
    elif args.action == "eval":
        evaluate(args.plan, args.record)
    elif args.action == "environment":
        environment(plan)
    else:
        load_runtime(plan)
        cleanup(plan, args.output)
