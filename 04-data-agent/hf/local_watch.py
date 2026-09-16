"""Run the qualified local eval controller and feed completed scores to Trackio."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def admit_with_hf(original, plan, api=None):
    """The caller holds the joint admission lock across inspection/submission."""
    from huggingface_hub import HfApi
    terminal = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"}
    jobs = (api or HfApi()).list_jobs(namespace=plan.get("namespace", "HuggingEnvs"),
        labels={"experiment": "data-agent-daytona", "role": "eval"})
    if any(job.status.stage not in terminal for job in jobs):
        return False
    return original(plan)


def dispatch(plan_path):
    """Wrap admission only; keep the live-qualified evaluator source unchanged."""
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan["supervisor_sha256"]:
        raise ValueError("Score supervisor changed after preparation")
    spec = importlib.util.spec_from_file_location("qualified_local_followup", plan_path.parent / "local_followup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.admit_eval
    module.admit_eval = lambda p: admit_with_hf(original, p)
    module.watch(plan_path, True)


def forward_scores(plan_path):
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text())
    training = Path(plan["root"]) / "outputs" / f"local-train-{plan['arm']}-{plan['training_job']}"
    destination = training / "checkpoint-scores"
    destination.mkdir(parents=True, exist_ok=True)
    changed = False
    if plan.get("baseline_sha256"):
        encoded = Path(plan["baseline_score"]).read_bytes()
        if hashlib.sha256(encoded).hexdigest() != plan["baseline_sha256"]:
            raise ValueError("Qualified baseline score changed")
        event = {"step": 0, "scores": json.loads(encoded),
                 "source": {"baseline_sha256": plan["baseline_sha256"]}}
        target = destination / "step-000000.json"
        if target.exists() and json.loads(target.read_text()) != event:
            raise ValueError("Conflicting baseline for the training curve")
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(event, indent=2) + "\n")
            temporary.replace(target)
            changed = True
    for path in (plan_path.parent / "checkpoint-evals").glob("scores-*.json"):
        record = json.loads(path.read_text())
        proof, evaluation, scores = record["proof"], record["evaluation"], record["scores"]
        if proof.get("qualification_only"):
            continue
        if (not proof.get("passed") or not scores.get("comparison_ready") or
                proof.get("training_arm") != plan["arm"] or
                proof.get("graded_cells") != (1000 if plan["arm"] in {"opencode", "blackbox"} else 250) or
                proof["bundle_sha256"] != plan["bundle_sha256"] or
                proof["manifest_sha256"] != evaluation["manifest_sha256"] or
                proof["step"] != evaluation["step"]):
            raise ValueError("Unverified checkpoint scores cannot enter the training curve")
        event = {"step": evaluation["step"], "scores": scores,
            "source": {"job_id": evaluation["job_id"], "manifest_sha256": proof["manifest_sha256"]}}
        target = destination / f"step-{evaluation['step']:06d}.json"
        if target.exists():
            if json.loads(target.read_text()) != event:
                raise ValueError("Conflicting completed scores for the same checkpoint")
            continue
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(event, indent=2) + "\n")
        temporary.replace(target)
        changed = True
    return plan, training, changed


def watch(plan_path):
    from dotenv import dotenv_values
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    root = Path(plan["root"])
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != plan["supervisor_sha256"]:
        raise ValueError("Score supervisor changed after preparation")
    python = root / ".venv312/bin/python"
    values = dotenv_values(plan["env_file"])
    os.environ["HF_TOKEN"] = values.get("HF_API_KEY") or values["HF_TOKEN"]
    process = subprocess.Popen([str(python), "-u", str(Path(__file__).resolve()),
                               "--dispatch", "--plan", str(plan_path)])
    while True:
        _, training, changed = forward_scores(plan_path)
        late_synced = False
        if (training / "trackio-stop").exists():
            # The training collector has stopped. Replay late eval metrics once
            # on CPU, using the exact logger shipped with the qualified runtime.
            status = json.loads((training / "status.json").read_text())
            version = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (training / "checkpoint-scores").glob("step-*.json")}
            receipt = plan_path.parent / "late-trackio-scores.json"
            imported = json.loads(receipt.read_text()) if receipt.exists() else {}
            if status.get("finished_at") and version and version != imported:
                environment = {**os.environ, "REPRO_ROOT": str(root), "RUN_OWNER": training.name,
                    "BUNDLE_SHA256": plan["bundle_sha256"], "TRACKIO_DIR": str(training / "trackio"),
                    "JOB_FLAVOR": "hopper-prod-2h100"}
                subprocess.run([str(python), str(root / "hf/runtime/logging_sync.py"),
                    "--out", str(training), "--arm", plan["arm"]], env=environment, check=True)
                receipt.write_text(json.dumps(version, indent=2) + "\n")
                late_synced = True
        if changed or late_synced:
            from huggingface_hub import HfApi
            config = json.loads((root / "hf/configs/deployment.json").read_text())
            destination = ("hf://buckets/" + config["resources"]["artifacts_bucket"] + "/" +
                config["run_id"] + "/jobs/" + training.name)
            HfApi().sync_bucket(str(training), destination, quiet=True,
                include=["checkpoint-scores/**", "trackio-events.jsonl", "trackio-backup/**", "trackio_verified.json"])
        code = process.poll()
        if code is not None:
            if code:
                raise RuntimeError(f"Checkpoint controller exited with status {code}")
            return
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    (dispatch if args.dispatch else watch)(args.plan)
