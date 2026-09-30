"""Stop an HF trainer only after verifying its next full checkpoint, then evaluate it.

Uses the existing checkpoint controller and immutable GPU bundle. It never changes
the running trainer, and it preserves an explicit planned-cancellation receipt.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TERMINAL = {"COMPLETED", "ERROR", "CANCELED", "CANCELLED", "DELETED"}
READY = "checkpoint.hf.ready.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_checkpoint(directory, *, job, step):
    manifest = read(directory / READY)
    if (manifest["arm"] != "whitebox" or manifest["step"] != step
            or manifest["bundle_sha256"] != job.environment["BUNDLE_SHA256"]
            or manifest["base_model"] != "Qwen/Qwen3.5-2B"
            or manifest["base_revision"] != "15852e8c16360a2fea060d615a32b45270f8a8fc"):
        raise ValueError("Checkpoint provenance mismatch")
    required = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"}
    if not required.issubset(manifest["files"]) or not any(n.endswith(".safetensors") for n in manifest["files"]):
        raise ValueError("Full resumable training state is missing")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha(directory / name) != expected:
            raise ValueError("Checkpoint content hash mismatch: " + name)
    if read(directory / "trainer_state.json")["global_step"] != step:
        raise ValueError("Checkpoint optimizer step mismatch")
    return manifest


def render(out, state):
    lines = ["# SETA planned stop and final evaluation", "", f"Updated: {now()}", "",
             f"- Training job: `{state['training_job']}`.",
             f"- Target checkpoint: **{state['target_step']}**; phase: **{state['phase']}**.",
             "- User requested the next regular save, then stop and evaluate the synchronous SETA run.",
             "- Training stops only after a local readback verifies every checkpoint file hash, including optimizer/RNG state.",
             "- The provider records CANCELED for this intentional stop; it is not a successful 1,000-step completion.",
             "- Evaluation: the unchanged 250 fixed tests through native bash/SETA, pass@1, concurrency 50, one A100.",
             "- All three environment Spaces use CPU Basic after SETA stops; limits remain 1024 transport sessions and 64/100/61 sandbox slots (Harbor/native OpenCode/SETA).",
             "- CPU Basic preserves configured limits; no claim of equivalent measured peak throughput is made.", ""]
    if state.get("checkpoint_verified"):
        lines += [f"Verified checkpoint manifest: `{state['checkpoint_verified']['manifest_sha256']}`.", ""]
    if state.get("evaluation"):
        result = state["evaluation"]
        score = result["scores"]
        lines += [f"Evaluation job: `{result['job_id']}`. Final pass@1: **{100 * score['average_pass_at_1']:.1f}%**.",
                  f"Complete graded coverage: {score['graded_cells']}/{score['expected_cells']}; TiTO: {score['tito_pass']}.", "",
                  "```json", json.dumps(score, indent=2), "```", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    args = p.parse_args()
    request = read(args.request)
    out = args.request.parent
    lock = (out / "operation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from dotenv import dotenv_values
    from huggingface_hub import HfApi
    from huggingface_hub.errors import EntryNotFoundError
    import httpx
    creds = dotenv_values(request["env_file"])
    os.environ["HF_TOKEN"] = creds.get("HF_API_KEY") or creds["HF_TOKEN"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    state = read(out / "status.json") if (out / "status.json").exists() else {
        "training_job": request["training_job"], "target_step": request["target_step"],
        "user_requested_stop": True, "started_at": now(), "phase": "waiting_for_checkpoint"}

    def progress(phase, **values):
        state.update(phase=phase, checked_at=now(), **values)
        save(out / "status.json", state)
        render(out, state)
        print(json.dumps({k: state[k] for k in ["phase", "checked_at", "target_step"]}), flush=True)

    try:
        job = api.inspect_job(job_id=request["training_job"], namespace="HuggingEnvs")
        assert job.labels["role"] == "train" and job.labels["arm"] == "whitebox"
        bucket = job.environment["ARTIFACT_BUCKET"]
        prefix = job.environment["RUN_ID"] + "/jobs/" + job.environment["RUN_OWNER"]
        cp_prefix = prefix + f"/run/checkpoint-{request['target_step']}"
        checkpoint = out / f"checkpoint-{request['target_step']}"
        checkpoint.mkdir(exist_ok=True)
        deadline = time.monotonic() + 4 * 3600
        if not state.get("checkpoint_verified"):
            while True:
                try:
                    api.download_bucket_files(bucket, [(cp_prefix + "/" + READY, checkpoint / READY)], raise_on_missing_files=True)
                    break
                except EntryNotFoundError:
                    job = api.inspect_job(job_id=request["training_job"], namespace="HuggingEnvs")
                    if job.status.stage in TERMINAL or time.monotonic() > deadline:
                        raise RuntimeError("Trainer ended or checkpoint publication timed out before safe stop")
                    progress("waiting_for_checkpoint", training_stage=job.status.stage)
                    time.sleep(30)
            manifest = read(checkpoint / READY)
            if any(Path(n).name != n for n in manifest["files"]):
                raise ValueError("Invalid checkpoint member")
            progress("verifying_full_checkpoint")
            api.download_bucket_files(bucket, [(cp_prefix + "/" + n, checkpoint / n) for n in manifest["files"]], raise_on_missing_files=True)
            verify_checkpoint(checkpoint, job=job, step=request["target_step"])
            proof = {"training_job": job.id, "step": request["target_step"],
                     "checkpoint": "hf://buckets/" + bucket + "/" + cp_prefix,
                     "manifest_sha256": sha(checkpoint / READY), "full_checkpoint_verified": True,
                     "user_requested_stop": True, "verified_at": now()}
            save(out / "checkpoint-verified.json", proof)
            progress("checkpoint_verified", checkpoint_verified=proof)

        # Freeze and test the final-eval controller before canceling the trainer.
        controller = out / "controller"
        if not (controller / "plan.json").exists():
            old = Path(request["previous_controller_plan"]).parent
            shutil.copytree(old, controller, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("output", "*.out", "*.err", "__pycache__", "plan.json"))
            plan = read(old / "plan.json")
            for key in ["root", "logger_root"]:
                plan[key] = str(controller / Path(plan[key]).name)
            plan["final_checkpoint"] = state["checkpoint_verified"]
            coordinator = Path(plan["root"]) / "hf/runtime/coordinator.py"
            shutil.copy2(out / "source/coordinator.py", coordinator)
            shutil.copy2(out / "source/late_hf_logging.py", controller / "late_hf_logging.py")
            followup = controller / "hf_followup.py"
            code = followup.read_text()
            old_call = 'run(out / "output", "whitebox", admission=admission)'
            assert old_call in code
            followup.write_text(code.replace(old_call, 'run(out / "output", "whitebox", admission=admission, final_checkpoint=plan.get("final_checkpoint"))'))
            (controller / "controller.slurm").write_text((out / "stop.slurm").read_text())
            plan["files"] = {n: sha(controller / n) for n in plan["files"]}
            save(controller / "plan.json", plan)
        if not state.get("training_stopped"):
            # Shut down only the old CPU coordinator; its evaluated checkpoint100 persists.
            subprocess.run(["scancel", str(request["previous_controller_job"])], check=True)
            launch = Path(request["launch_state"])
            metadata = read(launch)
            metadata.update(controller_job=os.environ["SLURM_JOB_ID"], controller_plan=str(controller / "plan.json"),
                            planned_stop={"target_step": request["target_step"], "request": str(args.request), "user_requested": True})
            save(launch, metadata)
            job = api.inspect_job(job_id=job.id, namespace="HuggingEnvs")
            if job.status.stage not in TERMINAL:
                api.cancel_job(job_id=job.id, namespace="HuggingEnvs")
            for _ in range(60):
                job = api.inspect_job(job_id=job.id, namespace="HuggingEnvs")
                if job.status.stage in TERMINAL:
                    break
                time.sleep(5)
            if job.status.stage not in TERMINAL:
                raise RuntimeError("Cancellation did not become terminal")
            progress("training_stopped", training_stopped=True, training_stage=job.status.stage,
                     stopped_at=now(), final_saved_step=request["target_step"])

        receipt = read(out / "space-hardware.json")
        for repo, row in receipt["spaces"].items():
            if not row.get("request_sent"):
                api.request_space_hardware(repo, hardware="cpu-basic")
                row.update(request_sent=True, requested_at=now())
                save(out / "space-hardware.json", receipt)
            for _ in range(60):
                runtime = api.get_space_runtime(repo)
                if runtime.hardware == "cpu-basic" and runtime.stage == "RUNNING":
                    url = "https://" + repo.replace("/", "-").lower() + ".hf.space"
                    response = httpx.get(url + "/deployment", timeout=30)
                    if response.status_code == 200:
                        current = response.json()
                        variables = api.get_space_variables(repo)
                        assert all(variables[k].value == v for k, v in row["concurrency_variables"].items())
                        assert current["bundle_sha256"] == row["bundle_sha256"]
                        row.update(verified_at=now(), hardware=runtime.hardware, stage=runtime.stage, after_deployment=current,
                                   concurrency_unchanged=True)
                        save(out / "space-hardware.json", receipt)
                        break
                time.sleep(10)
            else:
                raise RuntimeError("Space did not become healthy on CPU Basic: " + repo)
        progress("final_evaluation_controller_running", spaces_cpu_basic=True)
        subprocess.run([sys.executable, "-u", str(controller / "hf_followup.py"), "watch", "--plan", str(controller / "plan.json")], check=True)
        score_path = controller / "output/decisions/scores" / f"step-{request['target_step']:06d}.json"
        result = read(score_path)
        if not result["scores"].get("comparison_ready"):
            raise RuntimeError("Final evaluation did not pass its gates")
        progress("complete", evaluation=result, completed_at=now())
    except Exception as exc:
        progress("needs_attention", error_type=type(exc).__name__, error=str(exc)[:500])
        raise


if __name__ == "__main__":
    main()
