"""Verify the completed HF qualification and save an unsubmitted main job preview."""
import argparse
import json
from pathlib import Path
import shlex
import sys

from deploy import credentials, save, submit


def finalize(args):
    from huggingface_hub import HfApi
    output = args.out.resolve()
    plan_path = output / "plan.json"
    plan = json.loads(plan_path.read_text())
    record = json.loads((args.controller_output / "decisions/scores/step-000004.json").read_text())
    config = json.loads(Path(plan["config"]).read_text())
    if plan["arm"] != "whitebox" or plan.get("main_training_submitted"):
        raise ValueError("Expected an unsubmitted Whitebox plan")
    secrets = credentials(args.env_file)
    job_args = argparse.Namespace(role="train", arm="whitebox", phase="long", flavor="h200x2",
        timeout="24h", dp=1, limit=0, resume_eval_owner=None, training_job=None,
        baseline_job=plan["baseline_job"], smoke_job=plan["smoke_job"],
        checkpoint_eval_job=record["job_id"], external_checkpoint_coordinator=True, dry_run=True)
    preview = submit(HfApi(token=secrets["HF_TOKEN"]), config, secrets, output, job_args)
    if preview.get("submitted") or not preview.get("proof", {}).get("checkpoint_eval", {}).get("passed"):
        raise RuntimeError("Expected a successful read-only qualification preview")
    hf = Path(__file__).resolve().parent
    command = [sys.executable, str(hf / "deploy.py"), "job", "--env-file", str(args.env_file.resolve()),
        "--config", plan["config"], "--out", str(output), "--role", "train", "--arm", "whitebox",
        "--phase", "long", "--flavor", "h200x2", "--timeout", "24h", "--dp", "1",
        "--baseline-job", plan["baseline_job"], "--smoke-job", plan["smoke_job"],
        "--checkpoint-eval-job", record["job_id"], "--external-checkpoint-coordinator"]
    plan.update(checkpoint_eval_job=record["job_id"], status="qualification_passed_main_not_submitted",
                launch_preview=str(output / "launch-preview.json"), launch_command=command,
                controller_launcher=str(hf / "hf_followup.py"),
                controller_note="After trainer submission, prepare and submit hf_followup.py for its returned job ID")
    save(plan_path, plan)
    (output / "launch-command.txt").write_text(shlex.join(command) + "\n")
    print(json.dumps({"plan": str(plan_path), "qualified": True, "main_training_submitted": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--controller-output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    finalize(parser.parse_args())
