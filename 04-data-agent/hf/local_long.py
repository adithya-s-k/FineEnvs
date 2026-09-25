"""Prepare a local long run using the exact runtime that passed optimizer qualification.

No jobs are submitted unless --submit is provided. Smoke and baseline evidence are
required even for planning. The CPU checkpoint controller is a separately frozen
orchestration artifact; it does not change the qualified trainer source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def proof_for(smoke, arm):
    root = smoke.parent.parent
    proof = json.loads((smoke / "training_smoke_verified.json").read_text())
    identity = json.loads((root / "local_manifest.json").read_text())
    if (proof.get("arm") != arm or proof.get("bundle_sha256") != identity["sha256"] or
            not all(proof.get(k) for k in ["passed", "remote_restore_verified", "tito_pass", "weights_updated", "native_optimizer_state_verified"])):
        raise ValueError("No matching successful optimizer/save/remote-resume proof")
    encoded = (root / "bundle_manifest.json").read_bytes()
    if hashlib.sha256(encoded).hexdigest() != identity["sha256"]:
        raise ValueError("Qualified source manifest changed")
    for name, expected in json.loads(encoded)["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Qualified source file changed: {name}")
    return root, identity


def validate_baseline(path, arm):
    score = json.loads(path.read_text())
    if arm == "opencode":
        score = score.get("daytona", score)
        valid = score.get("implementation") == "standalone-opencode"
    else:
        valid = score.get("arm") == arm
    if not (valid and score.get("comparison_ready") and score.get("graded_cells") == (1000 if arm == "blackbox" else 250) and score.get("tito_pass")):
        raise ValueError("A completed matching fixed-task baseline is required")


def validate_native_grading(root, baseline):
    """Bind the native runtime to the frozen-tolerance and deterministic rescore audit."""
    proof_path = baseline.parent / "verification.json"
    if not proof_path.is_file():
        raise ValueError("Native training requires the explicit-tolerance baseline verification")
    proof = json.loads(proof_path.read_text())
    if (not proof.get("passed") or proof.get("parameters_verified") != 1250 or
            not proof.get("original_graded_records_preserved") or len(proof.get("reports", [])) != 250):
        raise ValueError("Incomplete native grading/data verification")
    package = root / "experiments/daytona_harness_comparison/logs/20260915/source/packages/data_agent_env"
    for name in ("task.py", "tasks.py", "verifier.py", "grader.py"):
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != proof["runtime_files"][name]:
            raise ValueError("Native grader differs from the verified frozen-tolerance protocol: " + name)
    return str(proof_path.resolve())


def prepare(args):
    import re
    for name in ("partition", "cpu_partition"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", getattr(args, name, "hopper-prod")):
            raise ValueError("Invalid Slurm partition name")
    smoke, output = args.smoke_run.resolve(), args.out.resolve()
    root, identity = proof_for(smoke, args.arm)
    validate_baseline(args.baseline_score, args.arm)
    grading_proof = validate_native_grading(root, args.baseline_score) if args.arm == "opencode" else None
    config = json.loads((root / "hf/configs/deployment.json").read_text())
    comparison = args.baseline_score
    if args.arm == "opencode":
        from launch_gates import protocol_identity, validate_comparison_baseline
        comparison = getattr(args, "comparison_baseline_score", None)
        if comparison is None:
            raise ValueError("Native training also requires --comparison-baseline-score for the four-harness curve")
        validate_comparison_baseline(json.loads(comparison.read_text()))
        baseline_root = comparison.resolve().parents[2]
        baseline_config = json.loads((baseline_root / "hf/configs/deployment.json").read_text())
        if protocol_identity(config) != protocol_identity(baseline_config):
            raise ValueError("Comparison baseline task/model/sampling protocol differs")
    output.mkdir(parents=True, exist_ok=False)
    controller = output / "local_followup.py"
    shutil.copyfile(Path(__file__).with_name("local_followup.py"), controller)
    supervisor = output / "local_watch.py"
    supervisor_source = Path(__file__).with_name("local_watch.py")
    if supervisor_source.exists():
        shutil.copyfile(supervisor_source, supervisor)
    args.coordination_dir.mkdir(parents=True, exist_ok=True)
    plan = {"gpu_partition": getattr(args, "partition", "hopper-prod"),
        "namespace": config["namespace"],
        "cpu_partition": getattr(args, "cpu_partition", "hopper-cpu"), "root": str(root), "arm": args.arm, "bundle_sha256": identity["sha256"],
        "env_file": str(args.env_file.resolve()), "baseline_score": str(comparison.resolve()),
        "baseline_sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
        "native_baseline_score": str(args.baseline_score.resolve()) if args.arm == "opencode" else None,
        "smoke_run": str(smoke), "controller_sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
        "grading_verification": grading_proof,
        "coordination_dir": str(args.coordination_dir.resolve()), "training_job": None,
        "save_steps": 50, "eval_steps": 100, "eval_concurrency": 50, "eval_dp": 2,
        "eval_cells": 1000 if args.arm in {"blackbox", "opencode"} else 250,
        "long_training_submitted": False, "checkpoint_eval_gpu_validation": "pending"}
    if supervisor.exists():
        plan["supervisor_sha256"] = hashlib.sha256(supervisor.read_bytes()).hexdigest()
    if args.checkpoint_eval_proof:
        evaluation = json.loads(args.checkpoint_eval_proof.read_text())
        if (not evaluation.get("passed") or not evaluation.get("qualification_only") or
                evaluation.get("bundle_sha256") != identity["sha256"] or
                evaluation.get("controller_sha256") != plan["controller_sha256"] or
                evaluation.get("training_arm") != args.arm or evaluation.get("step") != 4 or
                evaluation.get("graded_cells") != (8 if args.arm in {"blackbox", "opencode"} else 2)):
            raise ValueError("No matching successful checkpoint-4 evaluation qualification")
        plan["checkpoint_eval_gpu_validation"] = "passed"
        plan["checkpoint_eval_proof"] = str(args.checkpoint_eval_proof.resolve())
    if args.qualify_checkpoint_eval:
        plan["training_job"] = smoke.name.rsplit("-", 1)[-1]
        if not plan["training_job"].isdigit():
            raise ValueError("Local smoke owner does not identify its Slurm job")
        plan["checkpoint_eval_qualification"] = True
        plan["eval_cells"] = 8 if args.arm in {"blackbox", "opencode"} else 2
    destination = ("hf://buckets/" + config["resources"]["artifacts_bucket"] + "/" + config["run_id"]
                   + "/jobs/" + smoke.name)
    command = [str(root / ".venv312/bin/python"), "-u", str(root / "hf/runtime/local_entry.py"),
               "--role", "train", "--arm", args.arm, "--phase", "long", "--dp", "1"]
    script = output / "train.slurm"
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=local-{args.arm}-long
#SBATCH --partition={plan["gpu_partition"]}
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output={output}/train-%j.out
#SBATCH --error={output}/train-%j.err
set -euo pipefail
export REPRO_ROOT={shlex.quote(str(root))}
export LOCAL_ENV_FILE={shlex.quote(str(args.env_file.resolve()))}
export SMOKE_PREFIX={shlex.quote(destination)}
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec {shlex.join(command)}
''')
    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    if args.submit:
        if args.qualify_checkpoint_eval or plan["checkpoint_eval_gpu_validation"] != "passed" or not supervisor.exists():
            raise RuntimeError("Long launch requires optimizer and live checkpoint-eval qualification; plan is saved")
        plan["submission_state"] = "submitting"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        job = subprocess.check_output(["sbatch", "--parsable", str(script)], text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError("Ambiguous training submission; reconcile Slurm before retrying")
        plan.update(training_job=job, long_training_submitted=True, submission_state="submitted")
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        command = [str(root / ".venv312/bin/python"), "-u", str(supervisor), "--plan", str(plan_path)]
        cpu = output / "controller.slurm"
        cpu.write_text("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n")
        controller_job = subprocess.check_output(["sbatch", "--parsable", "--partition=" + plan["cpu_partition"],
            "--cpus-per-task=2", "--mem=8G", "--time=36:00:00", f"--job-name=cmp-follow-{args.arm}",
            f"--output={output}/controller-%j.out", f"--error={output}/controller-%j.err", str(cpu)], text=True).strip()
        plan["controller_job"] = controller_job
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan": str(plan_path), "training_script": str(script), "submitted": args.submit}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-run", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--comparison-baseline-score", type=Path)
    parser.add_argument("--arm", choices=["blackbox", "whitebox", "opencode"], required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coordination-dir", type=Path, required=True)
    parser.add_argument("--partition", default="hopper-prod")
    parser.add_argument("--cpu-partition", default="hopper-cpu")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--checkpoint-eval-proof", type=Path,
                        help="Passing live checkpoint-4 evaluation proof required before --submit")
    parser.add_argument("--qualify-checkpoint-eval", action="store_true",
                        help="Follow the completed smoke's checkpoint with a separate two-task eval job")
    prepare(parser.parse_args())
