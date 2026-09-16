"""Stage an isolated copy of the verified portable runtime and submit it to a two-GPU Slurm allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent


def prepare(bundle, output, env_file, arm, phase, *, partition="hopper-prod", train_venv=None, env_venv=None):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", partition):
        raise ValueError("Invalid Slurm partition name")
    root = output / "repro"
    root.mkdir(parents=True, exist_ok=False)
    info = json.loads((bundle / "bundle.json").read_text())
    if hashlib.sha256((bundle / "bundle.tar.gz").read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("Bundle digest mismatch")
    with tarfile.open(bundle / "bundle.tar.gz") as archive:
        archive.extractall(root, filter="data")
    # The local adapter is orchestration supplied by this launcher. Its exact
    # bytes join the transformed manifest; portable trainer sources stay frozen.
    import shutil
    shutil.copyfile(Path(__file__).parent / "runtime/local_entry.py", root / "hf/runtime/local_entry.py")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh"}:
            text = path.read_text()
            if "/workspace/repro" in text:
                path.write_text(text.replace("/workspace/repro", str(root)))
    (root / "OpenEnv").mkdir(exist_ok=True)
    setup = []
    if bool(train_venv) != bool(env_venv):
        raise ValueError("Pass both existing venv paths, or neither to create locked environments")
    if train_venv:
        for source, target in [(train_venv, root / ".venv312"), (env_venv, root / "OpenEnv/.venv")]:
            if not (Path(source) / "bin/python").is_file():
                raise ValueError("Existing venv has no Python interpreter: " + str(source))
            target.symlink_to(Path(source).resolve(), target_is_directory=True)
    else:
        for name, target in [("env", root / "OpenEnv/.venv"), ("train", root / ".venv312")]:
            setup += [shlex.join(["uv", "venv", "--python", "3.12", str(target)]),
                      shlex.join(["uv", "pip", "sync", "--python", str(target / "bin/python"),
                                  "--require-hashes", str(root / f"hf/locks/requirements-{name}.lock")])]
        setup += [shlex.join(["uv", "pip", "install", "--python", str(root / ".venv312/bin/python"),
                             "--no-deps", "--no-build-isolation", "--editable",
                             str(root / "experiments/daytona_harness_comparison/logs/20260915/source/trl")])]
    # Record every transformed source; the original source snapshot stays untouched.
    manifest = json.loads((root / "bundle_manifest.json").read_text())
    manifest["files"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in manifest["files"]}
    encoded = json.dumps(manifest, sort_keys=True).encode()
    (root / "bundle_manifest.json").write_bytes(encoded)
    local = {"base_bundle": info["sha256"], "sha256": hashlib.sha256(encoded).hexdigest(), "root": str(root)}
    (root / "local_manifest.json").write_text(json.dumps(local, indent=2) + "\n")
    role = "eval" if phase == "baseline" else "train"
    command = [str(root / ".venv312/bin/python"), "-u", str(root / "hf/runtime/local_entry.py"),
               "--role", role, "--arm", arm, "--phase", phase, "--dp", "2" if role == "eval" else "1"]
    script = output / "job.slurm"
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=local-{arm}-{phase}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output={output}/slurm-%j.out
#SBATCH --error={output}/slurm-%j.err
set -euo pipefail
export REPRO_ROOT={shlex.quote(str(root))}
export LOCAL_ENV_FILE={shlex.quote(str(env_file))}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
{chr(10).join(setup)}
exec {shlex.join(command)}
''')
    return script, local


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--env-file", type=Path, required=True)
    p.add_argument("--arm", choices=["blackbox", "opencode", "whitebox"], required=True)
    p.add_argument("--phase", choices=["baseline", "smoke"], required=True)
    p.add_argument("--partition", default="hopper-prod")
    p.add_argument("--train-venv", type=Path)
    p.add_argument("--env-venv", type=Path)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--dependency", help="Slurm dependency, e.g. afterok:80486")
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    script, report = prepare(a.bundle.resolve(), a.out, a.env_file.resolve(), a.arm, a.phase, partition=a.partition, train_venv=a.train_venv, env_venv=a.env_venv)
    if a.submit:
        command = ["sbatch", "--parsable"]
        if a.dependency:
            import re
            if not re.fullmatch(r"afterok:\d+", a.dependency):
                raise ValueError("Use a single afterok Slurm job dependency")
            command += ["--dependency", a.dependency]
        job = subprocess.check_output(command + [str(script)], text=True).strip()
        report["slurm_job"] = job
        report["dependency"] = a.dependency
    (a.out / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
