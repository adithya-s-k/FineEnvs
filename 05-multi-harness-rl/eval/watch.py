"""Hash finished checkpoints and submit evaluations on separate GPU allocations."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recipe import write_json
from runtime.checkpoints import make_ready


def scan(root):
    for request in sorted(Path(root).glob("*/checkpoint-*/eval.request.json")):
        checkpoint = request.parent
        if not (checkpoint / "checkpoint.ready.json").exists():
            try:
                make_ready(checkpoint)
            except (OSError, ValueError) as exc:
                print(json.dumps({"checkpoint": str(checkpoint), "status": "waiting_for_complete_files",
                                  "error": type(exc).__name__}), flush=True)
                continue
        if json.loads(request.read_text())["evaluate"]:
            yield checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--backend", choices=["hf", "slurm"], required=True)
    p.add_argument("--bucket")
    p.add_argument("--namespace", default="FineEnvs")
    p.add_argument("--partition")
    p.add_argument("--flavor", default="h200x2")
    p.add_argument("--data", type=Path, default=ROOT / "prepared")
    p.add_argument("--concurrency", type=int, default=35)
    p.add_argument("--max-active", type=int, default=1)
    p.add_argument("--once", action="store_true")
    p.add_argument("--submit", action="store_true")
    args = p.parse_args()
    args.root = args.root.resolve()
    if args.backend == "hf" and not args.bucket:
        p.error("--bucket is required for HF Jobs")
    if args.backend == "slurm" and not args.partition:
        p.error("--partition is required for Slurm")
    if args.max_active < 1 or args.concurrency < 1:
        p.error("Concurrency and max-active must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    state_path = args.root / "eval-watcher.json"
    with (args.root / "eval-watcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            ledger = json.loads(state_path.read_text()) if state_path.exists() else {}
            for key, entry in ledger.items():
                if entry["status"] != "active":
                    continue
                if args.backend == "hf":
                    from huggingface_hub import inspect_job
                    stage = inspect_job(job_id=entry["id"], namespace=args.namespace).status.stage
                    if stage in {"COMPLETED", "ERROR", "CANCELED", "DELETED"}:
                        entry["status"] = "done" if stage == "COMPLETED" else "failed"
                else:
                    status = subprocess.check_output(["sacct", "-j", entry["id"], "--noheader", "--format=State", "--parsable2"], text=True).splitlines()
                    if status and status[0] not in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}:
                        entry["status"] = "done" if status[0] == "COMPLETED" else "failed"
            active = sum(e["status"] == "active" for e in ledger.values())
            for checkpoint in scan(args.root):
                key = str(checkpoint.relative_to(args.root))
                if key in ledger or active >= args.max_active:
                    continue
                cfg = json.loads((checkpoint / "recipe.json").read_text())
                name = f'{checkpoint.parent.name}-eval-{checkpoint.name}'
                cmd = [sys.executable, str(ROOT / "runtime/launch.py"), args.backend, "eval",
                       "--mode", cfg["mode"], "--model", cfg["model"], "--run-name", name,
                       "--checkpoint", str(checkpoint), "--output-root", str(args.root),
                       "--config", str(checkpoint / "recipe.json"),
                       "--data", str(args.data), "--concurrency", str(args.concurrency)]
                if args.backend == "hf":
                    cmd += ["--bucket", args.bucket, "--namespace", args.namespace, "--flavor", args.flavor]
                else:
                    cmd += ["--partition", args.partition]
                print(json.dumps({"checkpoint": key, "command": cmd}), flush=True)
                if args.submit:
                    lines = subprocess.check_output(cmd + ["--submit"], text=True).strip().splitlines()
                    job_id = json.loads(lines[-1])["id"] if args.backend == "hf" else lines[-1].split(";")[0]
                    ledger[key] = {"id": job_id, "status": "active", "submitted_at": time.time()}
                    write_json(state_path, ledger)
                    active += 1
            write_json(state_path, ledger)
            if args.once:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
