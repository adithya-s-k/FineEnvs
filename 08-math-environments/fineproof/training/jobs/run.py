"""Start vLLM and the Harbor server, then train or evaluate, on a machine with two or more GPUs.

Adapted from 05-multi-harness-rl/jobs/run.py, keeping only the Harbor path. Training: the trainer on the last GPU and
vLLM data-parallel on the others (generation is what limits a run). Evaluation: vLLM data-parallel on all of them. Logs go to <output>/{vllm,harbor,train|eval}.log.

    python jobs/run.py train --output runs/train --steps 100
    python jobs/run.py eval  --output runs/eval-base
    python jobs/run.py eval  --output runs/eval-step100 --checkpoint runs/train/checkpoint-100 --step 100

In an HF Job, `train --eval-checkpoints` also sends each saved checkpoint to an eval Job of its own
(jobs/checkpoint_evals.py). The OpenEnv server logs at INFO (budget stops, token clamps, upstream retries) to
<output>/harbor.log, with the config in <output>/harbor_logging.json.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def local_checkpoint(checkpoint: str, output: Path) -> str:
    """A checkpoint on the read-only bucket mount (/outputs/...), copied to local disk first.

    vLLM reads a mounted 18 GB safetensors file far too slowly to start (the first checkpoint eval timed out after
    10 minutes at 0%); `hf buckets sync` downloads it at full speed. The copy goes outside the output folder, which is
    itself synced to the bucket.
    """
    path, bucket = Path(checkpoint), os.environ.get("FINEPROOF_OUTPUT_BUCKET")
    if not bucket or not path.is_relative_to("/outputs"):
        return checkpoint
    local = Path("/workspace/checkpoints") / path.relative_to("/outputs")
    start = time.monotonic()
    with (output / "checkpoint_copy.log").open("a") as log:
        subprocess.run(["hf", "buckets", "sync", f"hf://buckets/{bucket}/{path.relative_to('/outputs')}", str(local)],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"copied {checkpoint} to {local} in {time.monotonic() - start:.0f}s\n")
    if not (local / "config.json").is_file() or not any(local.glob("*.safetensors")):
        raise RuntimeError(f"{checkpoint} did not copy completely to {local}")
    return str(local)


def visible_gpus() -> list[str]:
    """CUDA_VISIBLE_DEVICES when set (Slurm sets it), else every GPU nvidia-smi lists (an HF Job may set neither)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [d for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d.strip()]
    try:
        listed = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                                text=True, check=True).stdout
        return [line.strip() for line in listed.splitlines() if line.strip()] or ["0", "1"]
    except (OSError, subprocess.CalledProcessError):
        return ["0", "1"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["train", "eval"])
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--checkpoint")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=128)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--space-id")
    parser.add_argument("--trackio-bucket", help="Bucket the Trackio Space keeps its data in")
    parser.add_argument("--trackio-run", help="Eval: the Trackio run to log into (default: the output's name)")
    parser.add_argument("--eval-checkpoints", action="store_true",
                        help="Train in an HF Job: eval each saved checkpoint in an HF Job of its own")
    parser.add_argument("--eval-concurrency", type=int, default=64, help="With --eval-checkpoints: rollouts in flight")
    parser.add_argument("--sandbox", default="daytona", choices=["daytona", "hf-sandbox"],
                        help="Where Harbor runs each task's sandboxes (hf-sandbox: HF Jobs, needs a docker_image task)")
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("FINEPROOF_ROUNDS", "1")),
                        help="Check-and-rewrite rounds of the FineProof harness (0: one call per proof)")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Output cap per model call")
    parser.add_argument("--grad-accum", type=int, default=40, help="Train: rollouts per update")
    parser.add_argument("--inflight", type=int, default=32)
    parser.add_argument("--task-seed", type=int, help="Eval: score a seeded random --tasks subset of test")
    parser.add_argument("--task-list", help="Eval: score exactly the test tasks listed in this file")
    args = parser.parse_args()
    os.chdir(ROOT)
    subprocess.run([sys.executable, "check_setup.py"], check=True)
    if not Path("prepared/ready.json").exists():
        raise RuntimeError("Run prepare.py before allocating GPUs")
    if args.sandbox == "daytona" and not os.environ.get("DAYTONA_API_KEY"):
        raise RuntimeError("DAYTONA_API_KEY is needed for --sandbox daytona")
    if not os.environ.get("JUDGE_API_KEY"):
        raise RuntimeError("JUDGE_API_KEY is needed: each task's verifier calls the judge with it")
    if args.eval_checkpoints and (args.action != "train" or not os.environ.get("FINEPROOF_OUTPUT_BUCKET")):
        raise RuntimeError("--eval-checkpoints is for training in an HF Job (FINEPROOF_OUTPUT_BUCKET names its bucket)")
    output = Path(args.output).resolve()
    if output.exists() and args.action == "train":
        raise ValueError("Use a fresh training output directory")
    output.mkdir(parents=True, exist_ok=True)
    revisions = ROOT / ".deps/revisions.json"
    if revisions.exists():
        (output / "dependencies.json").write_text(revisions.read_text())
    with (output / "packages.txt").open("w") as packages:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=packages, check=True)
    devices = visible_gpus()
    if len(devices) < 2:
        raise RuntimeError("Allocate two or more GPUs")
    engine_devices = devices[:-1] if args.action == "train" else devices
    env = dict(os.environ, PYTHONUNBUFFERED="1", TRACKIO_DIR=str(output / "trackio"), VLLM_USE_DEEP_GEMM="0",
               VLLM_DEEP_GEMM_WARMUP="skip", VLLM_USE_FLASHINFER_SAMPLER="0",
               # Trackio: keep train.log's console in the run too, and finish the run (flushing it) on SIGTERM.
               TRACKIO_CAPTURE_CONSOLE="1", TRACKIO_HANDLE_SIGTERM="1",
               # A 4B checkpoint is 18 GB of fp32 weights; vLLM's default 10 minutes for engines to load is tight.
               VLLM_ENGINE_READY_TIMEOUT_S=os.environ.get("VLLM_ENGINE_READY_TIMEOUT_S", "1800"))
    # OpenEnv's Harbor modules log what they decide (budget stops, token clamps, upstream retries) at INFO, which
    # uvicorn's default config drops; this config keeps them, and Harbor's own loggers at WARNING.
    logging_config = output / "harbor_logging.json"
    logging_config.write_text(json.dumps({
        "version": 1, "disable_existing_loggers": False,
        "formatters": {"plain": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
        "handlers": {"stdout": {"class": "logging.StreamHandler", "formatter": "plain", "stream": "ext://sys.stdout"}},
        "root": {"level": "WARNING", "handlers": ["stdout"]},
        "loggers": {name: {"level": "INFO"} for name in ("openenv", "harbor_env", "uvicorn", "uvicorn.error")},
    }, indent=2))
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(3)]
        for listener in sockets:
            listener.bind(("127.0.0.1", 0))
        engine_port, server_port, capture_port = [s.getsockname()[1] for s in sockets]
    engine_url, server_url = f"http://127.0.0.1:{engine_port}", f"http://127.0.0.1:{server_port}"
    (output / "services.json").write_text(json.dumps({"vllm": engine_url, "openenv": server_url}))
    children = []

    def spawn(command, name, extra=None):
        with (output / (name + ".log")).open("a") as stream:
            process = subprocess.Popen(command, env={**env, **(extra or {})}, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process

    def healthy():
        if any(p.poll() is not None for p in children):
            raise RuntimeError("A service exited; inspect logs in " + str(output))

    def wait(url, timeout=2400):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            healthy()
            try:
                if httpx.get(url, timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise TimeoutError(url)

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        command = [sys.executable, "jobs/serve_model.py", "--model", args.model, "--output", str(output),
                   "--port", str(engine_port)]
        if args.action == "eval":
            command += ["--eval"] + (["--checkpoint", local_checkpoint(args.checkpoint, output)] if args.checkpoint else [])
        command += ["--data-parallel", str(len(engine_devices))]
        spawn(command, "vllm", {"CUDA_VISIBLE_DEVICES": ",".join(engine_devices)})
        wait(engine_url + "/health")
        # OpenEnv's own Harbor server: it resolves `--harness fineproof_agent.agent:FineProofAgent` itself and serves
        # each dataset folder as a split named by its path (tasks.dataset_dir).
        spawn([sys.executable, "-m", "uvicorn", "harbor_env.server.app:app", "--host", "127.0.0.1",
               "--port", str(server_port), "--ws-ping-timeout", "1800", "--log-config", str(logging_config)], "harbor", {
            "OPENENV_DATASETS": ",".join(str(Path("prepared", "datasets", s).resolve()) for s in ("train", "test")),
            "OPENENV_HARBOR_REWARD_KEY": "reward",
            "FINEPROOF_ROUNDS": str(args.rounds),
            "FINEPROOF_MAX_TOKENS": str(args.max_tokens),
            "OPENENV_LLM_URL": engine_url,
            "OPENENV_MODEL": args.model,
            "OPENENV_HARBOR_TRIALS_DIR": str(output / "trials"),
            "OPENENV_MAX_OUTPUT_TOKENS": str(args.max_tokens),
            "OPENENV_CAPTURE_PORT": str(capture_port),
            # The FineProof harness runs host-side and reaches the capture proxy on loopback, so no public tunnel is needed.
            "OPENENV_EXPOSE": os.environ.get("OPENENV_EXPOSE", "direct"),
            "MAX_CONCURRENT_ENVS": str(max(args.inflight, args.concurrency) + 8),
            "ENABLE_WEB_INTERFACE": "false",
        })
        wait(server_url + "/health")
        common = ["--model", args.model, "--output", str(output), "--vllm-url", engine_url, "--server", server_url,
                  "--trials", str(output / "trials"), "--sandbox", args.sandbox]
        if args.space_id:
            common += ["--space-id", args.space_id]
        if args.trackio_bucket:
            common += ["--trackio-bucket", args.trackio_bucket]
        if args.action == "train":
            command = [sys.executable, "train.py", *common, "--steps", str(args.steps),
                       "--save-steps", str(args.save_steps), "--grad-accum", str(args.grad_accum),
                       "--inflight", str(args.inflight)] + (["--smoke"] if args.smoke else [])
        else:
            command = [sys.executable, "evaluate.py", *common, "--checkpoint", args.checkpoint or args.model,
                       "--step", str(args.step), "--tasks", str(args.tasks), "--samples", str(args.samples),
                       "--concurrency", str(args.concurrency), "--rounds", str(args.rounds),
                       "--max-tokens", str(args.max_tokens)]
            if args.task_seed is not None:
                command += ["--task-seed", str(args.task_seed)]
            if args.task_list:
                command += ["--task-list", args.task_list]
            if args.trackio_run:
                command += ["--trackio-run", args.trackio_run]
        evals = None
        if args.eval_checkpoints:
            from checkpoint_evals import CheckpointEvals

            evals = CheckpointEvals(output, bucket=os.environ["FINEPROOF_OUTPUT_BUCKET"], model=args.model,
                                    sandbox=args.sandbox, rounds=args.rounds, max_tokens=args.max_tokens,
                                    space_id=args.space_id, trackio_bucket=args.trackio_bucket,
                                    concurrency=args.eval_concurrency)
            evals.start()
        worker = spawn(command, args.action, {"CUDA_VISIBLE_DEVICES": devices[-1]})
        children.remove(worker)
        try:
            while worker.poll() is None:
                healthy()
                time.sleep(2)
            if evals is not None:
                evals.finish()  # the final checkpoint's eval, and any launch that failed earlier
            if worker.returncode:
                raise RuntimeError(f"{args.action} exited {worker.returncode}; inspect {output}")
        finally:
            children.append(worker)
    finally:
        for process in reversed(children):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in reversed(children):
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
