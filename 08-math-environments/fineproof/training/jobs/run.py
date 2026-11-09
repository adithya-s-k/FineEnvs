"""Start vLLM and the Harbor server, then train or evaluate, on a two-GPU machine.

Adapted from 05-multi-harness-rl/jobs/run.py, keeping only the Harbor path. Training: vLLM on GPU 0, the trainer
on GPU 1. Evaluation: vLLM data-parallel on both. Logs go to <output>/{vllm,harbor,train|eval}.log.

    python jobs/run.py train --output runs/train --steps 100
    python jobs/run.py eval  --output runs/eval-base
    python jobs/run.py eval  --output runs/eval-step100 --checkpoint runs/train/checkpoint-100 --step 100
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
    output = Path(args.output).resolve()
    if output.exists() and args.action == "train":
        raise ValueError("Use a fresh training output directory")
    output.mkdir(parents=True, exist_ok=True)
    revisions = ROOT / ".deps/revisions.json"
    if revisions.exists():
        (output / "dependencies.json").write_text(revisions.read_text())
    with (output / "packages.txt").open("w") as packages:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=packages, check=True)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(devices) < 2:
        raise RuntimeError("Allocate two GPUs")
    env = dict(os.environ, PYTHONUNBUFFERED="1", TRACKIO_DIR=str(output / "trackio"), VLLM_USE_DEEP_GEMM="0",
               VLLM_DEEP_GEMM_WARMUP="skip", VLLM_USE_FLASHINFER_SAMPLER="0")
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

    def wait(url, timeout=1200):
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
            command += ["--eval"] + (["--checkpoint", args.checkpoint] if args.checkpoint else [])
        spawn(command, "vllm", {"CUDA_VISIBLE_DEVICES": devices[0] if args.action == "train" else ",".join(devices[:2])})
        wait(engine_url + "/health")
        # OpenEnv's own Harbor server: it resolves `--harness fineproof_agent.agent:FineProofAgent` itself and serves
        # each dataset folder as a split named by its path (tasks.dataset_dir).
        spawn([sys.executable, "-m", "uvicorn", "harbor_env.server.app:app", "--host", "127.0.0.1",
               "--port", str(server_port), "--ws-ping-timeout", "1800"], "harbor", {
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
        worker = spawn(command, args.action, {"CUDA_VISIBLE_DEVICES": devices[1]})
        children.remove(worker)
        try:
            while worker.poll() is None:
                healthy()
                time.sleep(2)
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
