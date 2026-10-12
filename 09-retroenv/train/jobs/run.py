"""Start the RetroEnv server and vLLM on a two-GPU machine, then train or evaluate.

Training logic lives in train/sync_grpo.py and train/async_grpo.py; evaluation is the board's
own eval/evaluate.py pointed at vLLM. In training, the first half of the GPUs (--vllm-gpus)
serve the model, one vLLM replica each, and the rest train; more than one training GPU runs
under FSDP2 (train/jobs/fsdp2.yaml). Evaluation serves a replica on every GPU. The RetroEnv
server runs on CPU from the project's own environment.
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

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PYTHON = sys.executable  # .venv-train: vLLM, TRL, PEFT
ENV_PYTHON = str(ROOT / ".venv-env" / "bin" / "python")  # the project's locked environment: server, evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "eval"])
    parser.add_argument("--mode", choices=["sync", "async"], default="sync")
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--checkpoint", help="For eval: a LoRA checkpoint directory; omit for the base model")
    parser.add_argument("--set", default="core30", help="For eval: evaluation set or split")
    parser.add_argument("--tasks", type=int, help="For eval: only the first N tasks")
    parser.add_argument("--task-ids", help="For eval: a file of task ids in --set's split, one per line")
    parser.add_argument("--attempts", type=int, default=1, help="For eval: episodes per task")
    parser.add_argument("--temperature", type=float, help="For eval: sampling temperature (default: the board's 0)")
    parser.add_argument("--concurrency", type=int, default=16, help="For eval: episodes at once")
    parser.add_argument("--label", help="For eval: the name on the board")
    parser.add_argument("--vllm-gpus", type=int, help="For train: GPUs serving vLLM (default: half)")
    parser.add_argument("--inflight", type=int, help="For async train: episodes in flight (default: 32 per replica)")
    parser.add_argument("--space-id")
    args = parser.parse_args()
    os.chdir(ROOT)
    benchmark = os.environ.get("RETROENV_BENCHMARK_DIR")
    if not benchmark or not Path(benchmark, "serving").exists():
        raise RuntimeError("Prepare the serving snapshot first (entrypoint.sh sets RETROENV_BENCHMARK_DIR)")
    output = Path(args.output).resolve()
    if args.action == "train" and output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a fresh training output")
    output.mkdir(parents=True, exist_ok=True)
    for name, python in (("packages-train.txt", TRAIN_PYTHON), ("packages-env.txt", ENV_PYTHON)):
        with (output / name).open("w") as packages:
            subprocess.run(["uv", "pip", "freeze", "--python", python], stdout=packages, check=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") or subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, check=True
    ).stdout.replace("\n", ",")
    devices = [d.strip() for d in visible.split(",") if d.strip()]
    if args.action == "train":
        serving = args.vllm_gpus or max(1, len(devices) // 2)
        if not 0 < serving < len(devices):
            raise RuntimeError(f"Training needs a GPU for vLLM and one for the trainer; have {len(devices)}")
        vllm_devices, train_devices = devices[:serving], devices[serving:]
    else:
        vllm_devices, train_devices = devices, []
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        TRACKIO_DIR=str(output / "trackio"),
        VLLM_USE_DEEP_GEMM="0",
        VLLM_DEEP_GEMM_WARMUP="skip",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        # The evaluator's OpenAI client needs a key; vLLM runs without one. Not VLLM_API_KEY,
        # which `vllm serve` reads as the key every request must carry.
        RETROENV_EVAL_KEY="EMPTY",
    )
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(2)]
        for listener in sockets:
            listener.bind(("127.0.0.1", 0))
        server_port, engine_port = [s.getsockname()[1] for s in sockets]
    server_url = f"http://127.0.0.1:{server_port}"
    engine_url = f"http://127.0.0.1:{engine_port}"
    (output / "services.json").write_text(json.dumps({"retroenv": server_url, "vllm": engine_url}))
    children = []

    def spawn(command, name, extra=None):
        with (output / (name + ".log")).open("a") as stream:
            process = subprocess.Popen(
                command,
                env={**env, **(extra or {})},
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        children.append(process)
        return process

    def healthy():
        if any(p.poll() is not None for p in children):
            raise RuntimeError("A service exited; inspect the logs in " + str(output))

    def wait(url, timeout=2400):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            healthy()
            try:
                if httpx.get(url, timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        raise TimeoutError(url)

    def run(command, name, extra=None):
        """Run one step to completion while watching the services it depends on."""
        worker = spawn(command, name, extra)
        children.remove(worker)
        try:
            while worker.poll() is None:
                healthy()
                time.sleep(5)
            if worker.returncode:
                raise RuntimeError(f"{name} exited {worker.returncode}; inspect {output / (name + '.log')}")
        finally:
            # Only a step that is still running needs the teardown below; a finished one would
            # read as a crashed service to the next health check.
            if worker.poll() is None:
                children.append(worker)

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        # The RetroEnv server: CPU only, one WebSocket session per episode in flight.
        spawn(
            [
                ENV_PYTHON, "-m", "uvicorn", "retroenv_openenv.server:app",
                "--host", "127.0.0.1", "--port", str(server_port),
                "--ws-ping-timeout", "3600",
            ],
            "retroenv",
            {"MAX_CONCURRENT_ENVS": "256", "ENABLE_WEB_INTERFACE": "false", "OMP_NUM_THREADS": "4"},
        )  # fmt: skip
        weights = None
        if args.action == "eval" and args.checkpoint:
            weights = Path("/tmp/merged")  # local disk; tens of GB that need not reach the bucket
            run(
                [
                    TRAIN_PYTHON, "train/jobs/merge_lora.py",
                    "--model", args.model, "--adapter", args.checkpoint, "--output", str(weights),
                ],
                "merge",
                {"CUDA_VISIBLE_DEVICES": ""},
            )  # fmt: skip
        command = [TRAIN_PYTHON, "train/jobs/serve_model.py", "--model", args.model, "--output", str(output)]
        command += ["--port", str(engine_port), "--data-parallel-size", str(len(vllm_devices))]
        if args.action == "eval":
            command += ["--eval"] + (["--checkpoint", str(weights)] if weights else [])
        else:
            command += ["--trainer", args.mode]
        spawn(command, "vllm", {"CUDA_VISIBLE_DEVICES": ",".join(vllm_devices)})
        wait(server_url + "/health")
        wait(engine_url + "/health")

        if args.action == "train":
            # One training GPU runs the script directly; several run it under FSDP2.
            launcher = [TRAIN_PYTHON, "-m"]
            if len(train_devices) > 1:
                launcher = [
                    TRAIN_PYTHON, "-m", "accelerate.commands.launch",
                    "--config_file", "train/jobs/fsdp2.yaml",
                    "--num_processes", str(len(train_devices)),
                    "--module",
                ]  # fmt: skip
            command = [
                *launcher, f"train.{args.mode}_grpo",
                "--model", args.model,
                "--server", server_url,
                "--vllm-url", engine_url,
                "--output", str(output),
                "--steps", str(args.steps),
                "--save-steps", str(args.save_steps),
                "--max-depth", str(args.max_depth),
            ]  # fmt: skip
            command += ["--inflight", str(args.inflight or 32 * len(vllm_devices))]
            command += ["--smoke"] if args.smoke else []
            command += ["--space-id", args.space_id] if args.space_id else []
            # One JSON line per graded episode: task, reward, whether it submitted, tool calls.
            trace = {
                "CUDA_VISIBLE_DEVICES": ",".join(train_devices),
                "RETROENV_TRACE_PATH": str(output / "episodes.jsonl"),
            }
            run(command, "train", trace)
        else:
            # The board's evaluator and protocol, with vLLM as the provider.
            label = args.label or (f"{args.model} + {Path(args.checkpoint).name}" if args.checkpoint else args.model)
            command = [
                ENV_PYTHON, "eval/evaluate.py", args.model,
                "--set", args.set,
                "--server", server_url,
                "--output", str(output),
                "--concurrency", str(args.concurrency),
                "--provider", "custom",
                "--endpoint", engine_url + "/v1",
                "--api-key-env", "RETROENV_EVAL_KEY",
                "--tool-choice", "auto",
                "--label", label,
            ]  # fmt: skip
            command += ["--tasks", str(args.tasks)] if args.tasks else []
            # Passed through to run_eval.py: several sampled attempts per task label its difficulty.
            command += ["--task-ids", args.task_ids] if args.task_ids else []
            command += ["--attempts", str(args.attempts)] if args.attempts > 1 else []
            command += ["--temperature", str(args.temperature)] if args.temperature is not None else []
            run(command, "eval")
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
