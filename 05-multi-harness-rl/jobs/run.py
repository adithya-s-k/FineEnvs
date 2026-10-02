"""Launch services and one tutorial script on a two-GPU machine. Training logic lives in train/."""

import argparse
import json
import os
import secrets
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "eval"])
    parser.add_argument(
        "--mode",
        choices=["whitebox", "opencode", "multi_harness"],
        default="multi_harness",
    )
    parser.add_argument("--model", default="LiquidAI/LFM2.5-2.6B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--checkpoint")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--tasks", type=int, default=250)
    parser.add_argument("--concurrency", type=int, default=35)
    parser.add_argument("--space-id")
    args = parser.parse_args()
    os.chdir(ROOT)
    subprocess.run([sys.executable, "check_setup.py", "--mode", args.mode], check=True)
    if not Path("prepared/ready.json").exists():
        raise RuntimeError("Run prepare.py before allocating GPUs")
    output = Path(args.output).resolve()
    if output.exists() and args.action == "train":
        raise ValueError("Use a fresh training output directory")
    output.mkdir(parents=True, exist_ok=True)
    revisions = ROOT / ".deps/revisions.json"
    if revisions.exists():
        (output / "dependencies.json").write_text(revisions.read_text())
    with (output / "packages.txt").open("w") as packages:
        subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], stdout=packages, check=True
        )
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(devices) < 2:
        raise RuntimeError("Allocate two GPUs")
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        TRACKIO_DIR=str(output / "trackio"),
        VLLM_USE_DEEP_GEMM="0",
        VLLM_DEEP_GEMM_WARMUP="skip",
        VLLM_USE_FLASHINFER_SAMPLER="0",
    )
    with ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(4)]
        for listener in sockets:
            listener.bind(("127.0.0.1", 0))
        engine_port, server_port, capture_port, proxy_port = [
            s.getsockname()[1] for s in sockets
        ]
    engine_url = f"http://127.0.0.1:{engine_port}"
    server_url = f"http://127.0.0.1:{server_port}"
    (output / "services.json").write_text(
        json.dumps({"vllm": engine_url, "openenv": server_url})
    )
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
        command = [
            sys.executable,
            "jobs/serve_model.py",
            "--model",
            args.model,
            "--output",
            str(output),
            "--port",
            str(engine_port),
        ]
        if args.action == "eval":
            command += ["--eval"]
            if args.checkpoint:
                command += ["--checkpoint", args.checkpoint]
        spawn(
            command,
            "vllm",
            {
                "CUDA_VISIBLE_DEVICES": devices[0]
                if args.action == "train"
                else ",".join(devices[:2])
            },
        )
        wait(engine_url + "/health")
        if args.mode != "whitebox" and (
            args.action == "eval" or args.mode == "multi_harness"
        ):
            spawn(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "smoldataenv_harbor.server:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(server_port),
                    "--ws-ping-timeout",
                    "1800",
                ],
                "harbor",
                {
                    "OPENENV_DATASETS": ",".join(
                        str(Path("prepared/datasets", s).resolve())
                        for s in ("train", "test")
                    ),
                    "OPENENV_LLM_URL": engine_url,
                    "OPENENV_MODEL": args.model,
                    "OPENENV_HARBOR_TRIALS_DIR": str(output / "trials"),
                    "OPENENV_MAX_OUTPUT_TOKENS": "4096",
                    "OPENENV_CAPTURE_PORT": str(capture_port),
                    "OPENENV_EXPOSE": "gradio",
                    "MAX_CONCURRENT_ENVS": str(max(32, args.concurrency) + 8),
                    "ENABLE_WEB_INTERFACE": "false",
                    "OPENENV_HARBOR_AGENT_VERSIONS": '{"opencode":"1.18.31","claude-code":"2.1.270","codex":"0.154.0","mini-swe-agent":"2.4.6"}',
                },
            )
            wait(server_url + "/health")
        if args.action == "train" and args.mode == "opencode":
            env["SANDBOX_VLLM_KEY"] = secrets.token_urlsafe(32)
            location = output / "inference-url.txt"
            spawn(
                [
                    sys.executable,
                    "jobs/inference_proxy.py",
                    "--output",
                    str(location),
                    "--port",
                    str(proxy_port),
                    "--upstream",
                    engine_url,
                ],
                "inference-proxy",
            )
            deadline = time.monotonic() + 180
            while not location.exists():
                healthy()
                if time.monotonic() > deadline:
                    raise TimeoutError("Public inference tunnel")
                time.sleep(1)
            env["SANDBOX_VLLM_URL"] = location.read_text()
            # Confirm that the sandbox-facing endpoint has the expected model.
            response = httpx.get(
                env["SANDBOX_VLLM_URL"] + "/v1/models",
                headers={"Authorization": "Bearer " + env["SANDBOX_VLLM_KEY"]},
                timeout=30,
            )
            response.raise_for_status()
        if args.mode == "whitebox":
            spawn(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "smoldataenv_whitebox.server:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(server_port),
                ],
                "whitebox",
                {"ENABLE_WEB_INTERFACE": "false"},
            )
            wait(server_url + "/health")
        if args.action == "train" and args.mode == "opencode":
            spawn(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "smoldataenv_opencode.server:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(server_port),
                    "--ws-ping-timeout",
                    "1800",
                ],
                "opencode",
                {"ENABLE_WEB_INTERFACE": "false"},
            )
            wait(server_url + "/health")
        common = [
            "--model",
            args.model,
            "--output",
            str(output),
            "--vllm-url",
            engine_url,
        ]
        common += ["--server", server_url]
        if args.action == "eval" or args.mode == "multi_harness":
            common += ["--trials", str(output / "trials")]
        if args.space_id:
            common += ["--space-id", args.space_id]
        if args.action == "train":
            command = [
                sys.executable,
                "-m",
                f"train.{args.mode}",
                *common,
                "--steps",
                str(args.steps),
                "--save-steps",
                str(args.save_steps),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "eval.evaluate",
                *common,
                "--mode",
                "whitebox" if args.mode == "whitebox" else "blackbox",
                "--checkpoint",
                args.checkpoint or args.model,
                "--step",
                str(args.step),
                "--tasks",
                str(args.tasks),
                "--concurrency",
                str(args.concurrency),
            ]
        if args.smoke and args.action == "train":
            command += ["--smoke"]
        worker = spawn(command, args.action, {"CUDA_VISIBLE_DEVICES": devices[1]})
        children.remove(worker)
        try:
            while worker.poll() is None:
                healthy()
                time.sleep(2)
            if worker.returncode:
                raise RuntimeError(
                    f"{args.action} exited {worker.returncode}; inspect {output}"
                )
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
