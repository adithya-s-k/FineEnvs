"""Plan, serve, train or evaluate the three SmolDataEnv modes."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

from recipe import ROOT, MODES, config, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "train", "eval", "smoke", "environment", "_train", "_eval"])
    p.add_argument("--config", type=Path)
    p.add_argument("--model", choices=["lfm", "qwen"])
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--data", type=Path, default=ROOT / "prepared")
    p.add_argument("--output", type=Path)
    p.add_argument("--run-name")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--limit", type=int, help="Eval tasks per harness; omit for all 250")
    p.add_argument("--smoke-eval", action="store_true", help="Reload the smoke checkpoint for two tasks per harness")
    p.add_argument("--server", help="External URL, or a local bind URL; owned services default to unused ports")
    p.add_argument("--vllm-url")
    p.add_argument("--capture-port", type=int)
    p.add_argument("--external-services", action="store_true", help="Use the supplied server URLs without starting services")
    return p


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_ready(url, process, timeout=1200):
    import httpx
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited with {process.returncode}; inspect its log")
        try:
            if httpx.get(url + "/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(f"Service readiness exceeded {timeout}s")


def launch_local(args, cfg):
    from runtime.bootstrap import activate
    activate()
    from runtime.models import tokenizer_for, serving_command, check_visible_response
    from runtime.environment import credentials

    credentials()
    if not (args.data / "ready.json").exists():
        raise ValueError("Run prepare.py before allocating GPUs")
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    if args.action in {"train", "smoke"} and not args.resume and (output / "metrics.jsonl").exists():
        raise ValueError("Existing training output; choose a new run name or pass --resume")
    training = args.action in {"train", "smoke"}
    if args.checkpoint:
        from runtime.checkpoints import verify
        info = verify(args.checkpoint)
        if info["base_model"] != cfg["profile"]["id"]:
            raise ValueError("Checkpoint model differs from configured model")
    if args.resume:
        from runtime.checkpoints import verify
        info = verify(args.resume, resume=True)
        if info["base_model"] != cfg["profile"]["id"] or info["base_revision"] != cfg["profile"]["revision"]:
            raise ValueError("Resume model differs from configured model")
    model = str(args.resume or args.checkpoint or cfg["profile"]["id"])
    template = output / "chat_template.jinja"
    tokenizer_for(cfg, template)
    config_path = output / "config.json"
    write_json(config_path, cfg)
    write_json(output / "services.json", {"environment": args.server, "vllm": args.vllm_url,
                                          "capture_port": args.capture_port})
    env = dict(os.environ, PYTHONUNBUFFERED="1", VLLM_SERVER_DEV_MODE="1",
               VLLM_USE_DEEP_GEMM="0", VLLM_DEEP_GEMM_WARMUP="skip", VLLM_USE_FLASHINFER_SAMPLER="0",
               TRACKIO_DIR=str(output / "trackio"))
    processes = []
    def spawn(command, name, child_env=None):
        with (output / f"{name}.log").open("a") as stream:
            proc = subprocess.Popen(command, env=child_env or env, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True, cwd=ROOT)
        processes.append(proc)
        return proc
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(devices) < 2:
        raise ValueError("Allocate two GPUs: inference + trainer, or two evaluation replicas")
    try:
        inference_env = dict(env, CUDA_VISIBLE_DEVICES=devices[0] if training else ",".join(devices[:2]))
        port = urlparse(args.vllm_url).port or 8000
        engine = spawn(serving_command(cfg, model, training=training, template_path=template, port=port), "vllm", inference_env)
        wait_ready(args.vllm_url, engine)
        write_json(output / "inference-check.json", check_visible_response(cfg, args.vllm_url))
        base = [sys.executable, str(ROOT / "run.py")]
        common = ["--config", str(config_path), "--data", str(args.data.resolve()), "--output", str(output),
                  "--server", args.server, "--vllm-url", args.vllm_url,
                  "--capture-port", str(args.capture_port)]
        service_mode = cfg["mode"] if training or cfg["mode"] == "whitebox" else "multi-harness"
        service = spawn(base + ["environment", *common, "--mode", service_mode], "environment")
        wait_ready(args.server, service)
        action = "_train" if training else "_eval"
        command = base + [action, *common]
        if args.resume:
            command += ["--resume", str(args.resume.resolve())]
        if args.checkpoint:
            command += ["--checkpoint", str(args.checkpoint.resolve())]
        if args.limit:
            command += ["--limit", str(args.limit)]
        worker = spawn(command, "train" if training else "eval", dict(env, CUDA_VISIBLE_DEVICES=devices[1]) if training else env)
        while worker.poll() is None:
            if engine.poll() is not None or service.poll() is not None:
                raise RuntimeError("Inference or environment service died; stopping worker")
            time.sleep(2)
        if worker.returncode:
            raise RuntimeError(f"Worker failed ({worker.returncode}); inspect {output}")
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in reversed(processes):
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def main():
    args = parser().parse_args()
    if args.smoke_eval and (args.action != "smoke" or args.external_services):
        raise ValueError("--smoke-eval requires a smoke with job-local services")
    if args.external_services and (not args.server or not args.vllm_url):
        raise ValueError("External services require --server and --vllm-url")
    args.server = args.server or f"http://127.0.0.1:{unused_port()}"
    args.vllm_url = args.vllm_url or f"http://127.0.0.1:{unused_port()}"
    args.capture_port = args.capture_port or unused_port()
    cfg = config(args.config, model=args.model, mode=args.mode, eval_concurrency=args.concurrency)
    name = args.run_name or cfg.get("run_name") or f'{cfg["model"]}-{cfg["mode"]}-{time.strftime("%Y%m%d-%H%M%S")}'
    cfg.update(run_name=name, output=str((args.output or Path(cfg.get("output", ROOT / "runs" / name))).resolve()))
    if args.action == "smoke":
        cfg.update(max_steps=2, save_steps=1, eval_steps=1, num_generations=2,
                   max_inflight=4, max_outstanding_rollouts=4, batch_size=1, gradient_accumulation_steps=2)
    if args.action == "plan":
        print(json.dumps(cfg, indent=2))
        return
    if args.action in {"train", "eval", "smoke"}:
        if not args.external_services:
            launch_local(args, cfg)
            if args.smoke_eval:
                from runtime.checkpoints import make_ready
                checkpoint = Path(cfg["output"]) / f'checkpoint-{cfg["max_steps"]}'
                make_ready(checkpoint)
                eval_args = argparse.Namespace(**vars(args))
                eval_args.action, eval_args.checkpoint, eval_args.resume = "eval", checkpoint, None
                eval_args.limit = args.limit or 2
                eval_cfg = {**cfg, "output": str(Path(cfg["output"]) / "reload-eval"),
                            "run_name": cfg["run_name"] + "-reload", "eval_concurrency": args.concurrency or 4}
                launch_local(eval_args, eval_cfg)
            return
        args.action = "_eval" if args.action == "eval" else "_train"
    from runtime.bootstrap import activate
    activate()
    if args.action == "environment":
        from runtime.environment import serve
        serve(cfg, args.data, args.vllm_url, port=urlparse(args.server).port or 8200,
              capture_port=args.capture_port)
    elif args.action == "_train":
        if cfg["mode"] == "whitebox":
            from train.whitebox import train
        else:
            from train.blackbox import train
        train(cfg, args.data, args.server, args.vllm_url, args.resume)
    else:
        from eval.evaluate import evaluate
        report = evaluate(cfg, args.data, args.server, args.vllm_url,
                          limit=args.limit, checkpoint=args.checkpoint or "baseline")
        if not report["complete"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
