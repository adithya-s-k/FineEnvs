"""Start vLLM and the Harbor server, then train or evaluate, on a machine with two or more GPUs.

Adapted from 05-multi-harness-rl/jobs/run.py, keeping only the Harbor path. Training: the trainer on the last GPU and
vLLM data-parallel on the others (generation is what limits a run). Evaluation: vLLM data-parallel on all of them. Logs go to <output>/{vllm,harbor,train|eval}.log.

    python jobs/run.py train --output runs/train --steps 100
    python jobs/run.py eval  --output runs/eval-base
    python jobs/run.py eval  --output runs/eval-step100 --checkpoint runs/train/checkpoint-100 --step 100

In an HF Job, `train --eval-checkpoints` also sends saved checkpoints to eval Jobs of their own
(jobs/checkpoint_evals.py). The OpenEnv server logs at INFO (budget stops, token clamps, upstream retries) to
<output>/harbor.log, with the config in <output>/harbor_logging.json.

A run that dies restarts inside the Job from its latest full checkpoint (--max-restarts). A Job that ends early (it
timed out, or was stopped) resumes when it is submitted again under the same --name: training starts from the latest
full checkpoint in the bucket, with the run's logs, rollouts.jsonl, Trackio data and eval record copied back first, so
every log and the Trackio run continue where they stopped (restore_run below).
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = Path("/workspace/outputs")  # jobs/entrypoint.sh copies this folder to the bucket's root


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
        subprocess.run(["hf", "buckets", "sync", f"hf://buckets/{bucket}/{path.relative_to('/outputs')}", str(local),
                        "--exclude", "optimizer.pt", "--exclude", "scheduler.pt", "--exclude", "rng_state*"],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"copied {checkpoint} to {local} in {time.monotonic() - start:.0f}s\n")
    if not (local / "config.json").is_file() or not any(local.glob("*.safetensors")):
        raise RuntimeError(f"{checkpoint} did not copy completely to {local}")
    return str(local)


def latest_checkpoint(output: Path) -> Path | None:
    """The newest checkpoint-N whose trainer_state.json (written last) parses: a run can resume from it."""
    for path in sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        try:
            json.loads((path / "trainer_state.json").read_text())
        except (OSError, ValueError):
            continue
        if (path / "optimizer.pt").is_file() or any(path.glob("optimizer*")):
            return path
    return None


def bucket_url(output: Path) -> str | None:
    """Where jobs/entrypoint.sh copies `output` in the run's bucket, in an HF Job."""
    bucket = os.environ.get("FINEPROOF_OUTPUT_BUCKET")
    if not bucket or not output.is_relative_to(OUTPUTS_ROOT):
        return None
    return f"hf://buckets/{bucket}/{output.relative_to(OUTPUTS_ROOT).as_posix()}"


def _bucket_names(url: str) -> list[str]:
    """The entries directly under `url` (last path part), or nothing when it does not exist."""
    listed = subprocess.run(["hf", "buckets", "ls", url], capture_output=True, text=True)
    if listed.returncode != 0:
        return []
    return [line.split()[-1].rstrip("/").rsplit("/", 1)[-1] for line in listed.stdout.splitlines() if line.strip()]


def remote_checkpoint(url: str) -> str | None:
    """The newest checkpoint-N in the bucket that a run can resume from: trainer state and optimizer both there."""
    steps = sorted((int(n.split("-")[1]), n) for n in _bucket_names(url) if re.fullmatch(r"checkpoint-\d+", n))
    for _, name in reversed(steps):
        files = _bucket_names(f"{url}/{name}")
        if "trainer_state.json" in files and any(f.startswith("optimizer") for f in files):
            return name
    return None


def restore_run(output: Path) -> Path | None:
    """Continue a run an earlier Job left in the bucket: copy back its state and its latest full checkpoint.

    The small files come first (logs, rollouts.jsonl, threads.jsonl, restarts.jsonl, checkpoint_evals.jsonl and the
    Trackio folder), so every log continues where it stopped and the eval launcher does not submit the same
    checkpoint twice; trials and older checkpoints stay in the bucket (the sync to it never deletes). Returns the local
    checkpoint to resume from, or None for a new run.
    """
    url = bucket_url(output)
    name = remote_checkpoint(url) if url else None
    if name is None:
        return None
    start = time.monotonic()
    with (output / "resume_copy.log").open("a") as log:
        subprocess.run(["hf", "buckets", "sync", url, str(output), "--exclude", "checkpoint-*", "--exclude", "trials/*",
                        "--exclude", "final/*", "--exclude", "trackio/*.lock", "--format", "quiet"],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(["hf", "buckets", "sync", f"{url}/{name}", str(output / name), "--format", "quiet"],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    if latest_checkpoint(output) != output / name:
        raise RuntimeError(f"{url}/{name} did not copy completely to {output}")
    with (output / "resumes.jsonl").open("a") as handle:
        handle.write(json.dumps({"time": round(time.time(), 1), "from": name, "bucket": url,
                                 "copy_s": round(time.monotonic() - start, 1)}) + "\n")
    return output / name


class ThreadCensus(threading.Thread):
    """Every minute, threads and memory per process in this container, to <output>/threads.jsonl.

    A run died of `can't start new thread` after about 2,250 rollouts; this records which process's thread count
    grows, against the container's pid limit (cgroup pids.max) when it can read it.
    """

    # First match wins: train.py's command line also mentions --vllm-url.
    ROLES = (("train.py", "train"), ("evaluate.py", "eval"), ("harbor_env.server", "harbor"), ("jobs/run.py", "run"),
             ("serve_model", "vllm"), ("vllm", "vllm"))

    def __init__(self, path: Path, every_s: float = 60.0):
        super().__init__(name="thread-census", daemon=True)
        self.path, self.every_s, self.stopping = path, every_s, threading.Event()

    @staticmethod
    def _limit(name: str) -> int | None:
        for base in ("/sys/fs/cgroup", "/sys/fs/cgroup/pids"):
            try:
                value = Path(base, name).read_text().strip()
                return None if value == "max" else int(value)
            except (OSError, ValueError):
                continue
        return None

    def snapshot(self) -> dict:
        roles: dict[str, dict] = {}
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                status = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines() if ":" in line)
                cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            role = next((r for key, r in self.ROLES if key in cmdline), status.get("Name", "?").strip())
            entry = roles.setdefault(role, {"processes": 0, "threads": 0, "rss_mb": 0})
            entry["processes"] += 1
            entry["threads"] += int(status.get("Threads", "0").strip() or 0)
            entry["rss_mb"] += int(status.get("VmRSS", "0 kB").split()[0]) // 1024
        top = dict(sorted(roles.items(), key=lambda kv: -kv[1]["threads"])[:12])
        return {"time": round(time.time(), 1), "threads": sum(r["threads"] for r in roles.values()),
                "pids_current": self._limit("pids.current"), "pids_max": self._limit("pids.max"),
                "nproc_limit": self._nproc_limit(), "roles": top}

    @staticmethod
    def _nproc_limit() -> str | None:
        """This process's RLIMIT_NPROC (threads count against it), as /proc/self/limits shows it."""
        try:
            line = next(l for l in Path("/proc/self/limits").read_text().splitlines() if l.startswith("Max processes"))
            return " ".join(line.split()[2:4])
        except (OSError, StopIteration):
            return None

    def run(self) -> None:
        while not self.stopping.wait(self.every_s):
            try:
                with self.path.open("a") as handle:
                    handle.write(json.dumps(self.snapshot()) + "\n")
            except Exception:  # noqa: BLE001 - a census must never stop a run
                pass

    def stop(self) -> None:
        self.stopping.set()


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
    parser.add_argument("--save-steps", type=int, default=10)
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
    parser.add_argument("--eval-every", type=int, default=20,
                        help="With --eval-checkpoints: eval checkpoints whose step is a multiple of this, and the last")
    parser.add_argument("--sandbox", default="daytona", choices=["daytona", "hf-sandbox"],
                        help="Where Harbor runs each task's sandboxes (hf-sandbox: HF Jobs, needs a docker_image task)")
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("FINEPROOF_ROUNDS", "1")),
                        help="Check-and-rewrite rounds of the FineProof harness (0: one call per proof)")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Output cap per model call")
    parser.add_argument("--generations", type=int, default=16, help="Train: rollouts per problem (a GRPO group)")
    parser.add_argument("--grad-accum", type=int, default=128, help="Train: rollouts per update")
    parser.add_argument("--inflight", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-6, help="Train: AsyncGRPO learning rate")
    parser.add_argument("--truncated-reward", default="zero", choices=["zero", "judge"],
                        help="Train: reward of a proof cut off at the cap (see tasks.py)")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Train: linear learning-rate warm-up")
    parser.add_argument("--resume-from", help="Train: start from another run's checkpoint (/outputs/<run>/checkpoint-N)")
    parser.add_argument("--evals-run", help="With --eval-checkpoints: the Trackio run evals log to (default <name>-evals)")
    parser.add_argument("--max-restarts", type=int, default=3,
                        help="Train: restarts from the latest full checkpoint when training dies")
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
               VLLM_ENGINE_READY_TIMEOUT_S=os.environ.get("VLLM_ENGINE_READY_TIMEOUT_S", "1800"),
               # Thread pools default to one thread per CPU in every process (92 on h200x4); vLLM on three GPUs alone
               # then ran about 2,200 threads and a run hit the container's thread limit. Eight each is plenty here.
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", RAYON_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
    # Two judge samples per proof, averaged (the verifier's default is one): a single gpt-oss-20b grade is noisy, and
    # within a group that noise is most of the difference between near-equal proofs.
    env.setdefault("JUDGE_SAMPLES", "2")
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

    def stop_children():
        for process in reversed(children):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in reversed(children):
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        children.clear()

    def start_services(weights=None):
        """vLLM (serving `weights` when given: a resumed run's latest checkpoint), then the OpenEnv Harbor server."""
        command = [sys.executable, "jobs/serve_model.py", "--model", args.model, "--output", str(output),
                   "--port", str(engine_port), "--data-parallel", str(len(engine_devices))]
        if args.action == "eval":
            command += ["--eval"]
        if weights:
            command += ["--checkpoint", weights]
        spawn(command, "vllm", {"CUDA_VISIBLE_DEVICES": ",".join(engine_devices)})
        wait(engine_url + "/health")
        # OpenEnv's own Harbor server: it resolves `--harness fineproof_agent.agent:FineProofAgent` itself and serves
        # each dataset folder as a split named by its path (tasks.dataset_dir).
        spawn([sys.executable, "-m", "uvicorn", "harbor_env.server.app:app", "--host", "127.0.0.1",
               "--port", str(server_port), "--ws-ping-timeout", "3600", "--log-config", str(logging_config)], "harbor", {
            "OPENENV_DATASETS": ",".join(str(Path("prepared", "datasets", s).resolve()) for s in ("train", "test")),
            "OPENENV_HARBOR_REWARD_KEY": "reward",
            # OpenEnv (openenv/OpenEnv#1573 from bfe8b9d5) serves a `module:Class` harness only when the operator
            # allows it, since naming one makes the server import that module. Ignored by older OpenEnv.
            "OPENENV_HARBOR_ALLOW_HARNESS": "fineproof_agent.agent:FineProofAgent",
            "FINEPROOF_ROUNDS": str(args.rounds),
            "FINEPROOF_MAX_TOKENS": str(args.max_tokens),
            "FINEPROOF_REQUEST_TIMEOUT_SEC": "3600",  # a 32k-token proof on a loaded engine (tasks.AGENT_TIMEOUT_SEC)
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
                   "--inflight", str(args.inflight), "--learning-rate", str(args.learning_rate),
                   "--warmup-steps", str(args.warmup_steps), "--max-tokens", str(args.max_tokens),
                   "--generations", str(args.generations), "--truncated-reward", args.truncated_reward]
        command += ["--smoke"] if args.smoke else []
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
    census = ThreadCensus(output / "threads.jsonl")
    census.start()
    evals = None
    try:
        if args.eval_checkpoints:
            from checkpoint_evals import CheckpointEvals

            evals = CheckpointEvals(output, bucket=os.environ["FINEPROOF_OUTPUT_BUCKET"], model=args.model,
                                    sandbox=args.sandbox, rounds=args.rounds, max_tokens=args.max_tokens,
                                    space_id=args.space_id, trackio_bucket=args.trackio_bucket,
                                    concurrency=args.eval_concurrency, trackio_run=args.evals_run,
                                    eval_every=args.eval_every)
            evals.start()
        weights = local_checkpoint(args.checkpoint, output) if args.action == "eval" and args.checkpoint else None
        origin = None
        restored = restore_run(output) if args.action == "train" else None
        if restored is not None:  # an earlier Job of this run: continue it (its --resume-from is in its checkpoint)
            weights = str(restored)
            command += ["--resume"]
        elif args.action == "train" and args.resume_from:
            weights = origin = local_checkpoint(args.resume_from, output)  # vLLM serves, the trainer loads, these
            command += ["--resume-from", weights]
        restarts = 0
        while True:
            start_services(weights)
            worker = spawn(command, args.action, {"CUDA_VISIBLE_DEVICES": devices[-1]})
            children.remove(worker)  # healthy() watches the services; the worker's own exit is handled here
            try:
                while worker.poll() is None:
                    healthy()
                    time.sleep(2)
            except RuntimeError as exc:  # vLLM or the Harbor server died under a running trainer
                os.killpg(worker.pid, signal.SIGTERM)
                worker.wait()
                failure = str(exc)
            else:
                failure = None if worker.returncode == 0 else f"{args.action} exited {worker.returncode}"
            finally:
                children.append(worker)  # so a stop (SIGTERM) or the cleanup below also ends it
            try:  # whatever the worker left behind in its process group (spawned helpers of a crashed trainer)
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            latest = latest_checkpoint(output) if args.action == "train" else None
            target = latest or origin  # before this run's first checkpoint, start again from where it began
            if failure is None or args.action != "train" or target is None or restarts >= args.max_restarts:
                break
            # Training died (the services or the trainer): restart all of it from the latest full checkpoint, with
            # vLLM serving that checkpoint's weights, so a slow leak or a crashed engine costs minutes, not the run.
            restarts += 1
            with (output / "restarts.jsonl").open("a") as log:
                log.write(json.dumps({"time": round(time.time(), 1), "restart": restarts, "reason": failure,
                                      "from": Path(target).name}) + "\n")
            stop_children()
            weights = str(target)
            if latest is not None:
                command = [c for c in command if c != "--resume"] + ["--resume"]
        if evals is not None:
            evals.finish()  # the final checkpoint's eval, and any launch that failed earlier
        if failure is not None:
            raise RuntimeError(f"{failure} after {restarts} restart(s); inspect {output}")
    finally:
        census.stop()
        stop_children()

if __name__ == "__main__":
    main()
