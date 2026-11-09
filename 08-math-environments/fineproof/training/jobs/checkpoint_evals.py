"""Inside a training Job: send each saved checkpoint to the bucket, then to an eval HF Job of its own.

The trainer writes `checkpoint-N/trainer_state.json` last, so a checkpoint is complete once that file parses. Each
complete checkpoint is copied to `hf://buckets/<bucket>/<run>/checkpoint-N` and an eval Job is submitted through
jobs/hf_job.py (which mounts the bucket at /outputs). Every eval scores the same fixed benchmark as step 0, with the
same judge, and logs to one Trackio run (`<run>-evals`) at the checkpoint's step, so the curve reads across steps.

What was submitted goes to <output>/checkpoint_evals.jsonl and the subprocess output to checkpoint_evals.log. A
failure is logged and retried at the next scan; it never stops training.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The benchmark step 0 was scored on (results/fineproof-bench-4b-step0.json): 64 validated test problems x 4.
BENCHMARK = {"task_list": "benchmarks/test-validated-64.txt", "samples": 4, "concurrency": 64}
EVAL_JUDGE = {"JUDGE_MODEL": "openai/gpt-oss-20b:groq,openai/gpt-oss-20b:featherless-ai", "JUDGE_SAMPLES": "2"}


def checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def complete(path: Path) -> bool:
    """trainer_state.json is the last file a checkpoint gets; weights must be there too."""
    try:
        json.loads((path / "trainer_state.json").read_text())
    except (OSError, ValueError):
        return False
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


class CheckpointEvals(threading.Thread):
    def __init__(self, output: Path, *, bucket: str, model: str, sandbox: str, rounds: int, max_tokens: int,
                 space_id: str | None, trackio_bucket: str | None, concurrency: int = BENCHMARK["concurrency"],
                 scan_every_s: float = 60.0, dry_run: bool = False):
        super().__init__(name="checkpoint-evals", daemon=True)
        self.output, self.bucket, self.model, self.sandbox = Path(output), bucket, model, sandbox
        self.rounds, self.max_tokens, self.space_id, self.trackio_bucket = rounds, max_tokens, space_id, trackio_bucket
        self.scan_every_s, self.dry_run, self.concurrency = scan_every_s, dry_run, concurrency
        self.run_name = self.output.name
        self.done: set[int] = set()
        self.stopping = threading.Event()
        self.lock = threading.Lock()

    def commands(self, path: Path, step: int) -> tuple[list[str], list[str]]:
        sync = ["hf", "buckets", "sync", str(path), f"hf://buckets/{self.bucket}/{self.run_name}/{path.name}",
                "--format", "quiet"]
        submit = [sys.executable, "jobs/hf_job.py", "eval", "--name", f"{self.run_name}-eval-{step}",
                  "--bucket", self.bucket, "--model", self.model,
                  "--checkpoint", f"/outputs/{self.run_name}/{path.name}", "--step", str(step),
                  "--task-list", BENCHMARK["task_list"], "--samples", str(BENCHMARK["samples"]),
                  "--concurrency", str(self.concurrency), "--rounds", str(self.rounds),
                  "--max-tokens", str(self.max_tokens), "--sandbox", self.sandbox, "--timeout", "3h",
                  "--prepare-args", "--train-tasks 16",  # an eval needs only the test split
                  "--trackio-run", f"{self.run_name}-evals"]
        if self.space_id:
            submit += ["--space-id", self.space_id]
        if self.trackio_bucket:
            submit += ["--trackio-bucket", self.trackio_bucket]
        return sync, submit + ([] if self.dry_run else ["--submit"])

    def launch(self, path: Path, step: int) -> None:
        sync, submit = self.commands(path, step)
        env = {**os.environ, **EVAL_JUDGE}
        with (self.output / "checkpoint_evals.log").open("a") as log:
            log.write(f"\n== {time.strftime('%Y-%m-%d %H:%M:%S')} checkpoint-{step}\n")
            log.flush()
            if not self.dry_run:
                subprocess.run(sync, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
            done = subprocess.run(submit, check=True, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
            log.write(done.stdout + done.stderr)
        url = next((line.strip() for line in done.stdout.splitlines() if line.strip().startswith("http")), None)
        with (self.output / "checkpoint_evals.jsonl").open("a") as handle:
            handle.write(json.dumps({"step": step, "checkpoint": path.name, "job": url, "dry_run": self.dry_run,
                                     "time": round(time.time(), 1)}) + "\n")

    def scan(self) -> None:
        with self.lock:
            for path in sorted(self.output.glob("checkpoint-*"), key=lambda p: checkpoint_step(p) or -1):
                step = checkpoint_step(path)
                if step is None or step in self.done or not complete(path):
                    continue
                try:
                    self.launch(path, step)
                    self.done.add(step)
                except Exception as exc:  # noqa: BLE001 - retried at the next scan; training goes on
                    with (self.output / "checkpoint_evals.log").open("a") as log:
                        log.write(f"checkpoint-{step}: {type(exc).__name__}: {exc}\n")

    def run(self) -> None:
        while not self.stopping.wait(self.scan_every_s):
            self.scan()

    def finish(self) -> None:
        """After training: launch whatever is left (the final checkpoint), retrying a failed launch once."""
        self.stopping.set()
        self.join(timeout=7200)
        for _ in range(2):
            self.scan()
