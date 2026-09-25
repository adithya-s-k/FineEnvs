"""Lightweight inference/GPU samples outside the evaluator and optimizer loops."""
import json
import re
import subprocess
import threading
import time

METRICS = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
           "prompt_tokens_total", "generation_tokens_total", "request_success_total",
           "request_queue_time_seconds_sum", "request_queue_time_seconds_count",
           "time_to_first_token_seconds_sum", "time_to_first_token_seconds_count")


class Telemetry:
    def __init__(self, output, base_url="http://127.0.0.1:8000"):
        self.path = output / "inference_metrics.jsonl"
        self.base_url = base_url.rstrip("/")
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name="inference-telemetry")

    def run(self):
        import httpx
        with httpx.Client(timeout=5) as client:
            while not self.stop_event.is_set():
                row = {"time": time.time()}
                try:
                    response = client.get(self.base_url + "/metrics").raise_for_status()
                    samples = {}
                    for line in response.text.splitlines():
                        match = re.match(r"vllm:([^ {]+)(?:\{[^}]*\})? ([^ ]+)", line)
                        if match and match[1] in METRICS:
                            samples[match[1]] = samples.get(match[1], 0) + float(match[2])
                    row["vllm"] = samples
                    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
                                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                    row["gpu_csv"] = gpu.stdout.strip().splitlines() if gpu.returncode == 0 else []
                except Exception as exc:
                    row["error_type"] = type(exc).__name__
                with self.path.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                self.stop_event.wait(30)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=15)
