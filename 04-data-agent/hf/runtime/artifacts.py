"""Publish logs asynchronously and publish completed checkpoint manifests last."""
import os
import threading
import time
from pathlib import Path

from common import write_json
from bucket_io import sync_with_retry


class Publisher:
    def __init__(self, output):
        self.output = Path(output)
        self.dest = ("hf://buckets/" + os.environ["ARTIFACT_BUCKET"] + "/" + os.environ["RUN_ID"]
                     + "/jobs/" + os.environ["RUN_OWNER"])
        self.stop_event = threading.Event()
        self.published = set()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.loop, daemon=True, name="artifact-publisher")

    def sync(self):
        from huggingface_hub import HfApi
        api = HfApi()
        with self.lock:
            sync_with_retry(api, self.output, self.dest, exclude=["**/*.tmp", "run/checkpoint-*/**", "remote-resume/checkpoint-*/**", "inference-model/**", "trackio/**"])
            for checkpoint in sorted((self.output / "run").glob("checkpoint-*")):
                if checkpoint.name in self.published or not (checkpoint / "checkpoint.saved.json").is_file():
                    continue
                from checkpoint_store import READY, seal
                seal(checkpoint, arm=os.environ['COMPARISON_ARM'], bundle_sha256=os.environ['BUNDLE_SHA256'])
                target = self.dest + "/run/" + checkpoint.name
                print(f"Publishing full checkpoint: {checkpoint.name}", flush=True)
                sync_with_retry(api, checkpoint, target, exclude=[READY])
                # sync_bucket performs content checks for transfer; the consumer verifies native file hashes.
                sync_with_retry(api, checkpoint, target, include=[READY])
                self.published.add(checkpoint.name)
                print(f"Published full checkpoint: {checkpoint.name}", flush=True)
            write_json(self.output / "upload_status.json", {"last_success": time.time(), "destination": self.dest,
                       "published_checkpoints": sorted(self.published)})

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.sync()
            except Exception as exc:
                write_json(self.output / "upload_error.json", {"time": time.time(), "type": type(exc).__name__})
            self.stop_event.wait(60)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        # A full optimizer checkpoint can take longer than two minutes to
        # upload. Allow the same hour of grace reserved for checkpoint work.
        self.thread.join(timeout=3600)
        if self.thread.is_alive():
            raise TimeoutError("Artifact upload did not finish before shutdown")
        self.sync()
