"""Publish logs asynchronously and publish completed checkpoint manifests last."""
import os
import threading
import time
from pathlib import Path

from common import write_json


class Publisher:
    def __init__(self, output):
        self.output = Path(output)
        self.dest = ("hf://buckets/" + os.environ["ARTIFACT_BUCKET"] + "/" + os.environ["RUN_ID"]
                     + "/jobs/" + os.environ["RUN_OWNER"])
        self.stop_event = threading.Event()
        self.published = set()
        self.published_markers = {}
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.loop, daemon=True, name="artifact-publisher")

    def sync(self):
        from huggingface_hub import HfApi
        api = HfApi()
        with self.lock:
            api.sync_bucket(str(self.output), self.dest, exclude=["**/*.tmp", "run/checkpoint-*/**", "remote-resume/checkpoint-*/**", "parent/**", "inference-model/**", "trackio/**"], quiet=True)
            for checkpoint in sorted((self.output / "run").glob("checkpoint-*")):
                saved = checkpoint / "checkpoint.saved.json"
                if not saved.is_file():
                    continue
                marker = saved.read_bytes()
                if self.published_markers.get(checkpoint.name) == marker:
                    continue
                from checkpoint_store import READY, seal
                seal(checkpoint, arm=os.environ['COMPARISON_ARM'], bundle_sha256=os.environ['BUNDLE_SHA256'])
                target = self.dest + "/run/" + checkpoint.name
                api.sync_bucket(str(checkpoint), target, exclude=[READY], quiet=True)
                # sync_bucket performs content checks for transfer; the consumer verifies native file hashes.
                api.sync_bucket(str(checkpoint), target, include=[READY], quiet=True)
                self.published.add(checkpoint.name)
                self.published_markers[checkpoint.name] = marker
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
        self.thread.join(timeout=120)
        if self.thread.is_alive():
            raise TimeoutError("Artifact upload did not finish before shutdown")
        self.sync()
