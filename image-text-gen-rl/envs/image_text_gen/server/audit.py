"""Optional grading audit log, written to a mounted Hugging Face bucket on the Space.

Every graded step appends one JSON line (task, reward, metrics, every verifier's
transcription, policy IDs). A sampled fraction also stores the verifier-normalised
image as WebP, so reward behaviour on real policy outputs can be reviewed and the
reward iterated. Logging is best effort: a failed write never changes a reward.

  IMAGE_TEXT_GEN_AUDIT_DIR          e.g. /data/audit (the bucket mount); unset disables
  IMAGE_TEXT_GEN_AUDIT_IMAGE_RATE   fraction of steps whose image is kept (default 0.05)
"""

import io
import json
import os
import random
import threading
import time
from functools import lru_cache
from pathlib import Path

from PIL import Image


class AuditLog:
    def __init__(self, directory, image_rate=0.05):
        self.directory = Path(directory)
        self.image_rate = image_rate
        self.lock = threading.Lock()
        self.process = f"{os.getpid()}-{random.getrandbits(24):06x}"

    def record(self, task, reward, metrics, transcriptions, info, policy_id, scoring, png=None):
        try:
            keep_image = png is not None and random.random() < self.image_rate
            sha = info.get("image_sha256", "")
            image_path = None
            if keep_image:
                image_path = f"images/{sha[:2]}/{sha}.webp"
                target = self.directory / image_path
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(io.BytesIO(png)) as image:
                        image.save(target, "WEBP", quality=85)
            line = json.dumps(
                {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "task_id": task["task_id"],
                    "split": task["split"],
                    "target_text": task["target_text"],
                    "reward": reward,
                    "metrics": metrics,
                    "transcriptions": transcriptions,
                    "image_sha256": sha,
                    "image": image_path,
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "grading_policy_id": policy_id,
                    "scoring": scoring,
                },
                ensure_ascii=False,
            )
            day = time.strftime("%Y-%m-%d", time.gmtime())
            path = self.directory / "records" / f"{day}-{self.process}.jsonl"
            with self.lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception as error:  # noqa: BLE001 - auditing must never affect grading
            print(f"audit log write failed: {type(error).__name__}: {error}", flush=True)


@lru_cache(maxsize=1)
def configured_audit():
    directory = os.environ.get("IMAGE_TEXT_GEN_AUDIT_DIR")
    if not directory:
        return None
    return AuditLog(directory, float(os.environ.get("IMAGE_TEXT_GEN_AUDIT_IMAGE_RATE", "0.05")))
