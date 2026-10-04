"""Download the workplace files for this task from the benchmark at a pinned revision, checking every file's hash."""
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

spec = json.load(open(sys.argv[1]))
BASE = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def digest(data: bytes, f: dict) -> bool:
    if "sha256" in f:
        return hashlib.sha256(data).hexdigest() == f["sha256"]
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == f["git_sha1"]


def get(f: dict) -> None:
    url = f"{BASE}/datasets/{spec['dataset']}/resolve/{spec['revision']}/{urllib.parse.quote(f['path'])}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                data = r.read()
            if len(data) == f["size"] and digest(data, f):
                os.makedirs(os.path.dirname(f["target"]), exist_ok=True)
                with open(f["target"], "wb") as out:
                    out.write(data)
                return
            err = "hash mismatch"
        except OSError as e:
            err = str(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{f['path']}: {err}")


with ThreadPoolExecutor(8) as ex:
    list(ex.map(get, spec["files"]))
print(f"fetched {len(spec['files'])} files, {sum(f['size'] for f in spec['files']) / 1e6:.1f} MB")
