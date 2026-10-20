"""Downloads for the data build: polite retries, atomic writes, sha256 pins."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

AGENT = "agentenv-airops data build (+https://github.com/earakely-scale/agentenv-airops-plugin)"
RETRY_STATUS = {429, 500, 502, 503, 504}
LOG = "fetched.json"


def get(url: str, *, data: bytes | None = None, headers: dict | None = None, opener=None, attempts: int = 6,
        timeout: int = 300, pause: float = 0.0) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": AGENT, **(headers or {})})
    open_ = (opener or urllib.request.build_opener()).open
    for attempt in range(attempts):
        try:
            with open_(req, timeout=timeout) as resp:
                blob = resp.read()
            time.sleep(pause)
            return blob
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
        time.sleep(min(60.0, 2.0 * 2**attempt))
    raise AssertionError("unreachable")


def save(path: Path, blob: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(blob)
    os.replace(tmp, path)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj) -> str:
    """The sha256 of an object as sorted-key compact JSON: the pin of parsed records rather than of a web page."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def record(inputs: Path, rel: str, url: str, retrieved: str) -> None:
    """Remember where and when an input was fetched, for the manifest and the SOURCE.md files."""
    log = inputs / LOG
    entries = json.loads(log.read_text()) if log.exists() else {}
    entries[rel] = {"url": url, "retrieved": retrieved}
    save(log, (json.dumps(entries, indent=1, sort_keys=True) + "\n").encode())


def fetched(inputs: Path) -> dict:
    log = inputs / LOG
    return json.loads(log.read_text()) if log.exists() else {}
