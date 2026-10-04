"""Pin every task image to its content digest, so a converted task always runs the exact image it was checked with.

The dataset names images by tag (docker.io/xiaomimimo/mimo-v2.6-rl-oss:<tag>); tags can move. `images.lock.json`
records tag -> sha256 digest, and conversion reads only the lock file. Refresh it deliberately:

    uv run python -m mimo_harbor.images            # resolve tags missing from the lock
    uv run python -m mimo_harbor.images --refresh  # re-resolve every tag

Manifest HEAD requests do not count against Docker Hub's pull rate limit.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

LOCK = Path(__file__).with_name("images.lock.json")
REPO = "xiaomimimo/mimo-v2.6-rl-oss"
ACCEPT = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.docker.distribution.manifest.v2+json", "application/vnd.oci.image.manifest.v1+json"])


def load() -> dict[str, str]:
    return json.loads(LOCK.read_text()) if LOCK.exists() else {}


def pinned(image: str) -> str:
    """docker.io/xiaomimimo/mimo-v2.6-rl-oss:<tag> -> docker.io/xiaomimimo/mimo-v2.6-rl-oss@sha256:..."""
    repo, tag = image.rsplit(":", 1)
    digest = load().get(tag)
    if not digest:
        raise KeyError(f"{tag} is not in images.lock.json; run `python -m mimo_harbor.images`")
    return f"{repo}@{digest}"


def _token() -> str:
    r = httpx.get("https://auth.docker.io/token", params={"service": "registry.docker.io", "scope": f"repository:{REPO}:pull"},
                  timeout=30)
    r.raise_for_status()
    return r.json()["token"]


def resolve(tags: list[str], workers: int = 16) -> dict[str, str | None]:
    state = {"token": _token()}

    def head(tag: str) -> tuple[str, str | None]:
        for _ in range(4):
            r = httpx.head(f"https://registry-1.docker.io/v2/{REPO}/manifests/{tag}", timeout=30,
                           headers={"Authorization": f"Bearer {state['token']}", "Accept": ACCEPT})
            if r.status_code == 401:   # tokens last five minutes
                state["token"] = _token()
                continue
            if r.status_code == 404:
                return tag, None
            if r.status_code == 200:
                return tag, r.headers["docker-content-digest"]
        raise RuntimeError(f"{tag}: HTTP {r.status_code}")

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(head, tags))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-resolve tags already in the lock")
    args = ap.parse_args()
    from .source import load_rows, image_tag

    tags = sorted({t for t in (image_tag(r) for r in load_rows().values()) if t})
    lock = {} if args.refresh else load()
    todo = [t for t in tags if t not in lock]
    print(f"{len(tags)} tags, resolving {len(todo)}")
    got = resolve(todo)
    missing = sorted(t for t, d in got.items() if d is None)
    lock.update({t: d for t, d in got.items() if d})
    LOCK.write_text(json.dumps(dict(sorted(lock.items())), indent=0) + "\n")
    print(f"lock has {len(lock)} tags" + (f"; not on Docker Hub: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
