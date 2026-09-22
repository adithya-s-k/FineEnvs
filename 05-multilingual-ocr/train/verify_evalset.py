"""Roll out the frozen evaluation set against a server and check every reward path.

This is the preflight before a training run. For each selected task it confirms the
server serves exactly the pinned bytes, that the oracle answer earns full reward, and
that an empty answer earns none. Run it against a local server and against the Space:
identical results are what makes the two interchangeable for training.
"""

import argparse
import hashlib
import io
import json
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import requests
from nayana_ocr.client import connect
from nayana_ocr.data.corpus import CorpusCatalog
from nayana_ocr.data.evalset import INDEX_CACHE_BYTES, load
from nayana_ocr.training import step_with_judge_retry


# The environment raises on a judge transport failure without consuming the episode and
# tells the caller to retry. Measured at 16 concurrent sessions, 9 of 100 descriptive-VQA
# gradings failed this way and all 9 passed on a serial retry, so the caller must honour
# that contract or a concurrent evaluation reports failures the data does not have.
def _pixel_digest(raw):
    """Digest the decoded image, not its container.

    `section_ocr` and `page_ocr` are re-encoded as PNG by the server, and PNG encoding is
    not byte-reproducible across zlib/libpng builds: the Space returned 40,280 bytes where
    this machine produced 38,482 for pixel-identical output. The encoded hash is therefore
    the identity of one deployment, while the pixels are what the model actually sees and
    what has to match everywhere. Families served as untouched source JPEG are unaffected.
    """
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(repr(rgb.size).encode() + rgb.tobytes()).hexdigest()


def verify(
    url, evalset_path, manifest_path, cache_dir, *, limit=None, families=None, workers=1
):
    started = time.monotonic()
    with ExitStack() as stack:
        catalog = CorpusCatalog(
            manifest_path, cache_dir, index_cache_bytes=INDEX_CACHE_BYTES
        )
        stack.callback(catalog.close)
        client = stack.enter_context(connect(url))
        served = client.manifest()
        record = load(evalset_path, served["snapshot_id"])

        tasks = record["tasks"]
        if families:
            tasks = [t for t in tasks if t["family"] in families]
        if limit:
            # Keep the sample balanced rather than taking a prefix of one language.
            per_group, chosen = defaultdict(int), []
            for task in tasks:
                key = task["family"]
                if per_group[key] < max(
                    1, limit // len(set(t["family"] for t in tasks))
                ):
                    per_group[key] += 1
                    chosen.append(task)
            tasks = chosen[:limit]

        http = requests.Session()
        stack.callback(http.close)
        results, failures = defaultdict(Counter), []
        reencoded = Counter()
        guard = threading.Lock()

        def check(entry, client):
            task_id = entry["task_id"]
            group = f"{entry['language']}/{entry['family']}"
            try:
                observation = client.reset(task_id=task_id).observation
                if observation.task_id != task_id:
                    raise AssertionError("server returned a different task")
                if observation.snapshot_id != served["snapshot_id"]:
                    raise AssertionError("snapshot changed mid-run")
                media = http.get(url.rstrip("/") + observation.asset_path, timeout=180)
                media.raise_for_status()
                served_sha = hashlib.sha256(media.content).hexdigest()
                if served_sha != observation.asset_sha256:
                    raise AssertionError(
                        "served bytes do not match the hash this server reported"
                    )
                if served_sha != entry["asset_sha256"]:
                    # Re-encoded on a different image stack: compare what the model sees.
                    local = catalog.materialize(catalog.get(task_id))
                    expected, _ = catalog.asset_bytes(local["asset_sha256"], task_id)
                    if _pixel_digest(media.content) != _pixel_digest(expected):
                        raise AssertionError(
                            f"asset {served_sha[:12]} != pinned "
                            f"{entry['asset_sha256'][:12]} "
                            "and the decoded pixels differ"
                        )
                    with guard:
                        reencoded[entry["family"]] += 1

                reference = catalog.get(task_id)["reference"]
                if (
                    hashlib.sha256(reference.encode()).hexdigest()
                    != entry["reference_sha256"]
                ):
                    raise AssertionError(
                        "reference no longer matches the frozen digest"
                    )

                oracle = step_with_judge_retry(client, reference)
                if not oracle.done or float(oracle.reward) != 1.0:
                    raise AssertionError(
                        f"oracle answer scored {oracle.reward}, expected 1.0"
                    )

                client.reset(task_id=task_id)
                empty = step_with_judge_retry(client, "")
                if float(empty.reward) != 0.0:
                    raise AssertionError(
                        f"empty answer scored {empty.reward}, expected 0.0"
                    )
                with guard:
                    results[group]["passed"] += 1
            except Exception as error:
                with guard:
                    results[group]["failed"] += 1
                    failures.append(
                        {
                            "task_id": task_id,
                            "language": entry["language"],
                            "family": entry["family"],
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

        if workers <= 1:
            for entry in tasks:
                check(entry, client)
        else:
            # One session per worker: sessions are stateful, so they cannot be shared.
            # The server's NAYANA_MAX_SESSIONS and, for descriptive VQA, its
            # NAYANA_JUDGE_CONCURRENCY must both be at least this worker count.
            sessions = [stack.enter_context(connect(url)) for _ in range(workers)]
            queue = list(tasks)

            def drain(index):
                session = sessions[index]
                for entry in queue[index::workers]:
                    check(entry, session)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(drain, range(workers)))
        return {
            "status": "passed" if not failures else "failed",
            "url": url,
            "evalset_id": record["evalset_id"],
            "snapshot_id": served["snapshot_id"],
            "checked": sum(sum(c.values()) for c in results.values()),
            "passed": sum(c["passed"] for c in results.values()),
            "failed": len(failures),
            "families": dict(Counter(entry["family"] for entry in tasks)),
            "workers": workers,
            "reencoded_pixel_identical": dict(reencoded),
            "failures": failures[:50],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--evalset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--limit", type=int, help="Check a balanced sample instead of all 500 tasks"
    )
    parser.add_argument("--families", nargs="+")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent sessions; the server needs NAYANA_MAX_SESSIONS and "
        "NAYANA_JUDGE_CONCURRENCY at least this high",
    )
    args = parser.parse_args()
    result = verify(
        args.url,
        args.evalset,
        args.manifest,
        args.cache_dir,
        limit=args.limit,
        families=args.families,
        workers=args.workers,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(f"{result['failed']} task(s) failed rollout verification")


if __name__ == "__main__":
    main()
