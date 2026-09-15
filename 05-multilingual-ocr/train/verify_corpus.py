"""Audit full-index addressability and measure one cold prefetched block over real transport."""

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import requests
from nayana_ocr.client import connect
from nayana_ocr.corpus_training import CorpusAPI
from nayana_ocr.data.corpus import CorpusCatalog
from nayana_ocr.models import NayanaAction
from nayana_ocr.smoke import probe


def verify(url, manifest_path, output, *, smoke=True, benchmark=True):
    with ExitStack() as stack:
        cache_dir = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="nayana-audit-")
        )
        catalog = CorpusCatalog(manifest_path, cache_dir)
        stack.callback(catalog.close)
        api = CorpusAPI(url, catalog.snapshot_id)
        stack.callback(api.close)
        http = requests.Session()
        stack.callback(http.close)

        def stats():
            response = http.get(url + "/data/cache", timeout=180)
            response.raise_for_status()
            return response.json()

        before = stats()
        checked = 0
        started = time.monotonic()
        for split in ("train", "validation", "test"):
            offset = 0
            for lang in catalog.languages:
                count = sum(
                    c["tasks"]
                    for c in catalog.manifest["counts"]
                    if c["split"] == split and c["language"] == lang
                )
                for local_index in sorted({0, count // 2, count - 1}):
                    expected = catalog.at(split, offset + local_index)
                    response = http.post(
                        url + "/nayana_ocr/task_range",
                        json={
                            "split": split,
                            "start": offset + local_index,
                            "stop": offset + local_index + 1,
                        },
                        timeout=180,
                    )
                    response.raise_for_status()
                    rows = response.json()["tasks"]
                    assert len(rows) == 1 and rows[0]["task_id"] == expected["task_id"]
                    assert "reference" not in rows[0] and rows[0]["asset_path"] is None
                    checked += 1
                offset += count
            assert offset == catalog.count(split)
        after = stats()
        assert after["groups"].get("loads", 0) == before["groups"].get("loads", 0), (
            "Metadata lookup fetched source images"
        )
        result = {
            "snapshot_id": catalog.snapshot_id,
            "url": url,
            "pages": catalog.manifest["pages"],
            "indexed_tasks": sum(c["tasks"] for c in catalog.manifest["counts"]),
            "metadata": {
                "first_middle_last_checks": checked,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "image_group_loads": 0,
            },
        }
        if benchmark:
            blocks = api.blocks("train", ["en"], ["page_ocr"])
            block = blocks[len(blocks) // 2]
            rows = api.block_tasks(block["block_id"], "train", ["page_ocr"], limit=16)
            start_stats = stats()
            start = time.monotonic()
            api.prefetch(block_ids=[block["block_id"]])
            while True:
                loaded = stats()
                if not loaded["groups"]["pending_groups"]:
                    break
                if time.monotonic() - start > 240:
                    raise TimeoutError("Prefetch did not complete")
                time.sleep(0.25)
            prefetch_seconds = time.monotonic() - start
            timings, repeated = [], []
            with connect(url) as client:
                for row in rows:
                    started = time.monotonic()
                    observation = client.reset(task_id=row["task_id"]).observation
                    timings.append(time.monotonic() - started)
                    assert (
                        client.step(
                            NayanaAction(
                                answer=catalog.get(row["task_id"])["reference"]
                            )
                        ).reward
                        == 1
                    )
                    response = http.get(url + observation.asset_path, timeout=180)
                    response.raise_for_status()
                    assert (
                        hashlib.sha256(response.content).hexdigest()
                        == observation.asset_sha256
                    )
                before_repeat = stats()
                for _ in range(4):
                    started = time.monotonic()
                    observation = client.reset(task_id=rows[-1]["task_id"]).observation
                    repeated.append(time.monotonic() - started)
                    assert client.step(NayanaAction(answer="")).reward == 0
                final = stats()
            assert final["groups"].get("loads", 0) == loaded["groups"].get("loads", 0)
            assert final["assets"].get("loads", 0) == before_repeat["assets"].get(
                "loads", 0
            )
            assert final["groups"].get("remote_bytes", 0) == loaded["groups"].get(
                "remote_bytes", 0
            )
            result["block_benchmark"] = {
                "block": block,
                "tasks": len(rows),
                "repetitions": 4,
                "cold_group_loads": loaded["groups"].get("loads", 0)
                - start_stats["groups"].get("loads", 0),
                "prefetch_seconds": round(prefetch_seconds, 4),
                "new_task_reset_seconds": timings,
                "median_new_task_reset_seconds": statistics.median(timings),
                "repeated_reset_seconds": repeated,
                "additional_source_loads_after_prefetch": 0,
                "additional_renders_for_repetitions": 0,
                "transport": final["groups"]["transport"],
                "http_received_bytes": loaded["groups"].get("remote_bytes", 0)
                - start_stats["groups"].get("remote_bytes", 0)
                if final["groups"]["transport"] == "http-range"
                else None,
                "measurement_scope": "one block, serial requests, no model generation; not a throughput benchmark",
            }
        if smoke:
            # Two independent consumers use eight WebSocket sessions, within the
            # default 16-session limit, and exercise concurrent source/cache reads.
            started = time.monotonic()
            partitions = [
                part
                for part in (catalog.languages[::2], catalog.languages[1::2])
                if part
            ]
            with ThreadPoolExecutor(max_workers=2) as executor:
                checks = list(
                    executor.map(
                        lambda languages: probe(url, catalog, languages),
                        partitions,
                    )
                )
            coverage = {
                key: value
                for check in checks
                for key, value in check["coverage"].items()
            }
            result["service_smoke"] = {
                "status": "passed",
                "snapshot_id": catalog.snapshot_id,
                "coverage": coverage,
                "exact_reference_checked": True,
                "adapter_media_downloads": sum(
                    check["adapter_media_downloads"] for check in checks
                ),
                "concurrent_consumers": len(partitions),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        result["cache"] = stats()
        result["status"] = "passed"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "status": result["status"],
                    "metadata": result["metadata"],
                    "groups": list(result.get("service_smoke", {}).get("coverage", {})),
                },
                indent=2,
            )
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    verify(
        args.url.rstrip("/"),
        args.manifest,
        args.output,
        smoke=not args.skip_smoke,
        benchmark=not args.skip_benchmark,
    )


if __name__ == "__main__":
    main()
