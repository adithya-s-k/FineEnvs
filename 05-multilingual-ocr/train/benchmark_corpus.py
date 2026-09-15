"""Measure real adapter data paths; no model generation or optimizer work is simulated."""

import argparse
import hashlib
import json
import platform
import random
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import requests
from nayana_ocr.corpus_training import CorpusAPI
from nayana_ocr.data.corpus import CorpusCatalog
from nayana_ocr.runtime import local_server
from nayana_ocr.training import AssetCache, TrainingEnvironment, env_reward

LANGUAGES = ["en", "kn", "hi", "ar"]
FAMILIES = ["section_ocr", "page_ocr", "mcq_vqa"]


def distribution(values):
    if not values:
        return {}
    values = sorted(values)

    def quantile(q):
        index = (len(values) - 1) * q
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "max": values[-1],
    }


def build_plan(catalog, seed):
    rng = random.Random(seed)
    used = set()
    plan = {}
    for name in ("random_eval_serial", "random_eval_concurrent"):
        rows = []
        for language in LANGUAGES:
            for family in FAMILIES:
                for _ in range(100):
                    task = catalog.group_at(
                        "test",
                        language,
                        family,
                        rng.randrange(catalog.group_count("test", language, family)),
                    )
                    if task["block_id"] not in used:
                        used.add(task["block_id"])
                        rows.append(task)
                        break
                else:
                    raise RuntimeError("Could not select an unused evaluation block")
        rng.shuffle(rows)
        plan[name] = rows
    blocks = []
    for language in LANGUAGES * 2:
        candidates = catalog.blocks("train", [language], FAMILIES)
        rng.shuffle(candidates)
        for block in candidates:
            if block["block_id"] in used:
                continue
            rows = catalog.block_tasks(block["block_id"], "train", limit=1000)
            selected = []
            for family in FAMILIES:
                family_rows = [r for r in rows if r["family"] == family]
                if len(family_rows) < 4:
                    break
                selected.extend(rng.sample(family_rows, 4))
            if len(selected) != 12:
                continue
            rng.shuffle(selected)
            blocks.append(
                {"block": block, "rows": [catalog.get(r["task_id"]) for r in selected]}
            )
            used.add(block["block_id"])
            break
        else:
            raise RuntimeError("Could not select a balanced training block")
    plan["training_blocks"] = blocks
    return plan


class TimedClient:
    def __init__(self, client):
        self.client = client
        self.reset_seconds = 0

    def reset(self, **kwargs):
        started = time.perf_counter()
        try:
            return self.client.reset(**kwargs)
        finally:
            self.reset_seconds = time.perf_counter() - started

    def step(self, action):
        return self.client.step(action)

    def close(self):
        self.client.close()


def cache_delta(before, after):
    result = {}
    for name in ("indexes", "groups", "assets"):
        result[name] = {
            key: after[name].get(key, 0) - before[name].get(key, 0)
            for key in (
                "loads",
                "hits",
                "loaded_bytes",
                "load_failures",
                "evictions",
                "prefetch_completed",
                "prefetch_failures",
            )
        }
        result[name]["final_bytes"] = after[name]["bytes"]
        result[name]["max_bytes"] = after[name]["max_bytes"]
        assert after[name]["bytes"] <= after[name]["max_bytes"]
    result["http_received_bytes"] = (
        after["groups"].get("remote_bytes", 0) - before["groups"].get("remote_bytes", 0)
        if after["groups"]["transport"] == "http-range"
        else None
    )
    return result


def operation(env, task):
    record = {key: task[key] for key in ("task_id", "language", "family", "block_id")}
    started = time.perf_counter()
    try:
        content = env.reset(task["task_id"])
        ready = time.perf_counter()
        image = content[0]["image"]
        record["pixels"] = image.width * image.height
        image.close()
        score_start = time.perf_counter()
        scores = env_reward([task["reference"]], [env], [task["task_id"]])
        assert scores == [1.0], scores
        record.update(
            reset_seconds=env.client.reset_seconds,
            image_fetch_decode_seconds=ready - started - env.client.reset_seconds,
            data_ready_seconds=ready - started,
            reward_seconds=time.perf_counter() - score_start,
            status="passed",
        )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    record["total_seconds"] = time.perf_counter() - started
    return record


def summarize(records, elapsed):
    passed = [r for r in records if r["status"] == "passed"]
    return {
        "operations": len(records),
        "successful": len(passed),
        "errors": len(records) - len(passed),
        "wall_seconds": elapsed,
        "successful_operations_per_second": len(passed) / elapsed,
        "latency_seconds": {
            key: distribution([r[key] for r in passed])
            for key in (
                "reset_seconds",
                "image_fetch_decode_seconds",
                "data_ready_seconds",
                "reward_seconds",
                "total_seconds",
            )
        },
        "by_family": {
            family: distribution(
                [r["total_seconds"] for r in passed if r["family"] == family]
            )
            for family in FAMILIES
        },
        "records": records,
    }


def benchmark(
    url, catalog, plan, output, label, phases, warm_seconds=0, warm_workers=(1, 4, 8)
):
    result = {
        "status": "running",
        "label": label,
        "url": url,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": catalog.snapshot_id,
        "client_platform": platform.platform(),
        "python": platform.python_version(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "languages": LANGUAGES,
        "families": FAMILIES,
        "requested_phases": phases,
        "minimum_warm_seconds": warm_seconds,
        "scope": "Real TrainingEnvironment.reset, verified image download/RGB decode, and exact-reference env_reward; no model generation, processor tensorization, GPU, or optimizer",
        "cold_definition": "Application cache misses verified by counters; underlying bucket mount/CDN/OS caches are not flushed",
        "concurrency_scope": "Independent adapter sessions measure service capacity; this does not claim the TRL trainer schedules identical concurrency",
        "phases": [],
    }

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")

    with ExitStack() as stack:
        api = CorpusAPI(url, catalog.snapshot_id)
        stack.callback(api.close)
        http = requests.Session()
        stack.callback(http.close)

        def stats():
            response = http.get(url + "/data/cache", timeout=180)
            response.raise_for_status()
            return response.json()

        result["initial_cache"] = stats()
        timings = []
        for _ in range(12):
            start = time.perf_counter()
            response = http.get(url + "/healthz", timeout=30)
            response.raise_for_status()
            timings.append(time.perf_counter() - start)
        result["health_round_trip_seconds"] = distribution(timings)
        metadata_before = stats()
        timings = []
        for seed in range(12):
            start = time.perf_counter()
            rows = api.sample("test", LANGUAGES, FAMILIES, per_group=1, seed=seed)
            timings.append(time.perf_counter() - start)
            assert len(rows) == 12
        result["metadata_balanced_sample_seconds"] = distribution(timings)
        result["metadata_cache_delta"] = cache_delta(metadata_before, stats())
        save()

        def run_phase(name, rows, workers, *, prime=False, blocks=None):
            print(json.dumps({"phase": name, "event": "started"}), flush=True)
            cache = AssetCache(url)
            setup = time.perf_counter()
            with ExitStack() as phase_stack:
                envs = [
                    TrainingEnvironment(url, cache, catalog.snapshot_id)
                    for _ in range(workers)
                ]
                for env in envs:
                    env.client = TimedClient(env.client)
                    phase_stack.callback(env._close)
                pool = phase_stack.enter_context(
                    ThreadPoolExecutor(max_workers=workers)
                )

                def batch(items):
                    def worker(index):
                        return [
                            operation(envs[index], task)
                            for task in items[index::workers]
                        ]

                    return [
                        r for part in pool.map(worker, range(workers)) for r in part
                    ]

                setup_seconds = time.perf_counter() - setup
                prime_result = None
                if prime:
                    prime_start = time.perf_counter()
                    primed = batch(rows)
                    prime_result = summarize(primed, time.perf_counter() - prime_start)
                    # Keep invalid source candidates in the workload and record
                    # their failures; one bad item must not hide later phases.
                before = stats()
                downloads_before = cache.downloads
                started = time.perf_counter()
                boundaries = []
                if blocks:
                    records = []
                    for index, block in enumerate(blocks):
                        boundary_start = time.perf_counter()
                        reply = api.prefetch(
                            block_ids=[
                                b["block"]["block_id"]
                                for b in blocks[index : index + 2]
                            ]
                        )
                        block_records = batch(block["rows"])
                        records.extend(block_records)
                        boundaries.append(
                            {
                                "block": block["block"],
                                "prefetch_response": reply,
                                "wall_seconds": time.perf_counter() - boundary_start,
                            }
                        )
                else:
                    # Consecutive G=4 completions of each prompt, as in GRPO.
                    selected = rows
                    excluded = []
                    if prime and warm_seconds:
                        valid = {
                            r["task_id"] for r in primed if r["status"] == "passed"
                        }
                        excluded = [
                            r["task_id"] for r in rows if r["task_id"] not in valid
                        ]
                        selected = [r for r in rows if r["task_id"] in valid]
                        if not selected:
                            raise RuntimeError("No valid tasks for sustained warm test")
                    items = (
                        [row for row in selected for _ in range(4)]
                        if prime
                        else selected
                    )
                    records = batch(items)
                    if prime:
                        while time.perf_counter() - started < warm_seconds:
                            records.extend(batch(items))
                elapsed = time.perf_counter() - started
                after = stats()
                phase = {
                    "name": name,
                    "workers": workers,
                    "connection_setup_seconds": setup_seconds,
                    **summarize(records, elapsed),
                    "client_media_downloads": cache.downloads - downloads_before,
                    "client_cache_bytes": cache._bytes,
                    "client_cache_max_bytes": cache.max_bytes,
                    "cache_delta": cache_delta(before, after),
                    "prime": prime_result,
                    "block_boundaries": boundaries,
                }
                if prime and warm_seconds:
                    phase[
                        "invalid_priming_tasks_excluded_from_capacity_measurement"
                    ] = excluded
                    phase["valid_distinct_tasks"] = len(selected)
                    phase["stored_record_limit"] = 256
                    phase["records"] = records[:256]
                    phase["record_scope"] = (
                        "All operations contribute to summaries; first 256 detailed records retained"
                    )
                if name.startswith("random_eval"):
                    phase["distinct_requested_blocks"] = len(
                        {r["block_id"] for r in rows}
                    )
                    phase["all_requested_blocks_application_cold"] = (
                        phase["cache_delta"]["groups"]["loads"]
                        == phase["distinct_requested_blocks"]
                    )
                if blocks:
                    phase["unique_tasks"] = sum(len(b["rows"]) for b in blocks)
                    phase["block_count"] = len(blocks)
                    phase["tasks_per_block"] = 12
                    phase["policy"] = (
                        "Two-block lookahead, four concurrent consumers, 12 sampled tasks per block, no model delay; selected workload is not a full epoch"
                    )
                result["phases"].append(phase)
                save()
                print(
                    json.dumps(
                        {
                            "phase": name,
                            "event": "finished",
                            "ops_per_second": phase["successful_operations_per_second"],
                            "p95_seconds": phase["latency_seconds"][
                                "total_seconds"
                            ].get("p95"),
                            "errors": phase["errors"],
                            "source_loads": phase["cache_delta"]["groups"]["loads"],
                        }
                    ),
                    flush=True,
                )

        if "cold" in phases:
            run_phase("random_eval_serial", plan["random_eval_serial"], 1)
            run_phase("random_eval_concurrent", plan["random_eval_concurrent"], 4)
        for workers in warm_workers:
            if "warm" in phases:
                run_phase(
                    f"warm_grpo_g4_workers{workers}",
                    plan["random_eval_concurrent"],
                    workers,
                    prime=True,
                )
        if "prefetch" in phases:
            run_phase("training_prefetch", [], 4, blocks=plan["training_blocks"])
        result["final_cache"] = stats()
        result["total_measured_operations"] = sum(
            p["operations"] for p in result["phases"]
        )
        result["total_errors"] = sum(p["errors"] for p in result["phases"])
        result["priming_errors"] = sum(
            (p["prime"] or {}).get("errors", 0) for p in result["phases"]
        )
        result["status"] = (
            "passed"
            if result["total_errors"] + result["priming_errors"] == 0
            else "completed_with_errors"
        )
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--url")
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Existing bucket mount for an in-process local deployment",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--warm-seconds", type=float, default=0)
    parser.add_argument(
        "--warm-workers", type=int, nargs="+", choices=(1, 4, 8), default=[1, 4, 8]
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("cold", "warm", "prefetch"),
        default=["cold", "warm", "prefetch"],
    )
    args = parser.parse_args()
    if not 0 <= args.warm_seconds <= 600:
        parser.error("Use --warm-seconds between 0 and 600")
    with ExitStack() as stack:
        directory = Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix="nayana-speed-"))
        )
        catalog = CorpusCatalog(
            args.manifest, directory / "oracle", source_root=args.source_root
        )
        stack.callback(catalog.close)
        plan = build_plan(catalog, args.seed)
        url = args.url or stack.enter_context(
            local_server(
                args.manifest,
                cache_dir=directory / "server",
                source_root=args.source_root,
            )
        )
        benchmark(
            url,
            catalog,
            plan,
            args.output,
            args.label,
            args.phases,
            args.warm_seconds,
            args.warm_workers,
        )


if __name__ == "__main__":
    main()
