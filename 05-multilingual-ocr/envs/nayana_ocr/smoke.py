"""Check real HTTP, binary media, concurrent WebSockets, and the TRL adapter on CPU."""

import argparse
import hashlib
import json
import tempfile
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import requests

from .client import connect
from .data.catalog import SPLITS, Catalog
from .fixtures import make_fixture
from .models import NayanaAction
from .runtime import local_server
from .training import AssetCache, TrainingEnvironment, env_reward, task_rows


def probe(url, catalog=None):
    started = time.monotonic()
    with ExitStack() as stack:
        first = stack.enter_context(connect(url))
        second = stack.enter_context(connect(url))
        manifest = first.manifest()
        coverage = Counter()
        cache = AssetCache(url)
        environments = [
            TrainingEnvironment(url, cache, manifest["snapshot_id"]) for _ in range(2)
        ]
        for env in environments:
            stack.callback(env._close)
        checked = set()
        for split in SPLITS:
            for task in task_rows(url, split):
                group = (task["language"], task["family"])
                if group in checked:
                    continue
                checked.add(group)
                task_id = task["task_id"]
                obs = first.reset(task_id=task_id).observation
                repeated = second.reset(task_id=task_id).observation
                assert not obs.done and obs.task_id == repeated.task_id
                assert obs.asset_sha256 == repeated.asset_sha256
                assert "reference" not in obs.model_dump() and not obs.metrics
                media = requests.get(url + obs.asset_path, timeout=60)
                media.raise_for_status()
                assert hashlib.sha256(media.content).hexdigest() == obs.asset_sha256
                assert media.headers["content-type"].startswith(obs.mime)
                assert first.step(NayanaAction(answer="")).reward == 0.0
                target = catalog.get(task_id)["reference"] if catalog else None
                if target:
                    result = second.step(NayanaAction(answer=target))
                    assert (
                        result.done
                        and result.reward == 1.0
                        and result.observation.metrics["exact_match"]
                    )
                before = cache.downloads
                for env in environments:
                    blocks = env.reset(task_id=task_id)
                    assert blocks[0]["image"].size == (obs.width, obs.height)
                    blocks[0]["image"].close()
                assert cache.downloads - before <= 1
                logged = {}
                rewards = env_reward(
                    [target or "", ""],
                    environments,
                    [task_id, task_id],
                    log_metric=lambda name, value, values=logged: values.update(
                        {name: value}
                    ),
                )
                assert logged[f"nayana/{group[0]}/{group[1]}/reward"] == (
                    0.5 if target else 0.0
                )
                assert rewards == ([1.0, 0.0] if target else [0.0, 0.0])
                coverage[f"{group[0]}/{group[1]}"] += 1
    assert coverage, "Snapshot has no runnable tasks"
    return {
        "status": "passed",
        "snapshot_id": manifest["snapshot_id"],
        "coverage": dict(coverage),
        "exact_reference_checked": catalog is not None,
        "adapter_media_downloads": cache.downloads,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--snapshot", type=Path)
    choice.add_argument("--url")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with ExitStack() as stack:
        if args.url:
            result = probe(args.url)
        else:
            snapshot = args.snapshot
            if snapshot is None:
                snapshot = Path(
                    stack.enter_context(
                        tempfile.TemporaryDirectory(prefix="nayana-smoke-")
                    )
                )
                make_fixture(snapshot)
            url = stack.enter_context(local_server(snapshot))
            result = probe(url, Catalog(snapshot))
        result["source"] = (
            "remote"
            if args.url
            else "prepared-corpus"
            if args.snapshot
            else "synthetic-fixture"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
