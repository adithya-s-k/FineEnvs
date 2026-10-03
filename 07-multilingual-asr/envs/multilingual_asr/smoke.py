"""Check real HTTP transport, binary audio, independent sessions, and every reward path."""

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
from .models import AsrAction
from .runtime import local_server


def probe(url, catalog, languages=None):
    started = time.monotonic()
    coverage, units = Counter(), Counter()
    with ExitStack() as stack:
        first = stack.enter_context(connect(url))
        second = stack.enter_context(connect(url))
        manifest = first.manifest()
        expected = {
            (c["language"], c["family"])
            for c in manifest["counts"]
            if not languages or c["language"] in languages
        }
        checked = set()
        for split in SPLITS:
            for index in range(first.num_tasks(split)):
                task = catalog.at(split, index)
                group = (task["language"], task["family"])
                if group in checked:
                    continue
                checked.add(group)
                task_id = task["task_id"]

                observation = first.reset(task_id=task_id).observation
                repeated = second.reset(task_id=task_id).observation
                assert not observation.done
                assert observation.task_id == repeated.task_id == task_id
                assert observation.asset_sha256 == repeated.asset_sha256
                # Discovery must not leak the answer.
                assert "reference" not in observation.model_dump()
                assert not observation.metrics

                media = requests.get(url + observation.asset_path, timeout=60)
                media.raise_for_status()
                assert (
                    hashlib.sha256(media.content).hexdigest()
                    == observation.asset_sha256
                )
                assert media.headers["content-type"].startswith(observation.mime)

                reference = task["reference"]
                exact = first.step(AsrAction(transcript=reference))
                assert exact.done and exact.reward == 1.0, (group, exact.reward)
                assert exact.observation.metrics.get("exact_match") is True

                # An empty answer deletes every unit: error rate 1, reward 0.
                second.reset(task_id=task_id)
                empty = second.step(AsrAction(transcript=""))
                assert empty.reward == 0.0, (group, empty.reward)

                coverage[f"{group[0]}/{group[1]}"] += 1
                if task["family"] != "language_id":
                    units[task["error_unit"]] += 1
            if checked == expected:
                break
    assert coverage, "Snapshot has no runnable tasks"
    assert checked == expected, f"Missing smoke coverage: {expected - checked}"
    return {
        "status": "passed",
        "snapshot_id": manifest["snapshot_id"],
        "coverage": dict(sorted(coverage.items())),
        # Which units were actually exercised, not which the first task happened to use.
        "error_units": dict(sorted(units.items())),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with ExitStack() as stack:
        snapshot = args.snapshot
        if snapshot is None:
            snapshot = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="asr-smoke-"))
            )
            make_fixture(snapshot)
        url = stack.enter_context(local_server(snapshot))
        result = probe(url, Catalog(snapshot), args.languages)
        result["source"] = "prepared-snapshot" if args.snapshot else "synthetic-fixture"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
