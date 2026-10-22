"""Snapshot public metadata, source cards and pinned tau task eligibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_ENV_ROOT = str(ROOT / "envs")
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)
from adapters.tau2 import db_change_task, db_only_task  # noqa: E402

TAU_REVISION = "05736c27219b769d0be507574dd3135010916644"
DATASETS = {
    "nebius/SWE-rebench-openhands-trajectories": "35455389ab51bf5e2306bfd436ef72d0f98bf882",
    "microsoft/AgentRx": "88e871fecb58b2d090449f37ec80b8865594e0b5",
    "PatronusAI/TRAIL": "b424ce63d5973d5dcd7169b1bc3c07ccdee276d1",
    "Kevin355/Who_and_When": "59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c",
    "mcemri/MAST-Data": "95118ac951421753cf1deb87ddea3b01e693c41b",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch", action="store_true", help="Allow public HTTP metadata/task reads"
    )
    args = parser.parse_args()
    cache = ROOT / ".cache/source_checks"
    cache.mkdir(parents=True, exist_ok=True)

    def get(name, url):
        path = cache / name
        if args.fetch:
            with urlopen(url, timeout=45) as response:
                path.write_bytes(response.read())
        if not path.exists():
            raise ValueError("Missing source snapshot; use --fetch during setup")
        payload = path.read_bytes()
        return payload, {"url": url, "sha256": hashlib.sha256(payload).hexdigest()}

    report = {
        "checked_date": date.today().isoformat(),
        "tau_revision": TAU_REVISION,
        "tau_domains": {},
        "datasets": {},
    }
    for domain in ("airline", "telecom", "retail"):
        payload, meta = get(
            f"tau-{domain}.json",
            f"https://raw.githubusercontent.com/sierra-research/tau2-bench/{TAU_REVISION}/"
            f"data/tau2/domains/{domain}/tasks.json",
        )
        tasks = json.loads(payload)
        report["tau_domains"][domain] = {
            **meta,
            "tasks": len(tasks),
            "db_only_eligible": sum(map(db_only_task, tasks)),
            "eligible_task_ids": [t["id"] for t in tasks if db_only_task(t)],
        }
        report["tau_domains"][domain].update(
            db_change_eligible=sum(db_change_task(t, domain) for t in tasks),
            db_change_task_ids=[t["id"] for t in tasks if db_change_task(t, domain)],
            db_only_without_required_write=sum(
                db_only_task(t) and not db_change_task(t, domain) for t in tasks
            ),
        )
    for source, revision in DATASETS.items():
        payload, meta = get(
            source.replace("/", "--") + ".json",
            f"https://huggingface.co/api/datasets/{source}/revision/{revision}",
        )
        data = json.loads(payload)
        card_meta = _card_metadata(source, revision, get, cache)
        report["datasets"][source] = {
            "revision": revision,
            "metadata": meta,
            "card": card_meta,
            "license": data.get("cardData", {}).get("license"),
            "gated": data.get("gated"),
            "published_split_metadata": data.get("cardData", {})
            .get("dataset_info", {})
            .get("splits")
            if isinstance(data.get("cardData", {}).get("dataset_info"), dict)
            else None,
            "downloaded_by_metadata_check": 0,
        }
    output = ROOT / "data/source_checks.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "tau_eligible": {
                    k: v["db_only_eligible"] for k, v in report["tau_domains"].items()
                },
            }
        )
    )


def _card_metadata(source, revision, get, cache):
    try:
        card, card_meta = get(
            source.replace("/", "--") + ".md",
            f"https://huggingface.co/datasets/{source}/raw/{revision}/README.md",
        )
    except HTTPError as exc:
        if exc.code not in (401, 403):
            raise
        card_meta = {"status": "access gate; not accepted", "http_status": exc.code}
        (cache / (source.replace("/", "--") + ".gate.json")).write_text(
            json.dumps(card_meta), encoding="utf-8"
        )
    except ValueError:
        gate = cache / (source.replace("/", "--") + ".gate.json")
        if not gate.exists():
            raise
        card_meta = json.loads(gate.read_text(encoding="utf-8"))
    return card_meta


if __name__ == "__main__":
    main()
