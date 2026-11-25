"""Content-bound human review queues and label updates."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import defaultdict

from .records import DataError, training_split, validate_collection, validate_record


def fingerprint(row: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            row, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def assert_training_audits(records: list[dict], *, audit_records: list[dict] | None = None) -> None:
    """Require revision-bound human audits for SWE grader labels.

    Each pinned revision needs the specified 50-label audit. If that audit found
    grader disagreements, remaining grader-only labels require individual review.
    Human-reviewed outcomes remain usable without trusting the original grader.
    Audit evidence covers the entire revision, independently of split or rendering admission.
    """
    groups = defaultdict(list)
    for row in records:
        if (
            row["source"]["dataset"].casefold() == "nebius/swe-rebench-openhands-trajectories"
            and row["source"].get("data_kind") != "smoke_fixture"
        ):
            groups[row["source"]["revision"]].append(row)
    evidence = defaultdict(list)
    for row in records if audit_records is None else audit_records:
        if (
            row["source"]["dataset"].casefold() == "nebius/swe-rebench-openhands-trajectories"
            and row["source"].get("data_kind") != "smoke_fixture"
        ):
            evidence[row["source"]["revision"]].append(row)
    for revision, rows in groups.items():
        if not any(row["outcome"]["label_source"] == "verified_outcome" for row in rows):
            continue
        reviewed = [
            row
            for row in evidence[revision]
            if row["outcome"]["label_source"] in {"expert", "trace_evident"}
            and any(
                old.get("label_source") == "verified_outcome" and type(old.get("failed")) is bool
                for old in row.get("review_history", [])
            )
        ]
        disagreements = any(
            next(
                old
                for old in row["review_history"]
                if old.get("label_source") == "verified_outcome"
            )["failed"]
            != row["outcome"]["failed"]
            for row in reviewed
        )
        if len(reviewed) < 50 or disagreements:
            raise DataError(
                f"SWE revision {revision}: training blocked pending 50 human label audits "
                "with no grader disagreements, "
                "or individual human review of every training label"
            )


def make_queue(records: list[dict], *, per_source: int = 50, seed: int = 42) -> list[dict]:
    if type(per_source) is not int or per_source < 1:
        raise DataError("Positive review sample size required")
    groups = defaultdict(list)
    for row in records:
        validate_record(row)
        groups[row["source"]["dataset"]].append(row)
    rng, queue = random.Random(seed), []
    for source in sorted(groups):
        sample = sorted(groups[source], key=lambda row: row["record_id"])
        rng.shuffle(sample)
        for row in sample[:per_source]:
            queue.append(
                {
                    "record_id": row["record_id"],
                    "record_sha256": fingerprint(row),
                    "source": source,
                    "existing_outcome": copy.deepcopy(row["outcome"]),
                    "confirmed": False,
                    "failed": None,
                    "label_source": "expert",
                    "reviewer": None,
                    "checker": None,
                    "failures": None,
                }
            )
    return queue


def apply_reviews(records: list[dict], reviews: list[dict]) -> tuple[list[dict], dict]:
    validate_collection(records)
    rows = {row["record_id"]: copy.deepcopy(row) for row in records}
    migrations = _review_migrations(rows, reviews)
    approved, changed = 0, 0
    by_source = defaultdict(lambda: {"confirmed": 0, "compared_existing": 0, "disagreements": 0})
    for review in reviews:
        key = review.get("record_id")
        row = rows[key]
        if review.get("confirmed") is not True:
            continue
        _validate_review(row, review)
        previous = row["outcome"]["failed"]
        audit = by_source[row["source"]["dataset"]]
        audit["confirmed"] += 1
        audit["compared_existing"] += previous is not None
        audit["disagreements"] += previous is not None and previous != review["failed"]
        changed += previous is not None and previous != review["failed"]
        row.setdefault("review_history", []).append(copy.deepcopy(row["outcome"]))
        row["outcome"] = {
            key: copy.deepcopy(review[key])
            for key in ("failed", "label_source", "reviewer", "checker")
        }
        if review.get("failures") is not None:
            row["failures"] = copy.deepcopy(review["failures"])
        validate_record(row)
        approved += 1
    result = list(rows.values())
    validate_collection(result)
    for audit in by_source.values():
        audit["estimated_label_error_rate"] = (
            audit["disagreements"] / audit["compared_existing"]
            if audit["compared_existing"]
            else None
        )
    return result, {
        "confirmed": approved,
        "changed_existing_labels": changed,
        "sampled": len(reviews),
        "pending": len(reviews) - approved,
        "per_source": dict(by_source),
        "split_migrations": migrations,
    }


def _review_migrations(rows, reviews):
    # Validate every review against the original evidence before changing any split or label.
    migration_families = set()
    checked = set()
    for review in reviews:
        key = review.get("record_id")
        if key not in rows or key in checked:
            raise DataError("Unknown or duplicate review record")
        checked.add(key)
        if review.get("record_sha256") != fingerprint(rows[key]):
            raise DataError("Review is stale: record content changed")
        row = rows[key]
        if (
            review.get("confirmed") is True
            and row["source"]["dataset"].casefold() == "asappresearch/abcd"
            and row["split"] == "test"
        ):
            migration_families.add(row["group"]["task_family_id"])
    return _migrate_families(rows, migration_families)


def _migrate_families(rows, migration_families):
    migrations = []
    for family in sorted(migration_families):
        destination = training_split(family)
        members = [r for r in rows.values() if r["group"]["task_family_id"] == family]
        if any(r["source"]["dataset"].casefold() != "asappresearch/abcd" for r in members):
            raise DataError("Cannot migrate an ABCD family shared with another source")
        for row in members:
            row.setdefault("split_history", []).append(
                {
                    "from": row["split"],
                    "to": destination,
                    "reason": (
                        "Legacy ABCD family is train/dev only under the held-out-source protocol"
                    ),
                }
            )
            row["split"] = destination
        migrations.append(
            {"task_family_id": family, "from": "test", "to": destination, "records": len(members)}
        )
    return migrations


def _validate_review(row, review):
    if row["source"]["dataset"].casefold() in {"mast", "mast-data", "mcemri/mast-data"}:
        raise DataError("MAST stays analysis-only in v1")
    if type(review.get("failed")) is not bool or review.get("label_source") not in {
        "trace_evident",
        "expert",
    }:
        raise DataError("Human review needs a boolean outcome and human label_source")
    if (
        not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or not review.get("checker")
    ):
        raise DataError("Human review needs reviewer and outcome evidence/rubric")
