import copy
import json

import pytest
from trace_audit.core import DataError
from trace_audit.core.records import training_split, validate_collection
from trace_audit.core.review import apply_reviews, make_queue

from adapters.abcd import convert as abcd
from tools.data import main as data_main
from tools.data import write_jsonl


def test_review_is_explicit_content_bound_and_preserves_history(rows):
    queue = make_queue(rows, per_source=50)
    unchanged, report = apply_reviews(rows, queue)
    assert unchanged == rows and report["confirmed"] == 0
    first = queue[0]
    first.update(
        confirmed=True,
        reviewer="Rahul",
        failed=True,
        checker="Compared requested result with visible file output",
    )
    changed, report = apply_reviews(rows, queue)
    assert report["confirmed"] == 1
    row = next(r for r in changed if r["record_id"] == first["record_id"])
    assert row["outcome"]["reviewer"] == "Rahul" and row["review_history"]
    with pytest.raises(DataError, match="stale"):
        apply_reviews(changed, queue)
    first["reviewer"] = None
    with pytest.raises(DataError, match="reviewer"):
        apply_reviews(rows, queue)


def legacy_abcd(key, split="test"):
    row = abcd(
        {"convo_id": key, "original": [["customer", "Change my phone."], ["agent", "Done."]]},
        revision="a" * 40,
        split=split,
    )
    row["split"] = split
    return row


def confirm(item, failed=False):
    item.update(
        confirmed=True,
        failed=failed,
        reviewer="unit-test reviewer",
        checker="Explicit visible contradiction",
        label_source="trace_evident",
    )


def test_abcd_test_review_migrates_without_blocking_other_confirmations():
    records = [legacy_abcd("old-test"), legacy_abcd("old-train", "train")]
    queue = make_queue(records)
    for item in queue:
        confirm(item)
    reviewed, report = apply_reviews(records, queue)
    assert report["confirmed"] == 2 and len(report["split_migrations"]) == 1
    migrated = next(r for r in reviewed if r["source"]["original_id"] == "old-test")
    assert migrated["split"] == training_split(migrated["group"]["task_family_id"])
    assert migrated["split"] in {"train", "dev"}
    assert migrated["source"]["source_split"] == "test"
    assert migrated["split_history"][0]["from"] == "test"
    validate_collection(reviewed)
    assert records[0]["split"] == "test" and records[0]["outcome"]["failed"] is None


def test_migration_checks_original_hashes_before_changing_any_family_member():
    first = legacy_abcd("same-task")
    second = copy.deepcopy(first)
    second["record_id"] += "-run-2"
    second["source"]["original_id"] += "-run-2"
    queue = make_queue([first, second])
    for item in queue:
        confirm(item)
    reviewed, report = apply_reviews([first, second], queue)
    assert report["confirmed"] == 2 and report["split_migrations"][0]["records"] == 2
    assert len({r["split"] for r in reviewed}) == 1
    bad = copy.deepcopy(queue)
    bad[1]["record_sha256"] = "stale"
    with pytest.raises(DataError, match="stale"):
        apply_reviews([first, second], bad)
    assert first["split"] == second["split"] == "test"


def test_unconfirmed_abcd_test_records_stay_unlabelled_and_new_imports_avoid_test():
    legacy = legacy_abcd("old-test")
    unchanged, report = apply_reviews([legacy], make_queue([legacy]))
    assert unchanged == [legacy] and report["split_migrations"] == []
    for i in range(50):
        row = abcd(
            {"convo_id": i, "original": [["customer", "Help."]]}, revision="a" * 40, split="test"
        )
        assert row["split"] in {"train", "dev"} and row["source"]["source_split"] == "test"


def test_apply_reviews_cli_accepts_legacy_abcd_confirmation(tmp_path, monkeypatch):
    source, review, output = (
        tmp_path / name for name in ("input.jsonl", "review.jsonl", "output.jsonl")
    )
    records = [legacy_abcd("legacy-test"), legacy_abcd("train-case", "train")]
    queue = make_queue(records)
    for item in queue:
        confirm(item)
    write_jsonl(source, records)
    write_jsonl(review, queue)
    monkeypatch.setattr(
        "sys.argv", ["data.py", "apply-reviews", str(source), str(review), str(output)]
    )
    data_main()
    result = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(result) == 2 and all(r["split"] in {"train", "dev"} for r in result)
    summary = json.loads(output.with_suffix(".jsonl.reviews.json").read_text(encoding="utf-8"))
    assert summary["confirmed"] == 2 and len(summary["split_migrations"]) == 1
