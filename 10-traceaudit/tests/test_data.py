import copy
import json

import pytest
from conftest import ROOT
from test_adapters import swe_row, tau_row
from trace_audit.core import DataError, load_records, validate_record
from trace_audit.core.records import training_split

from adapters.swe_rebench import convert as swe
from adapters.tau2 import convert as tau


def test_pinned_source_eligibility_report():
    report = json.loads((ROOT / "data/source_checks.json").read_text())
    assert {key: value["db_only_eligible"] for key, value in report["tau_domains"].items()} == {
        "airline": 44,
        "telecom": 0,
        "retail": 2,
    }
    assert {key: value["db_change_eligible"] for key, value in report["tau_domains"].items()} == {
        "airline": 21,
        "telecom": 0,
        "retail": 2,
    }


def test_duplicate_json_and_groups_rejected(tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps(rows[0]).replace('"split": "train"', '"split": "test", "split": "train"'),
        encoding="utf-8",
    )
    with pytest.raises(DataError):
        load_records(path)


def test_loading_validates_each_record_once(rows, monkeypatch):
    import trace_audit.core.records as records

    checked = []
    validate = records.validate_record

    def track(row):
        checked.append(row["record_id"])
        validate(row)

    monkeypatch.setattr(records, "validate_record", track)
    assert load_records(ROOT / "data/sample.jsonl") == rows
    assert checked == [row["record_id"] for row in rows]


def test_official_harbor_all_stored_fixtures(rows):
    from tools.validate_harbor import prepare

    prepare()
    from harbor.utils.trajectory_validator import TrajectoryValidator

    for row in rows:
        validator = TrajectoryValidator()
        assert validator.validate(row["trajectory"], validate_images=False), validator.errors


def test_training_families_never_enter_test():
    assert set(training_split(f"task-{i}") for i in range(1000)) == {"train", "dev"}
    for i in range(50):
        raw = tau_row()
        raw["task"]["id"] = str(i)
        assert tau(raw, revision="a" * 40)["split"] in {"train", "dev"}
    with pytest.raises(DataError, match="train/dev"):
        swe(swe_row(), revision="a" * 40, split="test")
    raw = swe(swe_row(), revision="a" * 40)
    raw["split"] = "test"
    with pytest.raises(DataError, match="train/dev"):
        validate_record(raw)  # Hand-authored files cannot bypass the adapter's split rule.


def test_required_wrapper_fields_and_abcd_licence(rows):
    raw = copy.deepcopy(rows[0])
    del raw["task"]["success_criteria"]
    with pytest.raises(DataError, match="success_criteria"):
        validate_record(raw)
    raw = copy.deepcopy(next(r for r in rows if r["failures"]))
    del raw["failures"][0]["source_label"]
    with pytest.raises(DataError, match="source_label"):
        validate_record(raw)
    from argparse import Namespace

    from tools.data import import_source

    with pytest.raises(DataError, match="licence"):
        import_source(Namespace(source="abcd", license="CC-BY-4.0"))


def test_training_split_preserves_legacy_abcd_assignment():
    assert training_split("abcd:9539") == "train"
