from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .scoring import CATEGORIES
from .validation import DataError, validate_atif

HELDOUT_DATASETS = {
    "microsoft/agentrx": "microsoft/AgentRx",
    "patronusai/trail": "PatronusAI/TRAIL",
    "kevin355/who_and_when": "Kevin355/Who_and_When",
}


def heldout_source(row: dict) -> str | None:
    source = row["source"]
    dataset = source["dataset"].casefold()
    if dataset in HELDOUT_DATASETS:
        return HELDOUT_DATASETS[dataset]
    if dataset == "sierra-research/tau2-bench" and source.get("domain") == "retail":
        return "tau2:retail"
    return None


def unlabelled_reason(row: dict) -> str:
    return row["source"].get("label_exclusion_reason") or "unlabelled"


def scoring_exclusion_reason(row: dict, *, allow_fixtures: bool = False) -> str | None:
    """Shared admission for runtime, preparation, sampling and held-out gates.

    Legacy DB-only exports stay reviewable, but cannot bypass task eligibility.
    A named human review may establish an outcome on a read-only task.
    """
    if row["outcome"]["failed"] is None:
        return unlabelled_reason(row)
    if row["source"].get("data_kind") == "smoke_fixture":
        return None if allow_fixtures else "fixture_not_production_data"
    if (
        row["source"]["dataset"].casefold() == "sierra-research/tau2-bench"
        and row["outcome"]["label_source"] == "verified_outcome"
    ):
        checker = row["outcome"]["checker"]
        if not isinstance(checker, dict) or checker.get("task_requires_db_change") is not True:
            return "unverified_db_task_eligibility"
    return None


def training_split(family: str, *, seed: int = 42) -> str:
    bucket = (
        int(hashlib.sha256(json.dumps((seed, family), sort_keys=True).encode()).hexdigest()[:8], 16)
        % 100
    )
    return "train" if bucket < 85 else "dev"


def validate_record(row: dict) -> None:
    if not isinstance(row, dict):
        raise DataError("Record must be an object")
    for key in (
        "record_id",
        "source",
        "task",
        "trajectory",
        "outcome",
        "failures",
        "group",
        "split",
    ):
        if key not in row:
            raise DataError(f"Missing record field: {key}")
    if not isinstance(row["record_id"], str) or not row["record_id"]:
        raise DataError("Nonempty record_id required")
    source = _validate_source(row)
    _validate_task(row)
    outcome = _validate_outcome(row, source)
    validate_atif(row["trajectory"])
    _validate_failures(row)
    _validate_split(row, source, outcome)


def validate_collection(records: list[dict], *, validate_rows: bool = True) -> None:
    identities, groups = set(), {}
    originals = {}
    for row in records:
        if validate_rows:
            validate_record(row)
        if row["record_id"] in identities:
            raise DataError("Duplicate record_id")
        identities.add(row["record_id"])
        group = row["group"]["task_family_id"]
        identity = (row["source"]["dataset"], row["source"]["original_id"])
        if group in groups and groups[group] != row["split"]:
            raise DataError("Related task family crosses splits")
        if identity in originals and originals[identity] != row["split"]:
            raise DataError("Original source record crosses splits")
        groups[group] = originals[identity] = row["split"]


def _unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise DataError("Duplicate JSON key")
        obj[key] = value
    return obj


def load_records(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_unique,
                    parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)),
                )
                validate_record(row)
            except (ValueError, TypeError, KeyError) as exc:
                raise DataError(f"Invalid record at line {index}: {exc}") from exc
            records.append(row)
    validate_collection(records, validate_rows=False)
    return records


def _validate_source(row):
    source = row["source"]
    if not isinstance(source, dict) or any(
        not isinstance(source.get(key), str) or not source[key]
        for key in ("dataset", "revision", "license", "original_id")
    ):
        raise DataError("Source needs dataset, pinned revision, license, original_id")
    if "identity_strings" in source and (
        not isinstance(source["identity_strings"], list)
        or any(not isinstance(value, str) for value in source["identity_strings"])
    ):
        raise DataError("identity_strings must be an array of strings")
    for field in ("agent_model", "user_model"):
        value = source.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise DataError("Model provenance must be nonempty text or null")
    if "label_exclusion_reason" in source and (
        not isinstance(source["label_exclusion_reason"], str)
        or not source["label_exclusion_reason"]
    ):
        raise DataError("label_exclusion_reason must be nonempty text")
    if source["revision"].casefold() in {"main", "master", "latest", "head"} or source[
        "license"
    ].casefold() in {"unknown", "none", "unlicensed", "pending"}:
        raise DataError("Pinned revision and resolved licence required")
    return source


def _validate_task(row):
    if (
        not isinstance(row["task"], dict)
        or not isinstance(row["task"].get("instruction"), str)
        or not row["task"]["instruction"].strip()
    ):
        raise DataError("Nonempty task instruction required")
    if any(key not in row["task"] for key in ("policy", "success_criteria")):
        raise DataError("Task requires policy and success_criteria (text or null)")
    if row["task"]["success_criteria"] is not None and not isinstance(
        row["task"]["success_criteria"], str
    ):
        raise DataError("success_criteria must be text or null")
    if row["task"].get("policy") is not None and not isinstance(row["task"]["policy"], str):
        raise DataError("Policy must be text or null")
    if row["split"] not in {"train", "dev", "test"}:
        raise DataError("Invalid reviewer split")
    if (
        not isinstance(row["group"], dict)
        or not isinstance(row["group"].get("task_family_id"), str)
        or not row["group"]["task_family_id"]
    ):
        raise DataError("Task family required")


def _validate_outcome(row, source):
    outcome = row["outcome"]
    if not isinstance(outcome, dict) or any(
        key not in outcome for key in ("failed", "label_source", "checker", "reviewer")
    ):
        raise DataError("Outcome requires failed, label_source, checker and reviewer")
    if outcome.get("failed") is not None and type(outcome["failed"]) is not bool:
        raise DataError("Outcome must be bool or null")
    kind = outcome.get("label_source")
    if kind not in {"verified_outcome", "trace_evident", "expert", "none"}:
        raise DataError("Unknown label source")
    if outcome.get("failed") is not None:
        if kind == "none" or not outcome.get("checker"):
            raise DataError("Scored outcome lacks a trusted checker")
        if kind in {"trace_evident", "expert"} and (
            not isinstance(outcome.get("reviewer"), str) or not outcome["reviewer"].strip()
        ):
            raise DataError("Human-confirmed labels require a named reviewer")
        if source["dataset"].casefold() in {"mast", "mast-data", "mcemri/mast-data"}:
            raise DataError("MAST's model-only labels cannot enter v1 reward")
    return outcome


def _validate_failures(row):
    ids = {step["step_id"] for step in row["trajectory"]["steps"]}
    if not isinstance(row["failures"], list):
        raise DataError("failures must be an array")
    for failure in row["failures"]:
        if (
            not isinstance(failure, dict)
            or failure.get("category") not in CATEGORIES
            or type(failure.get("step_id")) is not int
            or failure["step_id"] not in ids
            or type(failure.get("critical")) is not bool
        ):
            raise DataError("Invalid failure annotation")
        if "source_label" not in failure:
            raise DataError("Failure annotation requires original source_label")
    if sum(f["critical"] for f in row["failures"]) > 1:
        raise DataError("At most one annotated critical step per trace")


def _validate_split(row, source, outcome):
    dataset = source["dataset"].casefold()
    if dataset in HELDOUT_DATASETS and row["split"] != "test":
        raise DataError("External attribution sources are test-only")
    if (
        dataset == "sierra-research/tau2-bench"
        and source.get("domain") == "retail"
        and row["split"] != "test"
    ):
        raise DataError("Retail is a held-out domain")
    training_source = dataset == "nebius/swe-rebench-openhands-trajectories" or (
        dataset == "sierra-research/tau2-bench" and source.get("domain") in {"airline", "telecom"}
    )
    if training_source and row["split"] == "test":
        raise DataError("Training sources are train/dev only; test must hold out sources/domains")
    if (
        row["split"] == "test"
        and outcome["failed"] is not None
        and source.get("data_kind") != "smoke_fixture"
        and heldout_source(row) is None
    ):
        raise DataError(
            "Scored test records must belong to the configured held-out sources/domains"
        )
