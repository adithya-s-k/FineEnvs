import copy
import itertools

import pytest
from trace_audit.core import DataError
from trace_audit.core.sampling import balanced_batch


def test_balancing_caps_sources_without_altering_reward(rows):
    production = copy.deepcopy([r for r in rows[:-1] if r["split"] == "train"])
    for row in production:
        row["source"]["data_kind"] = "test_inputs"
    batch, report = balanced_batch(production, 8)
    assert sum(r["outcome"]["failed"] for r in batch) == 4
    assert max(report["source_counts"].values()) <= 4
    with pytest.raises(DataError):
        balanced_batch(production, 12, source_cap=0.4)


def test_large_balanced_batch_and_single_source_cap(rows):
    records = copy.deepcopy(
        [r for r in rows if r["split"] == "train" and r["outcome"]["failed"] is not None]
    )
    for r in records:
        r["source"]["data_kind"] = "test_inputs"
    batch, report = balanced_batch(records, 4096, max_record_reuse=512)
    assert sum(r["outcome"]["failed"] for r in batch) == 2048
    assert max(report["source_counts"].values()) == 2048
    assert report["max_record_reuse"] <= 512
    with pytest.raises(DataError, match="reuse limit"):
        balanced_batch(records, 4096)
    only = [r for r in records if r["source"]["dataset"] == records[0]["source"]["dataset"]]
    with pytest.raises(DataError):
        balanced_batch(only, 4096)


def test_sampler_handles_all_small_support_patterns(rows):
    # Independent exhaustive oracle: enumerate feasible allocations for three sources.
    choices = [(0, 0), (1, 1)]  # Each used source/task must contain a whole outcome pair.
    for masks in itertools.product([(False,), (True,), (False, True)], repeat=3):
        records = []
        for i, labels in enumerate(masks):
            for label in labels:
                row = copy.deepcopy(
                    next(
                        r for r in rows if r["split"] == "train" and r["outcome"]["failed"] is label
                    )
                )
                row["source"].update(
                    dataset=f"source-{i}", data_kind="test_inputs", original_id=str(label)
                )
                row["record_id"] = f"{i}:{label}"
                records.append(row)
        feasible = any(
            sum(a for a, b in trial) == 2
            and sum(b for a, b in trial) == 2
            and all(
                (not a or False in labels) and (not b or True in labels)
                for (a, b), labels in zip(trial, masks, strict=False)
            )
            for trial in itertools.product(choices, repeat=3)
        )
        if feasible:
            batch, report = balanced_batch(records, 4)
            assert sum(r["outcome"]["failed"] for r in batch) == 2
            assert max(report["source_counts"].values()) <= 2
        else:
            with pytest.raises(DataError):
                balanced_batch(records, 4)
