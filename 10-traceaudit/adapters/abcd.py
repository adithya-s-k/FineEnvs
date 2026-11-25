"""Use ABCD as unlabelled reviewer evidence, never as a customer-service control environment."""

from __future__ import annotations

from trace_audit.core.records import training_split

from .common import DataError, make_record


def convert(row: dict, *, revision: str, split: str) -> dict:
    steps = []
    for index, turn in enumerate(row["original"]):
        speaker, text = turn
        if speaker not in {"agent", "customer", "action"} or not isinstance(text, str):
            raise DataError("Unexpected ABCD turn")
        step = {
            "step_id": len(steps) + 1,
            "source": {"agent": "agent", "customer": "user", "action": "system"}[speaker],
            "message": text if speaker != "action" else "",
            "extra": {"original_turn_index": index},
        }
        if speaker == "action":
            step["observation"] = {"results": [{"content": text}]}
        steps.append(step)
    instruction = next((text for speaker, text in row["original"] if speaker == "customer"), "")
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "ABCD human operator", "version": "source"},
        "steps": steps,
    }
    return make_record(
        source={
            "dataset": "asappresearch/abcd",
            "revision": revision,
            "license": "MIT",
            "original_id": str(row["convo_id"]),
            "data_kind": "human_dialogue",
            "source_split": split,
            "source_scenario": row.get("scenario"),
        },
        task={"instruction": instruction, "policy": None, "success_criteria": None},
        trajectory=trajectory,
        family=f"abcd:{row['convo_id']}",
        split=training_split(f"abcd:{row['convo_id']}"),
    )
