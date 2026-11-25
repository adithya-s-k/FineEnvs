"""Deterministic synthetic episodes for plumbing checks, not benchmark evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_ENV_ROOT = str(ROOT / "envs")
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)
from adapters.common import chat_to_atif, make_record  # noqa: E402


def build() -> list[dict]:
    rows = []
    for split in ("train", "dev", "test"):
        for source in ("fixture-files", "fixture-analysis"):
            for family_index in range(2):
                family = f"{source}:{split}:{family_index}"
                task = f"Write the value {family_index + 7} to result.txt and verify its contents."
                for variant in ("direct", "recovered", "false_completion"):
                    failed = variant == "false_completion"
                    messages = [{"role": "user", "content": task}]
                    if variant == "recovered":
                        messages += [
                            {
                                "role": "assistant",
                                "content": "Try the old path.",
                                "tool_calls": [
                                    {
                                        "id": "first",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": {"path": "old/result.txt"},
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": "first",
                                "content": "File not found",
                                "is_error": True,
                            },
                        ]
                    messages += [
                        {
                            "role": "assistant",
                            "content": "Write the requested value.",
                            "tool_calls": [
                                {
                                    "id": "write",
                                    "function": {
                                        "name": "write_file",
                                        "arguments": {
                                            "path": "result.txt",
                                            "value": family_index + (8 if failed else 7),
                                        },
                                    },
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "write", "content": "Written."},
                        {
                            "role": "assistant",
                            "content": "Check the file.",
                            "tool_calls": [
                                {
                                    "id": "read",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": {"path": "result.txt"},
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "read",
                            "content": str(family_index + (8 if failed else 7)),
                        },
                        {"role": "assistant", "content": "I've completed the task successfully."},
                    ]
                    trajectory = chat_to_atif(messages, agent_name="fixture-agent")
                    row = make_record(
                        source={
                            "dataset": source,
                            "revision": "fixture-v1",
                            "license": "MIT",
                            "original_id": f"{family}:{variant}",
                            "data_kind": "smoke_fixture",
                            "agent_model": "fixture-generator",
                            "user_model": "not_applicable",
                        },
                        task={
                            "instruction": task,
                            "policy": "The file must contain exactly the requested value.",
                            "success_criteria": None,
                        },
                        trajectory=trajectory,
                        family=family,
                        split=split,
                        failed=failed,
                        label_source="verified_outcome",
                        checker={"kind": "synthetic_expected_state", "not_a_real_run": True},
                        failures=[
                            {
                                "category": "false_completion",
                                "step_id": len(trajectory["steps"]),
                                "critical": True,
                                "source_label": "fixture",
                            }
                        ]
                        if failed
                        else [],
                    )
                    rows.append(row)
    unknown = json.loads(json.dumps(rows[0]))
    unknown["record_id"] = "unlabelled-fixture"
    unknown["source"]["original_id"] = "unlabelled-fixture"
    unknown["group"]["task_family_id"] = "unlabelled-fixture"
    unknown["outcome"] = {"failed": None, "label_source": "none", "checker": None, "reviewer": None}
    rows.append(unknown)
    return rows


if __name__ == "__main__":
    path = ROOT / "data" / "sample.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in build()),
        encoding="utf-8",
    )
    print(f"Wrote {len(build())} synthetic fixtures to {path}")
