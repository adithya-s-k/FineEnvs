"""Keep native correctness and verified tool count separate from shaping."""
import os
from pathlib import Path

from recipe import reward, write_json


def grade(session):
    verdict = session.sandbox.grade(session.submitted)
    correctness = verdict.get("reward")
    calls = sum(c["tool"] != "submit_solution" for c in session.calls)
    weight = float(os.environ["EFFICIENCY_WEIGHT"]) if session.split == "train" else 0.0
    value = reward(correctness, calls, verified=True, weight=weight,
                   budget=float(os.environ["TOOL_BUDGET"]))
    record = {"correctness": correctness, "tool_calls": calls, "reward": value,
              "task_index": session.index, "split": session.split, "tool_count_verified": True}
    write_json(Path(os.environ["WHITEBOX_RECORDS"]) / f"{session.session_id}.json", record)
    return {**verdict, "reward": value, "n_tool_calls": calls}
