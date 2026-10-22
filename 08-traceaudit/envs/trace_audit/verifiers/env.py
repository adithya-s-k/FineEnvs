"""Single-turn auditor port."""

from __future__ import annotations

import copy
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
_ENV_ROOT = str(HERE.parents[1])
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)
from trace_audit.core import (  # noqa: E402
    SYSTEM_PROMPT,
    TraceAuditEnv,
    load_records,
    parse_action,
    score_probability,
)
from trace_audit.core.shortcuts import training_shortcut_audit  # noqa: E402

TOOL_FUNCTIONS = []


def _training_records(core):
    # Pair only after rendering/budget admission, so dropping an oversized trace
    # cannot silently reintroduce a source/task label imbalance.
    batch, report = core.training_sample()
    report = copy.deepcopy(report)
    report["shortcut_audit"] = training_shortcut_audit(
        batch, render=lambda row: core.prompt(row["record_id"])
    )
    return batch, report


class TraceAuditToolkit:
    """Manual/TRL submission facade."""

    def __init__(
        self,
        records=None,
        *,
        split="train",
        allow_fixtures=False,
        token_counter=None,
        training_size=None,
        source_cap=0.5,
        sampling_seed=42,
        max_record_reuse=1,
    ):
        self._env = TraceAuditEnv(
            records if records is not None else load_records(ROOT / "data/sample.jsonl"),
            split=split,
            allow_fixtures=allow_fixtures,
            token_counter=token_counter,
            training_size=training_size,
            source_cap=source_cap,
            sampling_seed=sampling_seed,
            max_record_reuse=max_record_reuse,
            validated=records is None,
        )
        self._sampling_report = None
        if split == "train":
            _, report = self._env.training_sample()
            self._sampling_report = copy.deepcopy(report)

    def reset(self, *, record_id=None, seed=None) -> str:
        return self._env.reset(record_id=record_id, seed=seed)

    def submit_analysis(self, response: str) -> str:
        """Submit the final JSON analysis once. Ends the episode."""
        self._env.step(response)
        return "Analysis submitted. Episode complete."


def consume_toolkit_reward(toolkit: TraceAuditToolkit) -> float:
    """Consume a submission reward once."""
    return toolkit._env.consume_reward()


def create_verifiers_env(
    data_path=None,
    *,
    allow_fixtures=False,
    eval_split="dev",
    max_tokens=32000,
    token_counter=None,
    training_size=None,
    source_cap=0.5,
    sampling_seed=42,
    max_record_reuse=1,
    return_report=False,
):
    """Build legacy Verifiers 0.1.9 SingleTurnEnv; imports optional dependencies lazily."""
    from datasets import Dataset

    import verifiers as vf

    records = load_records(data_path or ROOT / "data/sample.jsonl")
    private = {}
    sampling_report = None

    def dataset(split):
        nonlocal sampling_report
        core = TraceAuditEnv(
            records,
            split=split,
            allow_fixtures=allow_fixtures,
            max_tokens=max_tokens,
            token_counter=token_counter,
            training_size=training_size,
            source_cap=source_cap,
            sampling_seed=sampling_seed,
            max_record_reuse=max_record_reuse,
            validated=True,
        )
        selected = list(core.records)
        if split == "train":
            selected, sampling_report = _training_records(core)
        rows = []
        for record in selected:
            key = record["record_id"]
            opaque = uuid.uuid4().hex
            private[opaque] = record
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": core.prompt(key)},
                    ],
                    "answer": opaque,
                }
            )
        return Dataset.from_list(rows)

    async def brier_reward(completion, answer, **kwargs):
        state = kwargs.get("state")
        if state is not None:
            if state.get("_traceaudit_rewarded"):
                return 0.0
            state["_traceaudit_rewarded"] = True
        record = private.get(answer)
        if record is None:
            return 0.0
        response = _completion_response(completion)
        action = parse_action(response, {s["step_id"] for s in record["trajectory"]["steps"]})
        return score_probability(action, record["outcome"]["failed"])

    if eval_split not in {"dev", "test"}:
        raise ValueError("Evaluation must use the original dev or test distribution")
    env = vf.SingleTurnEnv(
        dataset=dataset("train"),
        eval_dataset=dataset(eval_split),
        rubric=vf.Rubric(funcs=[brier_reward], weights=[1.0]),
        message_type="chat",
    )
    return (env, sampling_report) if return_report else env


load_environment = create_verifiers_env


def _completion_response(completion):
    if isinstance(completion, list):
        assistants = [m for m in completion if m.get("role") == "assistant"]
        # SingleTurnEnv must produce one final assistant message, no extra turn or tools.
        response = (
            assistants[0].get("content")
            if len(assistants) == 1 and not assistants[0].get("tool_calls")
            else None
        )
    else:
        response = completion if isinstance(completion, str) else None
    return response
