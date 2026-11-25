from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import dataclass

from .records import DataError, scoring_exclusion_reason, validate_collection
from .render import SYSTEM_PROMPT, render_record
from .scoring import parse_action, score_probability


@dataclass(frozen=True)
class Transition:
    reward: float
    done: bool
    info: dict


class TraceAuditEnv:
    def __init__(
        self,
        records: list[dict],
        *,
        split: str = "train",
        max_tokens: int = 32000,
        token_counter=None,
        allow_fixtures: bool = False,
        training_size=None,
        source_cap=0.5,
        sampling_seed=42,
        max_record_reuse=1,
        validated=False,
    ):
        if split not in {"train", "dev", "test"} or type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Invalid split or rendering budget")
        # UTF-8 bytes provide a conservative upper bound for byte-level tokenizers; callers may
        # supply the target model's tokenizer.
        count = token_counter or (lambda text: len(text.encode("utf-8")))
        self._records, self._prompts = {}, {}
        self._exclusions = Counter()
        if not validated:
            validate_collection(records)
        # Keep revision audit evidence even when its traces are in dev or exceed the budget.
        self._audit_records = copy.deepcopy(
            [
                row
                for row in records
                if row["source"]["dataset"].casefold()
                == "nebius/swe-rebench-openhands-trajectories"
                and row.get("review_history")
            ]
        )
        for raw in records:
            if raw["split"] != split:
                continue
            reason = scoring_exclusion_reason(raw, allow_fixtures=allow_fixtures)
            try:
                prompt = render_record(raw)
            except ValueError:
                reason, prompt = "unsupported_multimodal", ""
            if reason is None and count(SYSTEM_PROMPT + "\n" + prompt) > max_tokens:
                reason = "over_render_budget"
            if reason:
                self._exclusions[(raw["source"]["dataset"], reason)] += 1
                continue
            self._records[raw["record_id"]] = copy.deepcopy(raw)
            self._prompts[raw["record_id"]] = prompt
        if not self._records:
            raise DataError("No labelled, supported episodes fit this split and rendering budget")
        self._finished = True
        self._record = None
        self._last_transition = None
        self._split = split
        self._allow_fixtures = allow_fixtures
        self._training_ids = None
        self._training_config = {
            "size": training_size,
            "source_cap": source_cap,
            "seed": sampling_seed,
            "max_record_reuse": max_record_reuse,
        }

    def training_sample(self):
        if self._training_ids is None:
            from .sampling import balanced_batch

            self._training_batch, self._training_sampling_report = balanced_batch(
                list(self._records.values()),
                allow_fixtures=self._allow_fixtures,
                validated=True,
                audit_records=self._audit_records,
                **self._training_config,
            )
            self._training_ids = [row["record_id"] for row in self._training_batch]
        return self._training_batch, self._training_sampling_report

    @property
    def records(self):
        return tuple(self._records.values())

    @property
    def exclusions(self):
        return dict(self._exclusions)

    def prompt(self, record_id):
        return self._prompts[record_id]

    def consume_reward(self):
        """Consume the current submission reward once, closing an unsubmitted episode."""
        if self._record is None:
            return 0.0
        if self._last_transition is None and not self._finished:
            self.step(None)
        transition = self._last_transition
        self._last_transition = None
        return transition.reward if transition is not None else 0.0

    def reset(self, *, record_id: str | None = None, seed: int | None = None) -> str:
        if self._split == "train":
            self.training_sample()
            if record_id is None:
                record_id = random.Random(seed).choice(self._training_ids)
            elif record_id not in self._training_ids:
                raise DataError("Record is not in the paired training sample")
        record_id = record_id or random.Random(seed).choice(sorted(self._records))
        if record_id not in self._records:
            raise DataError("Episode unavailable in the selected split")
        self._record = self._records[record_id]
        self._finished = False
        self._last_transition = None
        return self._prompts[record_id]

    def step(self, action: str | dict | None) -> Transition:
        if self._record is None:
            raise RuntimeError("Call reset before submission")
        if self._finished:
            self._last_transition = Transition(0.0, True, {"already_submitted": True})
            return self._last_transition
        self._finished = True
        ids = {step["step_id"] for step in self._record["trajectory"]["steps"]}
        assessment = parse_action(action, ids)
        result = Transition(
            score_probability(assessment, self._record["outcome"]["failed"]),
            True,
            {
                "format_valid": assessment.p_failed is not None,
                "optional_warnings": list(assessment.warnings),
            },
        )
        self._last_transition = result
        return result
