"""Route each GRPO group to a harness, without changing TRL.

WHY THIS SHAPE. `HarnessRolloutWorker` hands the factory only `(prompt, seed, episode_id)` --
`async_rollout_worker.py` pulls `(group_id, row)` but calls `_generate_one(prompt, ..., group_id)`
and drops the row. So a factory cannot read a harness off a dataset column. What it CAN read is
`seed`, which `_run_session` sets to `group_id`, and `_repeat_iterator` yields the SAME group_id for
all `num_generations` of a group.

That is the load-bearing property: **harness is constant within a group.** Measured pass@4 across
harnesses on this suite spans 0.320 to 0.020, so a group whose members ran under different harnesses
would have a baseline averaging two competence levels, and the advantage would encode WHICH HARNESS
rather than which action. Constant-within-group makes the spread a BETWEEN-group constant, which
advantage normalisation removes entirely.

THE DEGENERACY TO AVOID. Group -> row is `group_id % len(dataset)`. If `gcd(len(dataset), H) > 1`,
`harnesses[group_id % H]` pairs every task with only one harness -- seed routing silently collapses
into a disjoint partition and you are not running a multi-harness experiment at all. `pair_rows`
below pads the task list to make them coprime and asserts it.
"""

from __future__ import annotations

import logging
from math import gcd
from typing import Any

from harbor_env.harness import HarborSession, HarborSessionFactory

logger = logging.getLogger(__name__)


class MultiHarborSessionFactory(HarborSessionFactory):
    """A HarborSessionFactory whose harness is chosen per GROUP, from `seed`."""

    def __init__(self, *args: Any, harnesses: list[str], **kw: Any) -> None:
        super().__init__(*args, **kw)
        if not harnesses:
            raise ValueError("harnesses must be non-empty")
        self.harnesses = list(harnesses)
        # group_id -> harness, so a violation is detectable rather than merely unlikely.
        self._group_harness: dict[int, str] = {}

    def __getstate__(self) -> dict[str, Any]:
        # The factory is pickled into a spawned child; the parent's live client must not go with it.
        state = super().__getstate__()
        state["harnesses"] = self.harnesses
        state["_group_harness"] = {}
        return state

    def harness_for(self, seed: int | None) -> str:
        return self.harnesses[(seed or 0) % len(self.harnesses)]

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> HarborSession:
        harness = self.harness_for(seed)

        # HARD failure, not a warning. If two generations of one group ran under different harnesses
        # the group's central claim is void, and a warning in a log nobody reads is how that ships.
        previous = self._group_harness.setdefault(int(seed or 0), harness)
        if previous != harness:
            raise RuntimeError(
                f"group {seed} mixed harnesses ({previous} then {harness}). The GRPO baseline would "
                f"average two competence levels (measured pass@4 spread 0.320-0.020 on this suite), "
                f"so the advantage would encode which harness, not which action."
            )

        instruction = _instruction_of(task)
        self.tasks()  # builds the instruction -> index map
        index = self._by_instruction.get(_instruction_id(instruction))
        if index is None:
            # Verbatim from HarborSessionFactory: a lookup failure must never silently run task 0.
            raise KeyError(
                "this prompt does not match any task on the server. Build the dataset from "
                "`prompt_rows()` so the instruction the trainer sends is the one the server has."
            )
        return HarborSession(
            env=self.new_client(),  # one client PER SESSION: a shared MCP socket raises
            owns_env=True,          # ConcurrencyError on concurrent recv, making every rollout unscorable
            split=self._split,
            task_index=index,
            instruction=instruction,
            harness=harness,        # <-- the only thing that varies
            sandbox=self.sandbox,
            llm_url=self.llm_url,
            model=self.model,
            reward_key=self.reward_key,
            api_key=self.api_key,
            auth_header=self.auth_header,
            agent_timeout_sec=self.agent_timeout_sec,
            agent_step_limit=self.agent_step_limit,
        )


def _instruction_of(task: Any) -> str:
    from harbor_env.harness import _instruction_of as f  # reuse, never reimplement
    return f(task)


def _instruction_id(text: str) -> str:
    from harbor_env.harness import instruction_id as f
    return f(text)


def pair_rows(factory: MultiHarborSessionFactory) -> list[dict[str, Any]]:
    """Dataset rows, padded so `gcd(len(rows), n_harnesses) == 1`.

    Without coprimality, `group_id % len(dataset)` and `group_id % H` stay in lockstep and each task
    only ever meets one harness -- the run looks multi-harness and is not.
    """
    rows = list(factory.prompt_rows())
    h = len(factory.harnesses)
    if h > 1:
        while len(rows) > 1 and gcd(len(rows), h) != 1:
            rows.append(dict(rows[len(rows) % len(rows)]))  # duplicate one row to break the common factor
        assert gcd(len(rows), h) == 1, f"gcd({len(rows)},{h}) != 1"
    logger.info("pair_rows: %d rows x %d harnesses, gcd=%d", len(rows), h, gcd(len(rows), h))
    return rows
