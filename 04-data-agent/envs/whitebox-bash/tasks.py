# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tasks, and the Task API over them.

CACHED AT MODULE LEVEL, NOT ON THE ENVIRONMENT
OpenEnv's HTTP server builds a throwaway environment instance per request and closes it in a
`finally`, so anything held on `self` dies with the request. A task cache on the instance would be
rebuilt for every `list_tasks` call; the cache therefore lives here, at module scope.

SPLITS ARE NAMED, NOT FILTERED
`get_task("train", 12)` must mean the same task on every call and in every process, because the index
IS the task's identity everywhere downstream -- in the dataset row the trainer holds, in the eval's
common item set, in a bug report. A difficulty *filter* applied on top of one list would shift every
index after it, which is why difficulty is baked into the split name instead.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any


# The built-in suite. Deliberately small, self-contained and checkable without network or a dataset
# download, so the environment can be smoke-tested the moment it is installed. `setup` runs before
# the agent sees the task, which is what lets a task ship its own input files.
@dataclass(frozen=True)
class Task:
    """One task.

    Attributes:
        instruction (`str`):
            What the agent is asked to do. This is the text `reset()` returns to the trainer.
        answer (`str`):
            Gold answer, compared against `submit`. Never sent to the client.
        difficulty (`str`):
            `"easy"`, `"medium"` or `"hard"`. Part of the split name, not a filter.
        setup (`str`, *optional*):
            Shell run in the sandbox before the agent starts, to stage inputs.
        check (`str`, *optional*):
            Shell whose exit code is an extra correctness signal, for tasks whose result is a
            side effect on the filesystem rather than a string.
    """

    instruction: str
    answer: str
    difficulty: str = "easy"
    setup: str = ""
    check: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        """Stable identity from the instruction, so an index change is detectable."""
        return hashlib.sha1(self.instruction.encode()).hexdigest()[:12]

    def public(self) -> dict[str, Any]:
        """The projection a client may see. The gold answer is withheld BY CONSTRUCTION.

        Returning the answer to the client would make every rollout trivially solvable and the reward
        meaningless, and nothing downstream would report it -- the scores would simply be perfect.
        """
        d = asdict(self)
        d.pop("answer", None)
        d.pop("check", None)
        d.pop("setup", None)
        d.pop("metadata", None)
        d["task_id"] = self.task_id
        return d


_BUILTIN: tuple[Task, ...] = (
    Task(
        instruction=(
            "There is a file `data.csv` in your working directory. How many rows does it have, "
            "excluding the header? Submit just the number."
        ),
        answer="4",
        difficulty="easy",
        setup=(
            "printf 'name,score\\nada,90\\nalan,85\\ngrace,95\\nedsger,88\\n' > data.csv"
        ),
    ),
    Task(
        instruction=(
            "A file `data.csv` has columns `name` and `score`. Which name has the highest score? "
            "Submit just the name."
        ),
        answer="grace",
        difficulty="easy",
        setup=(
            "printf 'name,score\\nada,90\\nalan,85\\ngrace,95\\nedsger,88\\n' > data.csv"
        ),
    ),
    Task(
        instruction=(
            "Several `.log` files are scattered under `logs/`. Count how many lines across all of "
            "them contain the word ERROR. Submit just the number."
        ),
        answer="3",
        difficulty="medium",
        setup=(
            "mkdir -p logs/a logs/b && "
            "printf 'ok\\nERROR disk\\nok\\n' > logs/a/one.log && "
            "printf 'ERROR net\\nfine\\n' > logs/b/two.log && "
            "printf 'ERROR cpu\\n' > logs/b/three.log"
        ),
    ),
    Task(
        instruction=(
            "Write a file `solution.py` that defines a function `fib(n)` returning the n-th "
            "Fibonacci number with fib(0)=0 and fib(1)=1. Then submit the value of fib(20)."
        ),
        answer="6765",
        difficulty="medium",
        check="python3 -c \"import solution; assert solution.fib(20)==6765\"",
    ),
    Task(
        instruction=(
            "The file `broken.py` has a syntax error. Fix it in place so that `python3 broken.py` "
            "prints OK, then submit the word OK."
        ),
        answer="OK",
        difficulty="hard",
        setup="printf 'def main()\\n    print(\"OK\")\\n\\nmain()\\n' > broken.py",
        check="python3 broken.py | grep -q OK",
    ),
)

# How many generated tasks each split holds. `train` and `test` are disjoint by construction -- the
# split name seeds the generator; see `suite.generate`.
N_TRAIN = int(os.environ.get("WHITE_BOX_BASH_N_TRAIN", "100"))
N_TEST = int(os.environ.get("WHITE_BOX_BASH_N_TEST", "30"))


def _generated(split: str, n: int) -> tuple[Task, ...]:
    from .suite import generate

    return generate(split, n)


# split name -> tasks. Difficulty is IN the name; see the module docstring.
#
# `demo` is the five hand-written tasks: enough to prove the loop runs, never enough to train on.
# `train`/`test` are generated and are what a real run uses.
def _splits() -> dict[str, tuple[Task, ...]]:
    from .suite import generate, signatures

    train = generate("train", N_TRAIN)
    # Test EXCLUDES every training instance. Seeding the two differently is not enough -- the
    # template parameter ranges are small enough to collide, and a test item the policy trained on
    # inflates the eval with nothing downstream to report it.
    test = generate("test", N_TEST, exclude=signatures(train))
    return {
        "demo": _BUILTIN,
        "train": train,
        "train:easy": tuple(t for t in train if t.difficulty == "easy"),
        "train:medium": tuple(t for t in train if t.difficulty == "medium"),
        "train:hard": tuple(t for t in train if t.difficulty == "hard"),
        "test": test,
    }


_SPLITS: dict[str, tuple[Task, ...]] = _splits()

_CACHE: dict[str, tuple[Task, ...]] = {}


# Where tasks come from. `data-agent` is the default and is the point of this environment living
# beside the two black-box ones: SAME tasks, SAME staging, SAME grader, so the only difference left
# between white box and black box is who owns the agent loop. `synthetic` is the generated suite,
# useful only for exercising the plumbing without network or HF credentials.
TASK_SOURCE = os.environ.get("WHITE_BOX_BASH_TASK_SOURCE", "data-agent")


def _load(split: str) -> tuple[Task, ...]:
    """Resolve a split, preferring a Hub dataset when one is configured.

    `WHITE_BOX_BASH_DATASET` swaps the built-in suite for a Hub dataset without touching this file.
    The built-ins stay as the fallback precisely so a fresh install is testable with no network.
    """
    if split in _CACHE:
        return _CACHE[split]
    if TASK_SOURCE == 'harbor-frozen':
        from daytona_whitebox_backend import load_frozen_tasks
        tasks = load_frozen_tasks(split)
        _CACHE[split] = tasks
        return tasks
    if TASK_SOURCE == "data-agent" and split not in ("demo",):
        from .dataagent import load as _load_data_agent

        tasks = _load_data_agent(split, limit=int(os.environ.get("WHITE_BOX_BASH_LIMIT", "0")))
        _CACHE[split] = tasks
        return tasks
    repo = os.environ.get("WHITE_BOX_BASH_DATASET", "").strip()
    if repo:
        from datasets import load_dataset

        base, _, tier = split.partition(":")
        rows = load_dataset(repo, split=base)
        tasks = tuple(
            Task(
                instruction=r["instruction"],
                answer=str(r.get("answer", "")),
                difficulty=str(r.get("difficulty", "easy")),
                setup=str(r.get("setup", "")),
                check=str(r.get("check", "")),
                metadata={k: r[k] for k in r.keys() if k not in
                          {"instruction", "answer", "difficulty", "setup", "check"}},
            )
            for r in rows
        )
        if tier:
            tasks = tuple(t for t in tasks if t.difficulty == tier)
    else:
        if split not in _SPLITS:
            raise KeyError(f"unknown split {split!r}; known: {sorted(_SPLITS)}")
        tasks = _SPLITS[split]
    _CACHE[split] = tasks
    return tasks


# --- the TaskProvider surface, declared structurally on the environment ------------------------
# Data-agent split names. Tier is part of the NAME, never a filter -- a filter shifts every index
# after it and the index is the task's identity everywhere downstream.
_DATA_AGENT_SPLITS = ("train", "train:easy", "train:medium", "train:hard", "test", "eval")


def list_splits() -> list[dict[str, Any]]:
    names = ('train', 'test') if TASK_SOURCE == 'harbor-frozen' else (_DATA_AGENT_SPLITS if TASK_SOURCE == 'data-agent' else tuple(sorted(_SPLITS)))
    return [{"name": n, "type": "train" if n.startswith("train") else "test"} for n in names]


def num_tasks(split: str) -> int:
    return len(_load(split))


def list_tasks(split: str, limit: int = 200) -> list[dict[str, Any]]:
    """A BOUNDED preview. `num_tasks` still reports the true total.

    Unbounded, this parses every task on every call, and the Task API calls it on a throwaway
    instance -- so an agent browsing splits would rebuild the whole suite each time.
    """
    return [t.public() for t in _load(split)[:limit]]


def get_task(split: str, index: int) -> dict[str, Any]:
    tasks = _load(split)
    if not 0 <= index < len(tasks):
        raise IndexError(f"index {index} out of range for split {split!r} ({len(tasks)} tasks)")
    return tasks[index].public()


def task_at(split: str, index: int) -> Task:
    """The FULL task, gold answer included. Server-side only -- never routed to a client."""
    tasks = _load(split)
    if not 0 <= index < len(tasks):
        raise IndexError(f"index {index} out of range for split {split!r} ({len(tasks)} tasks)")
    return tasks[index]


def get_task_range(split: str, start: int | None = None, stop: int | None = None) -> list[dict[str, Any]]:
    return [t.public() for t in _load(split)[slice(start, stop)]]
