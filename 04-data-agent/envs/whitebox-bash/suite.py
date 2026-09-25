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

"""A generated terminal-task suite, big enough to train on.

WHY GENERATED, AND WHAT THAT COSTS
The five hand-written tasks in `tasks.py` are enough to prove the loop runs and nothing more: four
GRPO steps over five tasks cannot show learning. This module produces a few hundred parameterised
tasks so a real run has something to move on.

Be clear about what these are: SYNTHETIC and TEMPLATED. A policy can improve here by learning the
shape of ten templates rather than by getting better at terminals, so a gain on this suite is
evidence that the training loop works, NOT evidence of general capability. SETA's 1,376 human-written
tasks are the real target; this exists so the plumbing can be exercised before that lands.

EVERY ANSWER IS COMPUTED IN PYTHON, NEVER BY RUNNING THE SETUP
The generator decides the data and derives the gold answer from the same values it writes into the
sandbox. Deriving it by executing the setup would make the grader agree with a buggy setup, and the
task would be unsolvable while scoring as if the model were wrong.

TRAIN AND TEST ARE DISJOINT BY CONSTRUCTION
Both draw from the same templates but from DIFFERENT random streams (the split name seeds the RNG),
so no test task appears in training. Sharing templates is intended -- it measures whether the policy
generalises across parameters -- but sharing an instance would be leakage.
"""

from __future__ import annotations

import random
from typing import Callable

from .tasks import Task


NAMES = ["ada", "alan", "grace", "edsger", "barbara", "linus", "ken", "brian",
         "donald", "john", "guido", "bjarne", "rob", "james", "anders"]
WORDS = ["ERROR", "WARN", "FATAL", "TIMEOUT", "RETRY"]
EXTS = ["log", "txt", "csv", "json", "cfg"]


def _csv(rng: random.Random, n: int) -> tuple[list[tuple[str, int]], str]:
    """A small CSV plus the shell that writes it. Rows carry DISTINCT scores.

    Distinct on purpose: a tie makes "which name has the highest score" ambiguous, and an ambiguous
    task punishes a correct answer. The generator must not create questions with two right answers.
    """
    names = rng.sample(NAMES, n)
    scores = rng.sample(range(10, 100), n)
    rows = list(zip(names, scores))
    body = "name,score\\n" + "\\n".join(f"{a},{b}" for a, b in rows) + "\\n"
    return rows, f"printf '{body}' > data.csv"


# Each template returns (instruction, answer, setup, check). `check` is "" where correctness is
# entirely captured by the answer.
def _t_count_rows(rng): 
    rows, setup = _csv(rng, rng.randint(3, 8))
    return ("There is a file `data.csv` in your working directory. How many rows does it have, "
            "excluding the header? Submit just the number.", str(len(rows)), setup, "")


def _t_max_name(rng):
    rows, setup = _csv(rng, rng.randint(3, 8))
    return ("A file `data.csv` has columns `name` and `score`. Which name has the highest score? "
            "Submit just the name.", max(rows, key=lambda r: r[1])[0], setup, "")


def _t_min_name(rng):
    rows, setup = _csv(rng, rng.randint(3, 8))
    return ("A file `data.csv` has columns `name` and `score`. Which name has the lowest score? "
            "Submit just the name.", min(rows, key=lambda r: r[1])[0], setup, "")


def _t_sum_col(rng):
    rows, setup = _csv(rng, rng.randint(3, 6))
    return ("A file `data.csv` has columns `name` and `score`. What is the sum of all scores? "
            "Submit just the number.", str(sum(s for _, s in rows)), setup, "")


def _t_grep_count(rng):
    word = rng.choice(WORDS)
    nfiles = rng.randint(2, 4)
    per = [rng.randint(0, 3) for _ in range(nfiles)]
    if sum(per) == 0:
        per[0] = 1
    cmds = ["mkdir -p logs"]
    for i, k in enumerate(per):
        lines = "\\n".join([f"{word} line {j}" for j in range(k)] + ["ok"])
        cmds.append(f"printf '{lines}\\n' > logs/f{i}.log")
    return (f"Several `.log` files are under `logs/`. How many lines across all of them contain the "
            f"word {word}? Submit just the number.", str(sum(per)), " && ".join(cmds), "")


def _t_count_files(rng):
    ext = rng.choice(EXTS)
    n = rng.randint(2, 6)
    other = rng.randint(1, 4)
    other_ext = rng.choice([e for e in EXTS if e != ext])
    cmds = ["mkdir -p tree/a tree/b"]
    for i in range(n):
        cmds.append(f"touch tree/{'a' if i % 2 else 'b'}/f{i}.{ext}")
    for i in range(other):
        cmds.append(f"touch tree/a/o{i}.{other_ext}")
    return (f"How many files with the extension `.{ext}` are there anywhere under `tree/`? "
            f"Submit just the number.", str(n), " && ".join(cmds), "")


def _t_largest_file(rng):
    names = rng.sample(NAMES, 4)
    sizes = rng.sample(range(20, 400), 4)
    cmds = ["mkdir -p blobs"]
    for nm, sz in zip(names, sizes):
        cmds.append(f"head -c {sz} /dev/zero | tr '\\\\0' 'x' > blobs/{nm}.dat")
    biggest = names[sizes.index(max(sizes))]
    return ("Which file under `blobs/` is the largest? Submit just its filename, without the "
            "directory.", f"{biggest}.dat", " && ".join(cmds), "")


def _t_hidden_value(rng):
    key = rng.choice(["port", "retries", "timeout", "workers"])
    val = rng.randint(2, 9999)
    depth = rng.randint(1, 3)
    path = "/".join(f"cfg{i}" for i in range(depth)) + "/app.conf"
    return (f"Somewhere under the current directory there is a config file containing a line like "
            f"`{key} = <value>`. Find it and submit just the value.", str(val),
            f"mkdir -p {path.rsplit('/', 1)[0]} && printf 'name = svc\\n{key} = {val}\\nmode = fast\\n' > {path}",
            "")


def _t_fib(rng):
    n = rng.randint(10, 60)
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return (f"Write a file `solution.py` defining a function `fib(n)` returning the n-th Fibonacci "
            f"number, with fib(0)=0 and fib(1)=1. Then submit the value of fib({n}).", str(a), "",
            f"python3 -c \"import solution; assert solution.fib({n})=={a}\"")


def _t_fix_syntax(rng):
    token = rng.choice(["OK", "DONE", "READY", "FINE", "PASS", "GOOD", "SET", "LIVE"])
    fn = rng.choice(["main", "run", "start", "go", "entry"])
    return (f"The file `broken.py` has a syntax error. Fix it in place so that `python3 broken.py` "
            f"prints {token}, then submit the word {token}.", token,
            f"printf 'def {fn}()\\n    print(\"{token}\")\\n\\n{fn}()\\n' > broken.py",
            f"python3 broken.py | grep -q {token}")


def _t_unique_count(rng):
    n_unique = rng.randint(2, 5)
    cats = rng.sample(["red", "blue", "green", "amber", "violet"], n_unique)
    rows = [rng.choice(cats) for _ in range(rng.randint(6, 12))]
    # Guarantee every category appears, or the stated answer is wrong.
    rows[:n_unique] = cats
    body = "colour\\n" + "\\n".join(rows) + "\\n"
    return ("A file `items.csv` has a single column `colour`. How many DISTINCT colours appear? "
            "Submit just the number.", str(len(set(rows))), f"printf '{body}' > items.csv", "")


TEMPLATES: tuple[Callable[[random.Random], tuple[str, str, str, str]], ...] = (
    _t_count_rows, _t_max_name, _t_min_name, _t_sum_col, _t_grep_count,
    _t_count_files, _t_largest_file, _t_hidden_value, _t_fib, _t_fix_syntax,
    _t_unique_count,
)

# Difficulty is a property of the TEMPLATE, not of a draw, so it is stable across seeds.
_DIFFICULTY = {
    _t_count_rows: "easy", _t_max_name: "easy", _t_min_name: "easy",
    _t_sum_col: "easy", _t_unique_count: "medium", _t_grep_count: "medium",
    _t_count_files: "medium", _t_largest_file: "medium", _t_hidden_value: "medium",
    _t_fib: "hard", _t_fix_syntax: "hard",
}


def _signature(instruction: str, answer: str, setup: str, check: str) -> tuple[str, str, str, str]:
    """What makes two tasks THE SAME task.

    Not the instruction alone: several templates ask a constant question ("how many rows does
    data.csv have") and vary only the data, which is deliberate -- it forces the agent to look
    instead of guessing. Identity is the whole instance.
    """
    return (instruction, answer, setup, check)


def generate(split: str, n: int, exclude: frozenset = frozenset()) -> tuple[Task, ...]:
    """Build `n` DISTINCT tasks for `split`, none of them in `exclude`.

    Rejection-sampled rather than merely seeded differently. Seeding alone makes collisions unlikely,
    not impossible, and the parameter ranges here are small: measured on a first attempt, 4 of 30
    test tasks were byte-identical to training tasks and 7 train tasks were internal duplicates. Test
    items the policy trained on inflate an eval and nothing downstream would report it.

    The split NAME seeds the RNG, so a given split is identical on every call and in every process --
    which it must be, because the index is the task's identity everywhere downstream.

    Args:
        exclude (`frozenset`, *optional*):
            Signatures (see `_signature`) that must not appear. Pass the training set when building
            an eval split.

    Raises:
        `RuntimeError`: If `n` distinct tasks could not be drawn. Better to fail loudly than to
            return a short split that silently changes what "index 87" means.
    """
    rng = random.Random(f"whitebox-bash/{split}")
    seen: set = set(exclude)
    out: list[Task] = []
    attempts = 0
    max_attempts = 200 * max(n, 1)
    while len(out) < n and attempts < max_attempts:
        template = TEMPLATES[len(out) % len(TEMPLATES)]
        instruction, answer, setup, check = template(rng)
        attempts += 1
        sig = _signature(instruction, answer, setup, check)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(Task(instruction=instruction, answer=answer,
                        difficulty=_DIFFICULTY[template], setup=setup, check=check,
                        metadata={"template": template.__name__, "n": len(out)}))
    if len(out) < n:
        raise RuntimeError(
            f"could only draw {len(out)} distinct tasks of {n} for split {split!r} after "
            f"{attempts} attempts; widen the template parameter ranges"
        )
    return tuple(out)


def signatures(tasks_: tuple[Task, ...]) -> frozenset:
    """Signatures of a task tuple, for passing to `generate(exclude=...)`."""
    return frozenset(_signature(t.instruction, t.answer, t.setup, t.check) for t in tasks_)
