"""Build the fixed task splits in data/tasks/.

    python -m scott_wu_math.scripts.build_tasks   # or: python scripts/build_tasks.py from the env dir

test:  50 deals per level (200), seed 2026
train: 1,000 draws per level, seed 7, keeping only deals that differ from every test deal and every
       earlier train deal in what grading sees: the target and the card values (suits do not matter).
       Ids keep the draw number, so they have gaps where a draw was dropped. Level 24 keeps 639: only
       1,362 of the 1,820 four-card hands can make 24, so its draws repeat often.
video: the two hands from the video, 2025 from eight cards and 6,843 from nine

Test is kept exactly as first built, since the published eval card was computed on it. It has two
pairs of level-24 deals with the same values (test-24-0008 / test-24-0023, test-24-0011 / test-24-0013).
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from scott_wu_math.game.cards import Card  # noqa: E402
from scott_wu_math.game.expr import parse, steps, to_text  # noqa: E402
from scott_wu_math.game.tasks import LEVELS, make_task  # noqa: E402

OUT = HERE / "data" / "tasks"

# As played in the video (the suits are not all visible, so they are filled in).
VIDEO = [
    {
        "id": "video-2025",
        "target": 2025,
        "cards": [(13, "S"), (12, "H"), (11, "D"), (10, "C"), (7, "S"), (9, "H"), (5, "D"), (5, "C")],
        "solution": "((13 - 12) + (11 - 10) + 7) * 9 * 5 * 5",
    },
    {
        "id": "video-6843",
        "target": 6843,
        "cards": [(6, "S"), (4, "H"), (12, "D"), (10, "C"), (10, "S"), (8, "H"), (11, "D"), (7, "C"), (3, "S")],
        "solution": "(((12 - 10) * (6 + 4) * 10 + 8) * 11 - 7) * 3",
    },
]


def deal_key(task: dict) -> tuple[int, tuple[int, ...]]:
    """What grading depends on: the target and the card values, without suits."""
    return task["target"], tuple(sorted(c["rank"] for c in task["cards"]))


def build(seed: int, per_level: int, prefix: str, skip: set[str], avoid: set | None = None) -> list[dict]:
    """Draw per_level deals per level whose id (target and cards with suits) is not in `skip`.

    With `avoid`, a draw whose `deal_key` is already there is dropped, and each kept deal's key is added.
    """
    rng = random.Random(seed)
    tasks: list[dict] = []
    for level in LEVELS:
        made = 0
        while made < per_level:
            task = make_task(rng, level=level)
            key = task["id"]
            if key in skip:
                continue
            skip.add(key)
            number, made = made, made + 1
            if avoid is not None:
                if deal_key(task) in avoid:
                    continue
                avoid.add(deal_key(task))
            task["id"] = f"{prefix}-{level}-{number:04d}"
            task["key"] = key
            tasks.append(task)
    return tasks


def write(name: str, tasks: list[dict]) -> None:
    path = OUT / f"{name}.jsonl"
    path.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks))
    per_level = Counter(t["level"] for t in tasks)
    print(f"{path.relative_to(HERE)}: {len(tasks)} tasks ({', '.join(f'{k} {v}' for k, v in per_level.items())})")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    test = build(2026, 50, "test", seen)
    train = build(7, 1000, "train", seen, avoid={deal_key(t) for t in test})
    video = []
    for v in VIDEO:
        cards = [Card(r, s) for r, s in v["cards"]]
        tree = parse(v["solution"])
        video.append(
            {
                "id": v["id"],
                "level": "year" if len(cards) == 8 else "four-digit",
                "target": v["target"],
                "cards": [c.to_dict() for c in cards],
                "solution": to_text(tree),
                "solution_steps": steps(tree),
            }
        )
    write("test", test)
    write("train", train)
    write("video", video)


if __name__ == "__main__":
    main()
