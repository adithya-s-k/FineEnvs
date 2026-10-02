"""Stage public task files in a sandbox; run the held-out grader on the host."""

from pathlib import Path

import tomllib


def load_tasks(data, split):
    root = Path(data).resolve() / "datasets" / split / "tasks"
    tasks = []
    for folder in sorted(root.iterdir()):
        spec = tomllib.loads((folder / "task.toml").read_text())
        tasks.append(
            {
                "name": folder.name,
                "folder": str(folder),
                "instruction": (folder / "instruction.md").read_text(),
                "difficulty": spec["metadata"]["difficulty_tier"],
            }
        )
    if not tasks or len({t["instruction"] for t in tasks}) != len(tasks):
        raise ValueError("Tasks must have unique instructions; run prepare.py first")
    return tasks
