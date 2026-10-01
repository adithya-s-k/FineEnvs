"""Stage public task files in a sandbox; run the held-out grader on the host."""

import json
import os
import subprocess
import sys
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


def stage_task(sandbox, folder):
    folder = Path(folder)
    spec = tomllib.loads((folder / "task.toml").read_text())
    # Only data-download credentials enter the sandbox, never verifier answers.
    env = {
        key: os.path.expandvars(value)
        for key, value in spec["environment"]["env"].items()
    }
    if any("${" in value for value in env.values()):
        raise ValueError("Set HF_TOKEN before staging task data")
    sandbox.write_text(
        "/opt/pull_bucket.py", (folder / "environment/pull_bucket.py").read_text()
    )
    result = sandbox.exec(
        "mkdir -p /workdir /home/user/input && python3 /opt/pull_bucket.py "
        '&& test -n "$(ls -A /home/user/input)"',
        envs=env,
        timeout=300,
    )
    if result.exit_code:
        raise RuntimeError("Task data download failed")


def grade_answer(sandbox, folder):
    folder = Path(folder)
    spec = tomllib.loads((folder / "task.toml").read_text())
    answer = (
        sandbox.read_text("/workdir/answer.txt")
        if sandbox.exists("/workdir/answer.txt")
        else ""
    )
    result = subprocess.run(
        [sys.executable, str(folder / "tests/grader.py"), "--json"],
        input=answer,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
        env={**os.environ, **spec["verifier"]["env"]},
    )
    if result.returncode:
        raise RuntimeError("Task grader failed")
    value = json.loads(result.stdout)["correctness"]
    if value not in (0, 1):
        raise ValueError("Expected a binary correctness grade")
    return float(value)
