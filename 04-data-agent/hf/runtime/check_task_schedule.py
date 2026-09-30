"""Check real task identities across both native catalogs and frozen schedules."""
import json
from common import RUN, configure


def check():
    configure()
    from daytona_whitebox_backend import load_frozen_tasks
    from openenv.harbor.tasks import resolve_task_dirs
    counts = {}
    for split, expected in [("train", 1000), ("test", 250)]:
        blackbox = resolve_task_dirs(str(RUN / "datasets" / split))
        whitebox = load_frozen_tasks(split)
        assert len(blackbox) == len(whitebox) == expected
        for index, (directory, task) in enumerate(zip(blackbox, whitebox)):
            assert task.metadata["task_index"] == index
            assert directory.name == task.metadata["source_name"], (split, index)
            assert task.instruction == (directory / "instruction.md").read_text()
        counts[split] = expected
        if split == "train":
            for filename in ["reference_schedule.json", "opencode_schedule.json"]:
                schedule = json.loads((RUN / filename).read_text())
                for row in schedule["tasks"]:
                    index = row["task_index"]
                    assert blackbox[index].name == row["name"]
                    assert whitebox[index].difficulty == row["difficulty"]
                for group in schedule["groups"]:
                    assert blackbox[group["task_index"]].name == group["task_name"]
    return {"passed": True, "catalog_tasks": counts, "both_schedules_match": True}


if __name__ == "__main__":
    print(json.dumps(check()), flush=True)
