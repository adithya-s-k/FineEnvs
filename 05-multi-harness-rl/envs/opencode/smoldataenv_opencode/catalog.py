"""The Task API serves names and instructions, never private verifier answers."""

import os
from functools import lru_cache

from .tasks import load_tasks


@lru_cache
def tasks(split):
    if split not in {"train", "test"}:
        raise ValueError("Choose train or test")
    return load_tasks(os.environ.get("SMOLDATA_DATA", "prepared"), split)


def task_by_name(split, name):
    return next(row for row in tasks(split) if row["name"] == name)


class TaskCatalog:
    def list_splits(self):
        return ["train", "test"]

    def num_tasks(self, split):
        return len(tasks(split))

    def get_task(self, split, index):
        return {
            key: value for key, value in tasks(split)[index].items() if key != "folder"
        }

    def get_task_range(self, split, start=None, stop=None):
        return [
            self.get_task(split, i)
            for i in range(*slice(start, stop).indices(self.num_tasks(split)))
        ]

    def list_tasks(self, split):
        return self.get_task_range(split)
