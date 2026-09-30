"""Per-rollout budgets and shared admission for the two environment Spaces."""
from contextlib import asynccontextmanager
import asyncio
import os
from pathlib import Path
import threading
import time


def workload(dataset):
    return "train" if Path(str(dataset).rstrip("/")).name == "train" else "eval"


def output_limit(dataset):
    role = workload(dataset)
    return int(os.environ.get("OPENENV_" + role.upper() + "_OUTPUT_TOKENS",
                              "16384" if role == "train" else "4096"))


class Admission:
    """Bound active sandboxes, keeping slots that eval cannot consume for training.

    Synchronous native whitebox tools and async Harbor rollouts use the same accounting.
    Waiting async callers never occupy the shared thread pool.
    """
    def __init__(self, total, train_reserve):
        if not 0 < train_reserve < total:
            raise ValueError("Require 0 < training reservation < sandbox capacity")
        self.total, self.train_reserve = total, train_reserve
        self.active = {"train": 0, "eval": 0}
        self.waiting = {"train": 0, "eval": 0}
        self.condition = threading.Condition()

    def _available(self, role):
        return (sum(self.active.values()) < self.total and
                (role == "train" or self.active["eval"] < self.total - self.train_reserve))

    def acquire(self, role, timeout=900):
        with self.condition:
            self.waiting[role] += 1
            try:
                if not self.condition.wait_for(lambda: self._available(role), timeout):
                    raise TimeoutError("Shared sandbox capacity wait expired")
                self.active[role] += 1
            finally:
                self.waiting[role] -= 1

    def release(self, role):
        with self.condition:
            if self.active[role] <= 0:
                raise RuntimeError("Sandbox reservation released twice")
            self.active[role] -= 1
            self.condition.notify_all()

    @asynccontextmanager
    async def slot(self, dataset, timeout=900):
        role = workload(dataset)
        with self.condition:
            self.waiting[role] += 1
        acquired = False
        waiting = True
        try:
            deadline = time.monotonic() + timeout
            while not acquired:
                with self.condition:
                    if self._available(role):
                        self.active[role] += 1
                        self.waiting[role] -= 1
                        waiting = False
                        acquired = True
                if not acquired:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Shared sandbox capacity wait expired")
                    await asyncio.sleep(0.1)
            yield
        finally:
            if waiting:
                with self.condition:
                    self.waiting[role] -= 1
            if acquired:
                self.release(role)

    def snapshot(self):
        with self.condition:
            return {"capacity": self.total, "train_reserved": self.train_reserve,
                    "active": dict(self.active), "waiting": dict(self.waiting)}


admission = Admission(int(os.environ.get("SANDBOX_CAPACITY", "128")),
                      int(os.environ.get("TRAIN_RESERVED_SANDBOXES", "24")))
