from pathlib import Path

import pytest

from airops_core.core.model import Task, load_pack

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tasks.jsonl"


@pytest.fixture(scope="session")
def pack() -> list[Task]:
    return load_pack(FIXTURES)


@pytest.fixture(scope="session")
def by_id(pack) -> dict[str, Task]:
    return {t.task_id: t for t in pack}
