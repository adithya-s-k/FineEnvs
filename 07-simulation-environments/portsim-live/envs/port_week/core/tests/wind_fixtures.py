"""The wind fixtures: the wind files built from synthetic weather (synthetic.py) by tools/wind_references.py, for the
tests to use in place of the real pack; and the real wind weeks, which tools/build_wind.py builds locally from
Meteocat readings. Tests on the real weeks skip until they are built, or fail with PORTSIM_REQUIRE_WIND=1, and read
nothing at import."""

import json
import os
from functools import cache
from pathlib import Path

import pytest
from berth_core import Task, TaskPack

from port_week_core import packs, wind

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PACK_DIR = FIXTURES / "dock-v1-wind"
WEATHER = FIXTURES / "wind" / "weather.jsonl"
REFERENCES = FIXTURES / "wind" / "references.jsonl"
STORM_TASK = "dock-24B-w07x1-busy-0-e00"
BUST_TASK = "dock-24B-w07x1-busy-0-e01"
OTHER_TASK = "dock-36A-w37x1-standard-0-e00"
REAL_DIR = packs.tasks_root() / packs.WIND_PACK
REAL_WEATHER = packs.tasks_root() / "wind" / "weather.jsonl"
REAL_REFERENCES = packs.references("wind")
REAL_WEEKS = ["2023-W06", "2023-W10", "2023-W34", "2023-W35", "2023-W44", "2023-W50", "2024-W13", "2024-W18",
              "2024-W44", "2024-W46", "2024-W47", "2024-W49", "2025-W03", "2025-W14", "2025-W43", "2025-W52"]
REAL_BUSTS = ("2023-W44", "2025-W14")
REAL_TASKS = ["dock-24B-w06x1-busy-0-e15", "dock-24B-w07x1-busy-0-e04", "dock-24B-w07x1-busy-0-e08",
              "dock-24B-w07x1-busy-0-e09", "dock-24B-w16x1-busy-0-e12", "dock-24B-w35x1-busy-0-e00",
              "dock-36A-w05x1-busy-0-e09", "dock-36A-w05x1-busy-0-e14", "dock-36A-w06x1-busy-0-e07",
              "dock-36A-w06x1-busy-0-e12", "dock-36A-w17x1-busy-0-e12", "dock-36A-w37x1-busy-0-e12",
              "dock-24B-w37x1-standard-0-e01", "dock-36A-w06x1-standard-0-e07", "dock-36A-w35x1-standard-0-e03"]
"""The 16 weather weeks and the 15 dealt pairs: identifiers, not readings."""
BUILT = all(p.exists() for p in (REAL_DIR / "tasks.jsonl", REAL_WEATHER, REAL_REFERENCES))
real = pytest.mark.skipif(not BUILT and os.environ.get("PORTSIM_REQUIRE_WIND") != "1",
                          reason="wind weeks not built: run tools/build_wind.py")


def require_wind() -> None:
    """Skips the calling module, at its import, when the wind weeks aren't built."""
    if not BUILT and os.environ.get("PORTSIM_REQUIRE_WIND") != "1":
        pytest.skip("wind weeks not built: run tools/build_wind.py", allow_module_level=True)


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@cache
def real_weeks() -> list[dict]:
    return lines(REAL_WEATHER)


@cache
def real_refs() -> list[dict]:
    return lines(REAL_REFERENCES)


@cache
def real_tasks() -> dict[str, Task]:
    return {t.task_id: t for t in TaskPack(REAL_DIR).tasks}


def use(monkeypatch) -> None:
    monkeypatch.setattr(wind, "pack_dir", lambda: PACK_DIR)
    monkeypatch.setattr(wind, "REFERENCES", REFERENCES)


def tasks_dir(tmp_path: Path) -> str:
    """A PORT_WEEK_TASKS_DIR for a server subprocess: the committed marine pack, with the fixture wind pack and
    references beside it."""
    root = tmp_path / "tasks"
    root.mkdir()
    (root / packs.MARINE_PACK).symlink_to(packs.TASKS_ROOT / packs.MARINE_PACK)
    (root / packs.WIND_PACK).symlink_to(PACK_DIR)
    (root / "wind").symlink_to(FIXTURES / "wind")
    return str(root)
