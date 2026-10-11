"""The synthetic wind weeks the core's tests use (../core/tests/fixtures: invented weather, no Meteocat data), in place
of the real wind pack, which tools/build_wind.py builds locally."""

from pathlib import Path

from port_week_core import packs, wind

FIXTURES = Path(__file__).resolve().parents[2] / "core" / "tests" / "fixtures"
PACK_DIR = FIXTURES / "dock-v1-wind"
REFERENCES = FIXTURES / "wind" / "references.jsonl"
STORM_TASK = "dock-24B-w07x1-busy-0-e00"
BUST_TASK = "dock-24B-w07x1-busy-0-e01"


def use(monkeypatch) -> None:
    monkeypatch.setattr(wind, "pack_dir", lambda: PACK_DIR)
    monkeypatch.setattr(wind, "REFERENCES", REFERENCES)


def tasks_dir(tmp_path: Path) -> Path:
    """A PORT_WEEK_TASKS_DIR for a server: the committed marine pack, with the fixture wind pack and references beside
    it."""
    root = tmp_path / "tasks"
    root.mkdir()
    (root / packs.MARINE_PACK).symlink_to(packs.TASKS_ROOT / packs.MARINE_PACK)
    (root / packs.WIND_PACK).symlink_to(PACK_DIR)
    (root / "wind").symlink_to(FIXTURES / "wind")
    return root
