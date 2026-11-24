"""The lines weeks, which tools/build_lines.py builds locally from the wind weeks: the tests on them skip until they are
built, or fail with PORTSIM_REQUIRE_LINES=1, and read nothing before deciding.

fixtures/dock-v1-lines holds the synthetic wind weeks with the ships' agents, for the server's tests, as
`tools/lines_references.py pack --out core/tests/fixtures` writes them with PORT_WEEK_TASKS_DIR holding
fixtures/dock-v1-wind and fixtures/wind beside the committed dock-v1-marine. It is never screened, so its anchors stay
the wind weeks'."""

import os

import pytest

from port_week_core import lines

BUILT = all(p.exists() for p in (lines.pack_dir() / "tasks.jsonl", lines.REFERENCES))


def require_lines() -> None:
    """Skips the calling module, at its import, when the lines weeks aren't built."""
    if not BUILT and os.environ.get("PORTSIM_REQUIRE_LINES") != "1":
        pytest.skip("lines weeks not built: run tools/build_lines.py", allow_module_level=True)
