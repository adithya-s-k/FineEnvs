"""The lines weeks, which tools/build_lines.py builds locally from the wind weeks: the tests on them skip until they are
built, or fail with PORTSIM_REQUIRE_LINES=1."""

import os

import pytest
from port_week_core import lines

BUILT = all(p.exists() for p in (lines.pack_dir() / "tasks.jsonl", lines.REFERENCES))
REQUIRED = os.environ.get("PORTSIM_REQUIRE_LINES") == "1"
REASON = "lines weeks not built: run tools/build_lines.py"
real_lines = pytest.mark.skipif(not BUILT and not REQUIRED, reason=REASON)


def require_lines() -> None:
    """Skips the calling module, at its import, when the lines weeks aren't built."""
    if not BUILT and not REQUIRED:
        pytest.skip(REASON, allow_module_level=True)
