import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_ENV_ROOT = str(ROOT / "envs")
if _ENV_ROOT not in sys.path:
    sys.path.insert(0, _ENV_ROOT)

from trace_audit.core import load_records  # noqa: E402


@pytest.fixture
def rows():
    return load_records(ROOT / "data/sample.jsonl")
