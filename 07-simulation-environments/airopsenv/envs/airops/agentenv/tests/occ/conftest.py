import sys

import pytest
from paths import ENV

sys.path[:0] = [str(ENV / "core" / "tests" / "occ"), str(ENV / "core" / "tests" / "programs")]


@pytest.fixture
def anyio_backend():
    return "asyncio"
