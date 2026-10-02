import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for mode in ("whitebox", "opencode", "harbor"):
    sys.path.insert(0, str(ROOT / "envs" / mode))

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--contract",
        action="store_true",
        help="Run integration checks against installed TRL and OpenEnv",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "contract: requires the tutorial's installed runtime"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--contract"):
        for item in items:
            if "contract" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Use --contract after installing TRL/OpenEnv"
                    )
                )
