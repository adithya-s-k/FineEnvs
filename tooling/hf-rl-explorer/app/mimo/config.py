"""The MiMo code's settings, from the explorer's own (app/config.py) plus what only MiMo needs."""

from __future__ import annotations

import os
from pathlib import Path

from ..config import (FLAVOR_PRICE_PER_HOUR, LOCAL_MODE, MAX_ACTIVE_PER_USER, MAX_ACTIVE_ROLLOUTS, PUBLIC_URL, ROUTER,  # noqa: F401
                      STORAGE_DIR)
from ..config import CACHE_DIR as _CACHE

DATASET = "XiaomiMiMo/MiMo-V2.6-RL-oss"
IMAGE_REPO = os.environ.get("IMAGE_REPO", "docker.io/xiaomimimo/mimo-v2.6-rl-oss")
DATA_DIR = Path(__file__).parent / "data"            # the browsing index (build_data.py)
CACHE_DIR = Path(os.environ.get("TASK_CACHE_DIR") or _CACHE / "mimo")   # General environments' folders, fetched on first use
# Code tasks' repositories (snapshot_repos.py in the MiMo explorer). On the Space: the MiMo explorer's own bucket,
# mounted read-only (MIMO_SNAPSHOT_DIR), so there is one copy and it is always the one MiMo's runs use.
SNAPSHOT_DIR = Path(os.environ.get("MIMO_SNAPSHOT_DIR") or STORAGE_DIR / "repo-snapshots")
SANDBOX_IDLE_TIMEOUT = int(os.environ.get("SANDBOX_IDLE_TIMEOUT", 1800))
# hardware a sandbox runs on, per domain
FLAVORS = {"code": "cpu-basic", "cyber": "cpu-basic", "general": "cpu-basic", "webdev": "cpu-upgrade"}
