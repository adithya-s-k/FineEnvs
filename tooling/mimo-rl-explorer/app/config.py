"""Settings, all from the environment, so the same code runs on a laptop and in the Space.

Local (no OAuth app): your own HF token (HF_TOKEN or `hf auth login`) runs the rollouts, and traces
are written to ./.local-runs. On the Space: users sign in with HF, their token runs *their* rollouts,
and traces go to the private bucket mounted at /data.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

DATASET = "XiaomiMiMo/MiMo-V2.6-RL-oss"
IMAGE_REPO = os.environ.get("IMAGE_REPO", "docker.io/xiaomimimo/mimo-v2.6-rl-oss")
ROUTER = os.environ.get("HF_ROUTER_URL", "https://router.huggingface.co/v1")

# OAuth is on when the Space has `hf_oauth: true`; the Hub injects these.
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co")
SPACE_HOST = os.environ.get("SPACE_HOST", "")
# Where a sandbox can reach this server. When set (always on a Space), sandboxes never hold the user's HF token or
# endpoint key: model and judge calls go through /api/llm/<per-rollout capability>. Locally a remote sandbox can't
# reach your machine, so the token is forwarded into it instead (set MIMO_PUBLIC_URL to a tunnel to avoid that).
PUBLIC_URL = (os.environ.get("MIMO_PUBLIC_URL") or (f"https://{SPACE_HOST}" if SPACE_HOST else "")).rstrip("/")
LOCAL_MODE = not OAUTH_CLIENT_ID
# `jobs` runs the sandbox on the user's account, `inference-api` calls Inference Providers as them.
OAUTH_SCOPES = ["openid", "profile", "inference-api", "jobs"]
SESSION_SECRET = os.environ.get("SESSION_SECRET") or OAUTH_CLIENT_SECRET or secrets.token_hex(32)
SESSION_DAYS = 7

_on_space = bool(os.environ.get("SPACE_ID"))
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR") or ("/data" if _on_space and Path("/data").is_dir() else ROOT / ".local-runs"))
CACHE_DIR = Path(os.environ.get("TASK_CACHE_DIR") or STORAGE_DIR / "task-cache")

# Capacity. A rollout is mostly waiting on the model, so a thread each is plenty.
MAX_ACTIVE_ROLLOUTS = int(os.environ.get("MAX_ACTIVE_ROLLOUTS", 24))
MAX_ACTIVE_PER_USER = int(os.environ.get("MAX_ACTIVE_PER_USER", 4))
SANDBOX_IDLE_TIMEOUT = int(os.environ.get("SANDBOX_IDLE_TIMEOUT", 1800))

# Hardware the sandbox runs on, per domain (price per hour, from `hf jobs hardware`).
FLAVORS = {"code": "cpu-basic", "cyber": "cpu-basic", "general": "cpu-basic", "webdev": "cpu-upgrade"}
FLAVOR_PRICE_PER_HOUR = {"cpu-basic": 0.01, "cpu-upgrade": 0.03}
