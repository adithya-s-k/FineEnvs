"""Settings, all from the environment, so the same code runs on a laptop and in the Space.

Local (no OAuth app): your own HF token (HF_TOKEN or `hf auth login`) reads datasets and runs rollouts, and
everything is written to ./.local-data. On the Space: visitors sign in with HF, their token runs *their*
rollouts, and runs and dataset indexes go to the private bucket mounted at /data.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"

ROUTER = os.environ.get("HF_ROUTER_URL", "https://router.huggingface.co/v1")

# OAuth is on when the Space has `hf_oauth: true`; the Hub injects these.
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co")
SPACE_HOST = os.environ.get("SPACE_HOST", "")
PUBLIC_URL = (os.environ.get("RLX_PUBLIC_URL") or (f"https://{SPACE_HOST}" if SPACE_HOST else "")).rstrip("/")
LOCAL_MODE = not OAUTH_CLIENT_ID
# `jobs` runs the sandbox on the visitor's account, `inference-api` calls Inference Providers as them,
# `read-repos` opens their private and gated datasets.
OAUTH_SCOPES = os.environ.get("RLX_OAUTH_SCOPES", "openid profile inference-api jobs read-repos").split()   # the admin Space asks for less
SESSION_SECRET = os.environ.get("SESSION_SECRET") or OAUTH_CLIENT_SECRET or secrets.token_hex(32)
SESSION_DAYS = 7

_on_space = bool(os.environ.get("SPACE_ID"))
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR") or ("/data" if _on_space and Path("/data").is_dir() else ROOT / ".local-data"))
# Dataset files read for indexes and task pages: thousands of small files, which a bucket mount is slow at,
# so they live on local disk; the indexes built from them are kept in STORAGE_DIR.
CACHE_DIR = Path(os.environ.get("RLX_CACHE_DIR") or ROOT / ".local-data" / "hub-cache")
INDEX_DIR = STORAGE_DIR / "indexes"
# every task's text per dataset, one file each: what makes a task page instant (see catalog's packs)
PACK_DIR = STORAGE_DIR / "packs"

# How many tasks a dataset may have and still be indexed from the page, and how many files its listing may hold.
MAX_INDEX_TASKS = int(os.environ.get("RLX_MAX_INDEX_TASKS", 20000))
MAX_LISTING_FILES = int(os.environ.get("RLX_MAX_LISTING_FILES", 400000))
INDEX_WORKERS = int(os.environ.get("RLX_INDEX_WORKERS", 16))
# Indexing is started by whoever opens a dataset first: at most this many at once, so nobody can swamp the server.
MAX_INDEX_JOBS = int(os.environ.get("RLX_MAX_INDEX_JOBS", 3))
# A rollout downloads its task's folder (all but the solution): refuse tasks bigger than this.
MAX_TASK_BYTES = int(os.environ.get("RLX_MAX_TASK_BYTES", 2 * 1024**3))
# Downloads of *public* dataset files may use this token (higher Hub rate limits). Listing and access checks stay
# anonymous, so it never opens a private dataset. Set by the precache command, not by the Space.
INDEX_TOKEN = os.environ.get("RLX_INDEX_TOKEN") or None

# How the capture proxy is reached from a sandbox when this runs locally (OpenEnv's forwarders: gradio, cloudflare,
# direct). On a Space it is mounted into this app instead.
CAPTURE_EXPOSE = os.environ.get("RLX_CAPTURE_EXPOSE", "gradio")

# Capacity. A rollout is mostly waiting on the model, so a thread each is plenty.
MAX_ACTIVE_ROLLOUTS = int(os.environ.get("MAX_ACTIVE_ROLLOUTS", 24))
MAX_ACTIVE_PER_USER = int(os.environ.get("MAX_ACTIVE_PER_USER", 4))

# Hardware a sandbox runs on, with its price per hour (from `hf jobs hardware`), for the cost shown.
FLAVOR_PRICE_PER_HOUR = {"cpu-basic": 0.01, "cpu-upgrade": 0.03}
