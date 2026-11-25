# SPDX-License-Identifier: BSD-3-Clause

"""Deploy the GeoGuesser environment to the Hugging Face Hub.

Three artifacts, split by what each is good at:

    bucket   HuggingEnvs/geoguesser-panos   22 GB of imagery + both indexes
    dataset  HuggingEnvs/geoguesser-tasks    the indexes alone, versioned
    space    AdithyaSK/geoguesser-env        the running environment

Override the owners with GEOGUESSER_HF_ORG (Space and dataset) and
GEOGUESSER_HF_BUCKET_OWNER (bucket).

The bucket exists because Space disk is ephemeral and capped well below 22 GB,
and because buckets are mutable object storage rather than git. The dataset repo
exists because a bucket is *not* versioned: a frozen benchmark needs a place
where a change to it is visible in history. The Space mounts the bucket
read-only at /data, so the same image serves a local checkout and the Hub with
nothing but environment variables between them.

`openenv push` is deliberately not used: it cannot attach a bucket volume, and
its default excludes would upload 22 GB of panoramas into git.

Usage:
    export HF_TOKEN=hf_...
    python dataset/deploy_hub.py --all
    python dataset/deploy_hub.py --space          # code only, fast iteration
    python dataset/deploy_hub.py --verify         # check a live deployment
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import shutil
import sys
import tempfile

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("deploy_hub")

ROOT = pathlib.Path(__file__).resolve().parents[1] / "env"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Owner of the Space and dataset. The bucket is separate on purpose: 22 GB of
# imagery should not be re-uploaded just because the Space moves namespace, and
# a Space can mount a bucket it has access to regardless of who owns it.
ORG = os.getenv("GEOGUESSER_HF_ORG", "HuggingEnvs")
BUCKET_OWNER = os.getenv("GEOGUESSER_HF_BUCKET_OWNER", ORG)
SPACE_ID = f"{ORG}/geoguesser-env"
BUCKET_ID = f"{BUCKET_OWNER}/geoguesser-panos"
DATASET_ID = f"{ORG}/geoguesser-tasks"
MOUNT_PATH = "/data"

EVAL_INDEX = "eval_pano_v3.jsonl"
TRAIN_INDEX = "train_pano_v3.jsonl"

# Anything matching these never reaches the Space repo. The panorama cache and
# the sequence pool are the dangerous ones: 22 GB and 254 MB respectively.
SPACE_EXCLUDES = (
    "geoguesser_env/data/panos/*",
    "geoguesser_env/data/pool/*",
    f"geoguesser_env/tasks/{TRAIN_INDEX}",
    "geoguesser_env/tasks/pool_offline_5k.jsonl",
    "geoguesser_env/tasks/eval_pano_v1.jsonl",
    "geoguesser_env/tasks/eval_pano_v2.jsonl",
    "geoguesser_env/tasks/eval_balanced_trial.jsonl",
    "geoguesser_env/rollouts/*",
    "geoguesser_env/tests/*",
    "geoguesser_env/.venv/*",
    # The Remotion project is a local video tool, not part of the environment.
    # Its node_modules also carries a .cache/ directory, which the Hub rejects
    # outright, so an unexcluded video/ fails the whole upload rather than just
    # bloating it.
    "geoguesser_env/video/*",
    "**/node_modules/*",
    "**/.cache/*",
    # Endpoint configs hold live tunnel URLs and both repos are public.
    "geoguesser_env/*models.json",
    # setuptools staging. The Space is pip-installed, and setuptools reuses
    # build/lib as its staging directory -- so an uploaded build/ shadows the
    # real sources and silently ships stale code. This cost three Job runs.
    "geoguesser_env/build/*",
    "**/*.egg-info/*",
    "**/__pycache__/*",
    "**/*.pyc",
    "src/**/__pycache__/*",
)

SPACE_VARIABLES = {
    "ENABLE_WEB_INTERFACE": "true",
    "GEOGUESSER_TASKS_EVAL": f"{MOUNT_PATH}/tasks/{EVAL_INDEX}",
    "GEOGUESSER_TASKS_TRAIN": f"{MOUNT_PATH}/tasks/{TRAIN_INDEX}",
    # Both splits are fully mirrored in the bucket, so either works. Override
    # with GEOGUESSER_SPACE_DEFAULT_SPLIT while a mirror is still uploading:
    # defaulting to a split whose frames are absent makes the Space look broken.
    "GEOGUESSER_DEFAULT_SPLIT": os.getenv("GEOGUESSER_SPACE_DEFAULT_SPLIT", "train"),
    "GEOGUESSER_CACHE": f"{MOUNT_PATH}/panos",
    # The mirror is complete, so a cache miss is a bug worth hearing about
    # rather than something to paper over with a network call.
    "GEOGUESSER_ALLOW_FETCH": "0",
    # Overpass only, and it caches to the container's own disk, so labelled
    # streets work even with imagery fetching disabled.
    "GEOGUESSER_STREET_DETAIL": "1",
    "GEOGUESSER_MAX_STEPS": "24",
    # Pinned rather than inherited. This Space serves play traffic and the
    # published leaderboard, so it keeps the game's own curve and shows the
    # Mapillary contributor credit. RL training wants the opposite on all three
    # -- GEOGUESSER_REWARD_SHAPE=mixture, GEOGUESSER_COST_MODE=multiply,
    # GEOGUESSER_HIDE_IDENTITY=1 -- because subtracting the cost floors 10-46%
    # of episodes at exactly 0.0 and the contributor username alone determines
    # the country for 74% of training tasks.
    "GEOGUESSER_REWARD_SHAPE": "geoguessr",
    "GEOGUESSER_COST_MODE": "subtract",
    "GEOGUESSER_HIDE_IDENTITY": "0",
    # Core defaults to 4 concurrent sessions, which is a demo setting. An eval
    # sweep opens one session per worker per endpoint and the surplus fail
    # outright with CAPACITY_REACHED.
    "MAX_CONCURRENT_ENVS": "64",
}

CARD_FRONTMATTER = """---
title: Geoguesser Environment
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - geolocation
  - mcp-server
---

"""


def token() -> str:
    """Resolve an HF token from the environment or the repo `.env`."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(name)
        if value:
            return value
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no HF_TOKEN; export it or add it to the repo .env")


def find_openenv_src() -> pathlib.Path:
    """
    Locate the OpenEnv source tree the Space image vendors.

    The image puts `src/openenv` on `PYTHONPATH` instead of pip-installing a
    release, because the Gradio tab naming this environment uses is newer than
    the published package. That means deploying needs an OpenEnv checkout, which
    this project no longer contains -- so say where to find one rather than
    copying a path that silently does not exist.

    Returns:
        `pathlib.Path`: a directory containing `openenv/`.
    """
    candidates = []
    if os.getenv("OPENENV_SRC"):
        candidates.append(pathlib.Path(os.environ["OPENENV_SRC"]))
    candidates += [
        REPO_ROOT.parent / "OpenEnv" / "src",       # sibling checkout
        REPO_ROOT.parent.parent / "OpenEnv" / "src",
    ]
    for c in candidates:
        if (c / "openenv").is_dir():
            return c
    raise SystemExit(
        "cannot find the OpenEnv source tree, which the Space image vendors.\n"
        "Clone it beside this repository, or point OPENENV_SRC at its src/:\n"
        "  git clone https://github.com/openenv/OpenEnv\n"
        "  export OPENENV_SRC=$PWD/OpenEnv/src\n"
        "Tried: " + ", ".join(str(c) for c in candidates)
    )


def stage_space(staging: pathlib.Path) -> None:
    """
    Assemble the Space repository layout.

    The Space is self-contained: it vendors `src/openenv` rather than installing
    a release, because the Gradio tab naming this environment uses is newer than
    the published package.
    """
    shutil.copytree(
        find_openenv_src(),
        staging / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    shutil.copytree(
        ROOT,
        staging / "geoguesser_env",
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            ".venv",
            "panos",
            "pool",
            "rollouts",
            "tests",
            "space",
            "osm_cache",
            TRAIN_INDEX,
            "pool_offline_5k.jsonl",
            "eval_pano_v1.jsonl",
            "eval_pano_v2.jsonl",
            "eval_balanced_trial.jsonl",
        ),
    )
    shutil.copy(ROOT / "Dockerfile", staging / "Dockerfile")
    # The card is the env README with Space frontmatter prepended, so the Hub
    # page and the repo documentation cannot drift apart.
    body = (ROOT / "README.md").read_text()
    if body.startswith("---"):
        body = body.split("---", 2)[-1].lstrip()
    (staging / "README.md").write_text(CARD_FRONTMATTER + body)

    # The detail vectors drive the labelled maps and are gitignored locally, so
    # confirm they made it rather than shipping a Space with bare maps.
    detail = staging / "geoguesser_env" / "data" / "geo" / "detail"
    present = sorted(p.name for p in detail.glob("*.json")) if detail.exists() else []
    missing = {"places.json", "roads.json", "rivers.json", "urban.json"} - set(present)
    if missing:
        logger.warning(
            "detail vectors missing from the staged Space: %s -- agent maps will "
            "have no towns or roads. Run dataset/fetch_detail_geo.py first.",
            sorted(missing),
        )
    size = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
    logger.info(
        "staged %d files, %.1f MB",
        sum(1 for p in staging.rglob("*") if p.is_file()),
        size / 1e6,
    )
    if size > 500e6:
        raise SystemExit(
            f"staged Space is {size / 1e6:.0f} MB, which means an exclude did "
            "not match. Refusing to push."
        )


def push_space(hf_token: str) -> None:
    """Create the Space if needed, upload the code, and wire up its runtime."""
    from huggingface_hub import HfApi, Volume

    api = HfApi(token=hf_token)
    api.create_repo(
        SPACE_ID, repo_type="space", space_sdk="docker", private=False, exist_ok=True
    )

    with tempfile.TemporaryDirectory() as tmp:
        staging = pathlib.Path(tmp) / "space"
        staging.mkdir()
        stage_space(staging)
        logger.info("uploading to %s", SPACE_ID)
        api.upload_folder(
            repo_id=SPACE_ID,
            repo_type="space",
            folder_path=str(staging),
            ignore_patterns=list(SPACE_EXCLUDES),
            commit_message="Deploy geoguesser environment with train/eval splits",
        )

    for key, value in SPACE_VARIABLES.items():
        api.add_space_variable(repo_id=SPACE_ID, key=key, value=value)
    logger.info("set %d Space variables", len(SPACE_VARIABLES))

    # set_space_volumes REPLACES the whole list, so anything already attached
    # has to be carried forward or it is silently unmounted.
    existing = []
    try:
        existing = list(api.get_space_runtime(SPACE_ID).volumes or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read current volumes: %r", exc)
    keep = [
        v
        for v in existing
        if getattr(v, "mount_path", None) != MOUNT_PATH
        and getattr(v, "source", None) != BUCKET_ID
    ]
    api.set_space_volumes(
        repo_id=SPACE_ID,
        volumes=keep
        + [
            Volume(
                type="bucket",
                source=BUCKET_ID,
                mount_path=MOUNT_PATH,
                read_only=True,
            )
        ],
    )
    logger.info("mounted %s read-only at %s", BUCKET_ID, MOUNT_PATH)
    logger.info("space: https://huggingface.co/spaces/%s", SPACE_ID)


def push_dataset(hf_token: str) -> None:
    """Publish the task indexes and a card describing how they were built."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(DATASET_ID, repo_type="dataset", private=False, exist_ok=True)

    eval_path = ROOT / "tasks" / EVAL_INDEX
    train_path = ROOT / "tasks" / TRAIN_INDEX
    stats = {}
    for name, path in (("eval", eval_path), ("train", train_path)):
        if not path.exists():
            logger.warning("%s missing, skipping: %s", name, path)
            continue
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        stats[name] = {
            "tasks": len(rows),
            "countries": len({r["country"] for r in rows}),
            "frames": sum(len(r["frames"]) for r in rows),
            "offline": sum(1 for r in rows if r["meta"].get("offline_ready")),
        }

    with tempfile.TemporaryDirectory() as tmp:
        staging = pathlib.Path(tmp)
        for path in (eval_path, train_path):
            if path.exists():
                shutil.copy(path, staging / path.name)
        (staging / "README.md").write_text(dataset_card(stats))
        api.upload_folder(
            repo_id=DATASET_ID,
            repo_type="dataset",
            folder_path=str(staging),
            commit_message="GeoGuesser task splits",
        )
    logger.info("dataset: https://huggingface.co/datasets/%s", DATASET_ID)


def dataset_card(stats: dict) -> str:
    """Render the dataset card from measured figures, never hardcoded ones."""
    rows = "\n".join(
        f"| `{name}` | {s['tasks']} | {s['countries']} | {s['frames']} | "
        f"{s['offline']}/{s['tasks']} |"
        for name, s in stats.items()
    )
    return f"""---
license: cc-by-sa-4.0
task_categories:
  - image-classification
tags:
  - geolocation
  - openenv
  - reinforcement-learning
pretty_name: GeoGuesser Task Splits
---

# GeoGuesser Task Splits

Task indexes for the [GeoGuesser OpenEnv environment](https://huggingface.co/spaces/{SPACE_ID}).
Each line is one episode: an ordered list of panorama frames with coordinates,
headings and capture dates, plus the sequence and contributor it came from.

| Split | Tasks | Countries | Frames | Fully mirrored |
|---|---|---|---|---|
{rows}

## What a task is

These files carry **metadata only**, not imagery. Every frame's coordinates,
heading and capture date are here, so the movement graph resolves with no
network access; only image bytes need fetching, and Mapillary's `thumb_*_url`
values are expiring signed CDN URLs that cannot be stored. Resolve them from
`image_id` through the Mapillary Graph API, or mirror them once with
`dataset/build_tasks.py` from the environment repository.

## How the split was made

Both splits are carved from one 3,673-task pool, so contamination is enforced
exactly once rather than reasoned about across two separate harvests. The rules
follow the OSV-5M paper, which built its train/test split from the same
Mapillary source:

- no shared `sequence_id` between splits
- no training task within **1 km** of an eval task

The buffer matters because frames sit about 3.3 m apart: holding out an image
while keeping its neighbour holds out nothing. The split script verifies its own
output and exits non-zero if either rule is violated.

Eval is carved first, balanced by country and capped at 4 tasks each, because at
a couple of hundred tasks the balance decides what the score means. An earlier
unbalanced attempt put 28% of the set in one country.

## Provenance

Imagery is from [Mapillary](https://www.mapillary.com), CC BY-SA 4.0. Each task
records its contributor in `attribution`, which the environment displays. Only
360-degree panoramas are included (`camera_type == "spherical"`; note that the
documented value `equirectangular` does not appear in practice).

Sequences were discovered by enumerating Mapillary's `mly1_public` vector tiles
at zoom 6, where the sequence layer carries `is_pano` — 1.2 million panorama
sequences worldwide. Candidates are sampled with weight proportional to local
image density raised to **-0.75**, the OSV-5M weighting, then capped per country
and per contributor: one contributor alone holds 8% of the pool.

## Reproducing

```bash
git clone https://github.com/openenv/OpenEnv
cd HuggingEnvs/03-geoguesser/env
export MAPILLARY_API_KEY_TRAIN="MLY|..."

python dataset/harvest_tiles.py                 # enumerate sequences worldwide
./dataset/build_dataset.sh                      # assemble and mirror tasks
python ../dataset/verify_offline.py tasks/pool_offline_5k.jsonl
python dataset/split_tasks.py tasks/pool_offline_5k.jsonl --eval 200
```

A rebuild will not reproduce these exact tasks — the pool is sampled and
upstream coverage changes — which is precisely why the split is published rather
than left to be regenerated.
"""


def verify(hf_token: str) -> int:
    """Exercise a deployed Space the way a client would. Returns an exit code."""
    import urllib.error
    import urllib.request

    base = f"https://{ORG.lower()}-geoguesser-env.hf.space"
    logger.info("verifying %s", base)
    failures = []

    def get(path: str) -> object:
        request = urllib.request.Request(
            base + path, headers={"Authorization": f"Bearer {hf_token}"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())

    try:
        get("/health")
        logger.info("  [PASS] /health")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"/health: {exc!r}")
        logger.error("  [FAIL] /health: %r", exc)
        return 1

    try:
        splits = get("/geoguesser_env/splits")
        names = {s["name"]: s["num_tasks"] for s in splits}
        logger.info("  [PASS] /splits -> %s", names)
        if "eval" not in names:
            failures.append("no eval split; is the bucket mounted?")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"/splits: {exc!r}")
        logger.error("  [FAIL] /splits: %r", exc)

    for failure in failures:
        logger.error("  %s", failure)
    return 1 if failures else 0


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Space and dataset.")
    parser.add_argument("--space", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if not any((args.all, args.space, args.dataset, args.verify)):
        parser.error("pick at least one of --all, --space, --dataset, --verify")

    hf_token = token()
    if args.all or args.dataset:
        push_dataset(hf_token)
    if args.all or args.space:
        push_space(hf_token)
    if args.verify:
        sys.exit(verify(hf_token))


if __name__ == "__main__":
    main()
