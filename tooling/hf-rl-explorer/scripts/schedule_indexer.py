"""Schedule the catalog indexer (app/indexer.py) as an HF scheduled Job. Prints what it would create; --apply creates it.

    uv run python scripts/schedule_indexer.py                       # print the Job and the equivalent `hf` command
    uv run python scripts/schedule_indexer.py --apply               # create it (asks nothing; needs write access to
                                                                    # FineEnvs' Jobs, and the explorer Space built)
    uv run python scripts/schedule_indexer.py --apply --replace     # replace one created before (same name label)

What the Job is:

  image     the explorer Space's own image, `hf.co/spaces/FineEnvs/RL-Explorer` (same code, same lockfile, already built
            by the Space; a Job can run any Space's image). The Space must exist and have built once.
  command   `python -m app.indexer --store /data --budget-min 40 --use-token`, from the image's app folder through
            uv, like the Space's own CMD.
  schedule  @hourly, with concurrency off: a run still going when the next is due makes the next one wait, never two
            writers. The run's own budget (40 min of index rebuilds) and the Job's timeout (55 min) keep it inside the
            hour.
  flavor    cpu-upgrade (8 vCPU, 32 GB): the full listing (every Space with its files) and a big dataset's index
            (13,825 tasks) fit with room to spare; SQLite builds on the container's local disk.
  volume    the private bucket FineEnvs/rl-explorer-data mounted read-write at /data (`Volume(type="bucket")`, the
            platform's hf-mount, the same volume the explorer and admin Spaces mount). STORAGE_DIR=/data, so catalog's
            own writers put indexes, packs and listing.json.gz exactly where the Spaces read them.
  secret    HF_TOKEN, a token of the Job's owner: raises the Hub's rate limits for downloading public dataset files
            (--use-token; listings and access checks stay anonymous, so nothing private is ever indexed). It's sent
            as a Job secret (encrypted, not shown in the Job's settings) and never printed here.

How the Job writes to the bucket. Two ways exist: the mounted volume (files written under /data are uploaded when
closed; a rename is one bucket batch, adding the new path and deleting the old, so temp-then-rename is atomic for
readers; other mounts see a change within their metadata TTL, ~10 s, and a full listing poll, ~30 s), or the bucket API
(`HfApi.batch_bucket_files`, one commit per call). The indexer writes through the mount: catalog's index writers
already work that way on the Spaces, and its snapshot store (`--store /data`) writes the database to a temp name,
renames it into `snapshots/`, checks it landed whole, then writes the pointer `snapshots/LATEST.v<SCHEMA>.json` last
the same way. `--store hf://buckets/FineEnvs/rl-explorer-data` publishes the snapshot through the bucket API instead
(the same two-step order: database, then pointer) if the mount ever proves unreliable for a large file.

The apps pick a new snapshot up on their own: each looks at the pointer every 60 s (app/snapshot.py). Nothing else
changes on the Spaces.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys

NAMESPACE = "FineEnvs"
NAME = "rl-explorer-indexer"
IMAGE = "hf.co/spaces/FineEnvs/RL-Explorer"
BUCKET = "FineEnvs/rl-explorer-data"
APP_DIR = "/home/user/app"   # the Dockerfile's WORKDIR


def spec(args: argparse.Namespace) -> dict:
    run = (f"cd {APP_DIR} && exec uv run --frozen --no-dev python -m app.indexer --store /data "
           f"--budget-min {args.budget_min:g} --max-builds {args.max_builds} --keep {args.keep}"
           + (" --use-token" if args.with_token else ""))
    return {
        "namespace": args.namespace,
        "name": args.name,
        "image": args.image,
        "command": ["sh", "-c", run],
        "schedule": args.schedule,
        "concurrency": False,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "env": {"STORAGE_DIR": "/data", "RLX_CACHE_DIR": "/tmp/rlx-cache", "HF_HOME": "/tmp/hf-home", "PYTHONUNBUFFERED": "1"},
        # anonymous by default: public datasets need no token. --with-token sends RLX_INDEXER_TOKEN (a read-only token
        # made for this) as HF_TOKEN, for the Hub's higher rate limits; never your login token, which can write
        "secrets": ["HF_TOKEN"] if args.with_token else [],
        "volumes": [{"type": "bucket", "source": args.bucket, "mount_path": "/data"}],
        "labels": {"app": "rl-explorer", "role": "indexer"},
    }


def cli(s: dict) -> str:
    """The same Job with the `hf` CLI."""
    parts = ["hf", "jobs", "scheduled", "run", s["schedule"], "--namespace", s["namespace"], "--name", s["name"],
             "--flavor", s["flavor"], "--timeout", s["timeout"], "--no-concurrency",
             "-v", f"hf://buckets/{s['volumes'][0]['source']}:{s['volumes'][0]['mount_path']}"]
    for k, v in s["env"].items():
        parts += ["-e", f"{k}={v}"]
    for k in s["secrets"]:
        parts += ["--secrets", k]
    for k, v in s["labels"].items():
        parts += ["-l", f"{k}={v}"]
    parts += [s["image"], *s["command"]]
    return " ".join(shlex.quote(p) for p in parts)


def apply(s: dict, replace: bool) -> int:
    from huggingface_hub import HfApi, Volume, get_token

    import os

    token = get_token()
    if not token:
        print("No HF token: run `hf auth login` (a token that can create Jobs in the FineEnvs organization).", file=sys.stderr)
        return 2
    job_token = os.environ.get("RLX_INDEXER_TOKEN")
    if s["secrets"] and not job_token:
        print("--with-token needs RLX_INDEXER_TOKEN: a read-only token made for the indexer (not your login token).", file=sys.stderr)
        return 2
    api = HfApi(token=token)
    existing = [j for j in api.list_scheduled_jobs(namespace=s["namespace"], labels={"name": s["name"]})]
    if existing and not replace:
        print(f"A scheduled Job named {s['name']} exists already ({', '.join(j.id for j in existing)}): --replace to replace it.",
              file=sys.stderr)
        return 1
    for j in existing:
        api.delete_scheduled_job(scheduled_job_id=j.id, namespace=s["namespace"])
        print(f"deleted scheduled Job {j.id}")
    job = api.create_scheduled_job(
        image=s["image"], command=s["command"], schedule=s["schedule"], concurrency=s["concurrency"], env=s["env"],
        secrets={k: job_token for k in s["secrets"]} or None, flavor=s["flavor"], timeout=s["timeout"], name=s["name"],
        labels=s["labels"], volumes=[Volume(**v) for v in s["volumes"]], namespace=s["namespace"])
    print(f"created scheduled Job {job.id} ({s['schedule']}); runs: hf jobs scheduled inspect {job.id} --namespace {s['namespace']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="create the scheduled Job (otherwise only print it)")
    ap.add_argument("--replace", action="store_true", help="with --apply: delete a scheduled Job of the same name first")
    ap.add_argument("--with-token", action="store_true", help="send RLX_INDEXER_TOKEN (a read-only token) as the Job's HF_TOKEN")
    ap.add_argument("--namespace", default=NAMESPACE)
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--image", default=IMAGE, help="a Docker image with this app: the explorer Space's, by default")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--schedule", default="@hourly", help='a preset (@hourly) or a cron expression ("17 * * * *")')
    ap.add_argument("--flavor", default="cpu-upgrade")
    ap.add_argument("--timeout", default="55m")
    ap.add_argument("--budget-min", type=float, default=40, help="minutes of index rebuilds per run")
    ap.add_argument("--max-builds", type=int, default=100)
    ap.add_argument("--keep", type=int, default=5, help="snapshots kept in the bucket")
    args = ap.parse_args(argv)
    s = spec(args)
    print(json.dumps(s, indent=1))
    print("\n# the same with the hf CLI:\n" + cli(s))
    if not args.apply:
        print("\n# nothing created: run again with --apply to create it")
        return 0
    return apply(s, args.replace)


if __name__ == "__main__":
    sys.exit(main())
