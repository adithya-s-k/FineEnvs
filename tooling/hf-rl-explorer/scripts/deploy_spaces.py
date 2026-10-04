"""Deploy only the public HF RL Explorer. The admin is maintained in its private Space.

    uv run python scripts/deploy_spaces.py           # show the plan
    uv run python scripts/deploy_spaces.py --apply   # update FineEnvs/RL-Explorer

Local data, credentials, admin source, tests and deployment scripts are excluded.
The shared data bucket and existing session secret are preserved.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import secrets
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, Volume

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "FineEnvs/rl-explorer-data"
EXPLORER_URL = "https://fineenvs-rl-explorer.hf.space"
DATA = Volume(type="bucket", source=BUCKET, mount_path="/data")
SPACES = {
    "explorer": {
        "repo": "FineEnvs/RL-Explorer", "private": False, "visibility": "public", "readme": "SPACE_README.md", "hardware": "cpu-basic",
        "variables": {"MIMO_SNAPSHOT_DIR": "/mimo-snapshots"},
        "volumes": [DATA, Volume(type="bucket", source="FineEnvs/mimo-explorer-runs", mount_path="/mimo-snapshots",
                                 path="repo-snapshots", read_only=True)],
    },
}
# never uploaded: local data and caches, the READMEs (each Space gets its own as README.md), editor and OS files
IGNORE = [".venv/*", ".local-data/*", "*/__pycache__/*", "__pycache__/*", "*.pyc", ".pytest_cache/*", ".ruff_cache/*",
          ".gradio/*", "node_modules/*", ".DS_Store", "*/.DS_Store", "README.md", "SPACE_README.md", "ADMIN_SPACE_README.md", "EXPLORER_AUDIT.md",
          ".env", ".env.*", "*/.env", "*/.env.*", "*.env", "*.pem", "*.key", ".git/*"]


SKIP_DIRS = {".venv", ".local-data", "__pycache__", ".pytest_cache", ".ruff_cache", ".gradio", "node_modules", ".git"}


def files(target: str = "explorer") -> list[str]:
    """What goes into the Space: walked without descending into local data, caches or the virtualenv."""
    if target not in SPACES:
        raise ValueError("unknown deployment target")
    out = []
    for root, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not (Path(root) / d).is_symlink())
        for n in sorted(names):
            rel = (Path(root) / n).relative_to(ROOT).as_posix()
            if rel.startswith(("tests/", "scripts/")):
                continue
            if rel.startswith("web-admin/") or rel in ("app/admin.py", "app/admin_app.py"):
                continue
            if not (Path(root) / n).is_symlink() and not any(fnmatch.fnmatch(rel, pat) for pat in IGNORE):
                out.append(rel)
    return out


def deploy(api: HfApi, name: str, spec: dict, listing_from_store: bool, apply: bool) -> None:
    repo = spec["repo"]
    variables = {**spec["variables"], **({"RLX_LISTING_FROM_STORE": "1"} if listing_from_store else {})}
    paths = files(name)
    print(f"\n{name}: {repo} ({spec['visibility']}, {spec['hardware']})")
    print(f"  volumes: " + "; ".join(f"{v.source}{'/' + v.path if v.path else ''} -> {v.mount_path}{' (read-only)' if v.read_only else ''}" for v in spec["volumes"]))
    print(f"  variables: {variables}")
    print(f"  files: {len(paths)} + README.md from {spec['readme']}")
    if not apply:
        return
    api.create_repo(repo, repo_type="space", space_sdk="docker", private=spec["private"], exist_ok=True,
                    space_hardware=spec["hardware"])
    api.set_space_volumes(repo, spec["volumes"])
    for k, v in variables.items():
        api.add_space_variable(repo, k, v)
    if "SESSION_SECRET" not in {getattr(x, "key", x) for x in api.get_space_secrets(repo)}:   # names, or objects with .key
        api.add_space_secret(repo, "SESSION_SECRET", secrets.token_urlsafe(48), description="signs this Space's sessions")
        print("  SESSION_SECRET: created")
    runtime = api.get_space_runtime(repo)
    if (runtime.requested_hardware or runtime.hardware) != spec["hardware"]:
        api.request_space_hardware(repo, spec["hardware"])
    ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(ROOT / rel)) for rel in paths]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(ROOT / spec["readme"])))
    keep = set(paths) | {"README.md", ".gitattributes"}
    ops += [CommitOperationDelete(path_in_repo=f) for f in api.list_repo_files(repo, repo_type="space") if f not in keep]
    commit = api.create_commit(repo, repo_type="space", operations=ops, commit_message="Deploy HF RL Explorer")
    api.update_repo_settings(repo, repo_type="space", visibility=spec["visibility"])
    print(f"  pushed {commit.oid[:10]}: https://huggingface.co/spaces/{repo}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="create or update the Spaces (without it, print the plan)")
    ap.add_argument("--only", choices=sorted(SPACES), help="deploy one Space")
    ap.add_argument("--listing-from-store", action="store_true", help="read the indexer Job's listing (set once the Job runs)")
    args = ap.parse_args()
    api = HfApi()
    me = api.whoami()
    orgs = {o["name"]: o.get("roleInOrg") for o in me.get("orgs", [])}
    if orgs.get("FineEnvs") not in ("admin", "write", "contributor"):
        sys.exit(f"{me['name']} can't write to FineEnvs (role: {orgs.get('FineEnvs')})")
    for name, spec in SPACES.items():
        if not args.only or args.only == name:
            deploy(api, name, spec, args.listing_from_store, args.apply)
    if not args.apply:
        print("\nNothing changed: run with --apply.")


if __name__ == "__main__":
    main()
