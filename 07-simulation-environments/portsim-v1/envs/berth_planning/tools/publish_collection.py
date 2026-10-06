"""Create or refresh the FineEnvs collection "Simulation RL Envs" and add every artifact that exists.

    openenv/.venv/bin/python tools/publish_collection.py [--patch-article]

--patch-article writes the collection's URL into the article's data chapter (before the article is deployed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DATA_MDX = ROOT / "content/articles/simulation-rl-environments/app/src/content/chapters/data.mdx"
PLACEHOLDER = "https://huggingface.co/collections/FineEnvs)"
ITEMS = [
    ("FineEnvs/simulation-rl-environments", "space", "The article: Simulation RL Environments, part 1"),
    ("FineEnvs/PortSimEnv", "space", "PortSimEnv v1: the OpenEnv environment, with the editor and 3D view"),
    ("FineEnvs/PortSimEnv-Eval", "space", "PortSimEnv v1 eval: the eval table and every model rollout in 3D"),
    ("FineEnvs/PortSimEnv", "dataset", "PortSimEnv v1: tasks, the source port calls, eval rollouts"),
]


def main(argv=None) -> int:
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch-article", action="store_true")
    args = ap.parse_args(argv)
    api = HfApi()
    col = api.create_collection(
        "Simulation RL Envs", namespace="FineEnvs", private=False, exists_ok=True,
        description="Real-world work rebuilt as deterministic RL simulations from seed data. Part 1: PortSimEnv v1, "
                    "berth planning at the Port of Barcelona.")
    for repo_id, kind, note in ITEMS:
        if api.repo_exists(repo_id, repo_type=kind):
            api.add_collection_item(col.slug, repo_id, kind, note=note, exists_ok=True)
            print(f"+ {kind} {repo_id}")
        else:
            print(f"- {kind} {repo_id} does not exist yet")
    url = f"https://huggingface.co/collections/{col.slug}"
    if args.patch_article and PLACEHOLDER in DATA_MDX.read_text():
        DATA_MDX.write_text(DATA_MDX.read_text().replace(PLACEHOLDER, url + ")"))
        print(f"article links {url}")
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
