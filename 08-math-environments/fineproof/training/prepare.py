"""Write the Harbor tasks one training run uses: prepared/datasets/{train,test}/tasks and prepared/ready.json.

train  FineProofs-RL problems minus the 128 test problems, kept where QED-Nano's base model (Qwen3-4B-Thinking,
       128 attempts graded by gpt-oss-20b) scored between --min-mean and --max-mean: hard enough to learn from,
       easy enough that a small model gets partial credit. The first --train-tasks of them in a seeded shuffle.
test   FineProofs-RL-test, all 128 problems, never trained on.

    python prepare.py                      # defaults below
    python prepare.py --train-tasks 64     # a smaller run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "envs" / "fineproof" / "harbor"))
from fineproof_harbor import adapter, data  # noqa: E402


def for_hf_sandbox(task: Path) -> None:
    """Make a task runnable on Harbor's hf-sandbox backend (sandboxes as HF Jobs).

    HF Sandbox cannot build a Dockerfile, so the task names a prebuilt image: the Dockerfile's own pinned base
    (Python; sympy is not in it). It also cannot change network policy after start, so the agent phase keeps the
    network. Both are harmless for the FineProof harness, which calls the model from outside the sandbox and only
    writes proof.md into it; an agent that runs commands in the sandbox should use Daytona.
    """
    image = next(line.split()[1] for line in (task / "environment" / "Dockerfile").read_text().splitlines()
                 if line.startswith("FROM "))
    toml = task / "task.toml"
    text = toml.read_text()
    if "docker_image" not in text:
        text = text.replace("[environment]\n", f'[environment]\ndocker_image = "{image}"\n', 1)
    toml.write_text(text.replace('network_mode = "no-network"', 'network_mode = "public"'))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "prepared")
    ap.add_argument("--cache", type=Path, default=ROOT / ".cache")
    ap.add_argument("--train-tasks", type=int, default=1024)
    ap.add_argument("--min-mean", type=float, default=0.25)
    ap.add_argument("--max-mean", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sandbox", default="daytona", choices=["daytona", "hf-sandbox"],
                    help="hf-sandbox: run each task from its Dockerfile's base image with the network on (see for_hf_sandbox)")
    args = ap.parse_args()

    sets = data.load_all(data.fetch(args.cache))
    test = sets["test"]
    test_ids = {r["id"] for r in test}
    pool = [r for r in sets["train"] if r["id"] not in test_ids and args.min_mean <= r["reward_mean"] <= args.max_mean]
    random.Random(args.seed).shuffle(pool)
    train = pool[: args.train_tasks]

    if args.out.exists():
        shutil.rmtree(args.out)
    summary = {}
    for split, records in (("train", train), ("test", test)):
        root = args.out / "datasets" / split
        for r in records:
            adapter.write_task(r, root)
            if args.sandbox == "hf-sandbox":
                for_hf_sandbox(root / "tasks" / r["id"])
        names = sorted(r["id"] for r in records)
        summary[split] = {
            "tasks": len(names),
            "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "mean_base_reward": round(sum(r["reward_mean"] for r in records) / len(records), 4),
        }
    summary["selection"] = {"min_mean": args.min_mean, "max_mean": args.max_mean, "seed": args.seed,
                            "pool": len(pool)}
    (args.out / "ready.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
