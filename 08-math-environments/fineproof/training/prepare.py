"""Write the Harbor tasks one training run uses: prepared/datasets/{train,test}/tasks and prepared/ready.json.

train  FineProofs-RL problems, decontaminated (fineproof_harbor.data: no test problems, no near-copies of test or
       benchmark problems, no duplicates), kept where QED-Nano's base model (Qwen3-4B-Thinking, 128 attempts graded
       by gpt-oss-20b) scored between --min-mean and --max-mean: hard enough to learn from, easy enough that a small
       model gets partial credit. The first --train-tasks of them in a seeded shuffle. Or, with --train-list, exactly
       the tasks a file names (see listed_train).
test   FineProofs-RL-test, all 128 problems, never trained on.

    python prepare.py                      # defaults below
    python prepare.py --train-tasks 64     # a smaller run
    python prepare.py --train-list benchmarks/fp-rl2-2b-train-tasks.txt   # the published run's 1,024 problems
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "envs" / "fineproof" / "harbor"))

HF_SANDBOX_WARNING = (
    "WARNING: --sandbox hf-sandbox turns the agent's network ON in every prepared task (network_mode = \"public\"), "
    "because HF Sandbox cannot change network policy after start. That is safe for the FineProof harness, which calls "
    "the model from the host and only writes proof.md into the sandbox, but these tasks are UNSAFE to reuse with an "
    "agent installed in the sandbox: most problems are public AoPS threads it could look up. Prepare such runs with "
    "--sandbox daytona.")


def for_hf_sandbox(task: Path) -> None:
    """Make a task runnable on Harbor's hf-sandbox backend (sandboxes as HF Jobs).

    HF Sandbox cannot build a Dockerfile, so the task names a prebuilt image: the Dockerfile's own pinned base
    (Python; sympy is not in it). It also cannot change network policy after start, so the agent phase keeps the
    network. Both are harmless for the FineProof harness, which calls the model from outside the sandbox and only
    writes proof.md into it; an agent that runs commands in the sandbox could search for the problem, so it must use
    Daytona and tasks prepared without this (main prints HF_SANDBOX_WARNING and ready.json records the network).
    """
    image = next(line.split()[1] for line in (task / "environment" / "Dockerfile").read_text().splitlines()
                 if line.startswith("FROM "))
    toml = task / "task.toml"
    text = toml.read_text()
    if "docker_image" not in text:
        text = text.replace("[environment]\n", f'[environment]\ndocker_image = "{image}"\n', 1)
    toml.write_text(text.replace('network_mode = "no-network"', 'network_mode = "public"'))


def read_list(path: Path) -> list[str]:
    """Task ids, one per line; blank lines and lines starting with `#` are skipped."""
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")]


def listed_train(names, train, removed, test_ids, raw_train):
    """Exactly the train tasks `names` lists, in that order, and the ids that needed the raw rows.

    Each id resolves against the decontaminated `train` first. An id that decontamination dropped (`removed` says
    why) falls back to its raw, undecontaminated FineProofs-RL row from `raw_train()` (called only then), with a
    warning naming it, so the list of an earlier run reproduces exactly. A test problem is never accepted, and an id
    in neither set is an error.
    """
    repeated = sorted(n for n, count in Counter(names).items() if count > 1)
    if repeated:
        raise ValueError(f"--train-list names {len(repeated)} tasks more than once: {repeated[:5]}")
    leaked = sorted(set(names) & set(test_ids))
    if leaked:
        raise ValueError(f"--train-list names {len(leaked)} test problems: {leaked[:5]}")
    kept = {r["id"]: r for r in train}
    dropped = [n for n in names if n not in kept]
    if dropped:
        raw = {r["id"]: r for r in raw_train()}
        unknown = [n for n in dropped if n not in raw]
        if unknown:
            raise ValueError(f"--train-list names {len(unknown)} tasks that are not in FineProofs-RL: {unknown[:5]}")
        why = {r["id"]: r for r in removed}
        for name in dropped:
            reason = why.get(name) or {}
            print(f"WARNING: {name} was dropped from train by decontamination ({reason.get('reason', 'reason unknown')}"
                  f"{', matches ' + reason['matches'] if reason.get('matches') else ''}); --train-list uses its raw "
                  "FineProofs-RL row", file=sys.stderr)
            kept[name] = raw[name]
    return [kept[n] for n in names], dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "prepared")
    ap.add_argument("--cache", type=Path, default=ROOT / ".cache")
    ap.add_argument("--train-tasks", type=int, default=1024)
    ap.add_argument("--min-mean", type=float, default=0.25)
    ap.add_argument("--max-mean", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-list", type=Path,
                    help="Build exactly the train tasks this file lists (one id per line, # comments) instead of "
                         "filtering and sampling; an id decontamination dropped falls back to its raw row, with a "
                         "warning")
    ap.add_argument("--sandbox", default="daytona", choices=["daytona", "hf-sandbox"],
                    help="hf-sandbox: run each task from its Dockerfile's base image with the network on (see for_hf_sandbox)")
    args = ap.parse_args()
    from fineproof_harbor import adapter, data

    paths = data.fetch(args.cache)
    sets = data.load_all(paths)  # train comes back decontaminated
    test = sets["test"]
    test_ids = {r["id"] for r in test}
    if args.train_list:
        names = read_list(args.train_list)
        train, fallback = listed_train(
            names, sets["train"], sets.get("removed", []), test_ids,
            lambda: data.fineproofs(paths["train"], "train", data.reference_proofs(paths["sft"])))
        selection = {"train_list": args.train_list.name,
                     "train_list_sha256": hashlib.sha256(args.train_list.read_bytes()).hexdigest(),
                     "raw_fallback": fallback}
    else:
        pool = [r for r in sets["train"]
                if r["id"] not in test_ids and args.min_mean <= r["reward_mean"] <= args.max_mean]
        random.Random(args.seed).shuffle(pool)
        train = pool[: args.train_tasks]
        selection = {"min_mean": args.min_mean, "max_mean": args.max_mean, "seed": args.seed, "pool": len(pool)}

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
    summary["selection"] = selection
    # The agent sandbox's network in these tasks; "public" makes them unsafe for an agent installed in the sandbox.
    summary["agent_network"] = "public" if args.sandbox == "hf-sandbox" else "none"
    (args.out / "ready.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.sandbox == "hf-sandbox":
        print(HF_SANDBOX_WARNING, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
