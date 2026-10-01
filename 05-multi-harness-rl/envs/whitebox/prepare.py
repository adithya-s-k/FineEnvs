"""Materialize the fixed train/test tasks from corrected, pinned Harbor datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent


def task_rows(split):
    return json.loads((ROOT / "data" / f"{split}.json").read_text())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


DATASETS = {
    "train": (
        "FineEnvs/SmolDataEnvs-harbor-train",
        "4719635555de1666f374d847baebb500d368e493",
    ),
    "test": (
        "FineEnvs/SmolDataEnvs-harbor-test",
        "b130595b579fc6026ad5762925a9a3b40e72f5bb",
    ),
}
GRADER_SHA256 = "16f906e9248fa8b7e1141939e425e88e59d52d65e98e360bbacef3720a43629f"


def prepare(destination, sources=None):
    import tomli_w
    from huggingface_hub import snapshot_download

    destination = Path(destination).resolve()
    manifests = {}
    for split in ("train", "test"):
        rows = task_rows(split)
        repo, revision = DATASETS[split]
        source = (sources or {}).get(split)
        if source is None:
            source = snapshot_download(
                repo,
                repo_type="dataset",
                revision=revision,
                allow_patterns=[f"tasks/{r['name']}/*" for r in rows],
                max_workers=16,
            )
        source = Path(source)
        tasks = []
        for row in rows:
            folder = source / "tasks" / row["name"]
            instruction = (folder / "instruction.md").read_bytes()
            if hashlib.sha256(instruction).hexdigest() != row["instruction_sha256"]:
                raise ValueError(
                    f"Historical task instruction changed: {split}/{row['name']}"
                )
            grader = (folder / "tests/grader.py").read_bytes()
            if hashlib.sha256(grader).hexdigest() != GRADER_SHA256:
                raise ValueError(
                    f"Expected corrected numeric-only Harbor grader: {folder}"
                )
            target = destination / "datasets" / split / "tasks" / row["name"]
            target.mkdir(parents=True, exist_ok=True)
            for rel in (
                "instruction.md",
                "task.toml",
                "environment/Dockerfile",
                "environment/pull_bucket.py",
                "tests/grader.py",
                "tests/test.sh",
            ):
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(folder / rel, dst)
            spec = tomllib.loads((target / "task.toml").read_text())
            spec["environment"]["memory_mb"] = 4096
            spec["environment"]["cpus"] = 1
            # Cluster user IDs may not exist inside Daytona's user namespace.
            spec["environment"].setdefault("env", {})["TAR_OPTIONS"] = "--no-same-owner"
            (target / "task.toml").write_text(tomli_w.dumps(spec))
            hashes = {
                str(f.relative_to(destination / "datasets" / split)): hashlib.sha256(
                    f.read_bytes()
                ).hexdigest()
                for f in sorted(target.rglob("*"))
                if f.is_file()
            }
            tasks.append({**row, "file_hashes": hashes})
        local = (sources or {}).get(split)
        manifest = {
            "source_dataset": repo,
            "source_revision": None if local else revision,
            "expected_hub_revision": revision,
            "local_source": str(local) if local else None,
            "task_count": len(tasks),
            "tasks": tasks,
        }
        write_json(destination / f"{split}_manifest.json", manifest)
        manifests[split] = manifest
    train = {r["source"] for r in manifests["train"]["tasks"]}
    test = {r["source"] for r in manifests["test"]["tasks"]}
    notebooks = [
        {r.get("notebook") for r in manifests[s]["tasks"]} - {None}
        for s in ("train", "test")
    ]
    if train & test or notebooks[0] & notebooks[1]:
        raise ValueError("Train/test overlap")
    write_json(
        destination / "ready.json",
        {
            "train": len(train),
            "test": len(test),
            "grader_sha256": GRADER_SHA256,
            "train_test_overlap": 0,
        },
    )
    return manifests


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "prepared")
    p.add_argument("--train-source", type=Path)
    p.add_argument("--test-source", type=Path)
    a = p.parse_args()
    result = prepare(
        a.output,
        {s: v for s, v in (("train", a.train_source), ("test", a.test_source)) if v},
    )
    print(json.dumps({s: m["task_count"] for s, m in result.items()}))
