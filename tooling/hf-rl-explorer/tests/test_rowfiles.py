"""Training commands keep a known split's files, even when it has many shards. No Hub or trainer calls."""

from __future__ import annotations

import ast
import glob
import shlex

import pytest

from app.envs import rowfiles, verl_rows
from app.envs.base import Dataset


@pytest.mark.parametrize("count", [40, 41])
def test_known_split_does_not_include_neighboring_validation_shards(tmp_path, monkeypatch, count):
    monkeypatch.chdir(tmp_path)
    train = [f"shared/train-{i:05d}.parquet" for i in range(count)]
    validation = "shared/validation-00000.parquet"
    for rel in [*train, validation]:
        p = tmp_path / rowfiles.local_dir("org/ds") / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()

    selected = [p for pattern in rowfiles.local_paths("org/ds", train, "**/*train*.parquet")
                for p in glob.glob(pattern, recursive=True)]
    assert sorted(selected) == sorted(f"data/ds/{rel}" for rel in train)


def test_known_split_keeps_shards_across_directories_and_formats():
    files = [f"part-a/train-{i:05d}.parquet" for i in range(40)] + ["part-b/train-00040.jsonl"]
    assert rowfiles.local_paths("org/ds", files, "**/*train*.parquet") == [f"data/ds/{f}" for f in files]


def test_unknown_split_keeps_its_fallback_glob():
    assert rowfiles.local_paths("org/ds", [], "**/*train*.parquet") == ["data/ds/**/*train*.parquet"]


@pytest.mark.parametrize("skyrl", [False, True])
def test_training_commands_preserve_the_exact_large_split(monkeypatch, skyrl):
    train = [f"shared/train-{i:05d}.parquet" for i in range(40)] + ["other/train-00040.parquet"]
    validation = ["shared/validation-00000.parquet"]
    monkeypatch.setattr(rowfiles, "files_of", lambda spec, meta, config, split: train if split == "train" else validation)
    ds = Dataset(spec="org/ds", tags=[], card={}, splits=[
        {"config": "subset", "split": "train"}, {"config": "subset", "split": "validation"},
    ])
    script = verl_rows.run_blocks(ds, "subset", "train", skyrl=skyrl, env_class="gsm8k", custom=False, tools=False)[0]["text"]
    args = shlex.split(script.replace("\\\n", ""), comments=True)

    def paths(key):
        value = next(arg.split("=", 1)[1] for arg in args if arg.startswith(key + "="))
        return ast.literal_eval(value) if skyrl else value.strip("[]").split(",")

    train_key, validation_key = ("data.train_data", "data.val_data") if skyrl else ("data.train_files", "data.val_files")
    assert paths(train_key) == [f"data/ds/{f}" for f in train]
    assert paths(validation_key) == [f"data/ds/{f}" for f in validation]
