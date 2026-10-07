from __future__ import annotations

import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mini-release"
_SPEC = importlib.util.spec_from_file_location("retroenv_prepare", ROOT / "envs/retro_route/openenv/prepare.py")
assert _SPEC and _SPEC.loader
prepare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare)


def _fake_hub(monkeypatch, root: Path) -> list[dict]:
    """A snapshot_download that copies a local benchmark into the requested layout, and records each call."""
    calls: list[dict] = []

    def snapshot_download(repo_id, *, repo_type, revision, local_dir, allow_patterns, token):
        calls.append({"repo": repo_id, "revision": revision, "patterns": allow_patterns, "token": token})
        prefix = allow_patterns[0].split("tasks-private")[0]
        for name in ("tasks-private", "stocks", "library"):
            shutil.copytree(root / name, Path(local_dir) / prefix / name)
        shutil.copy(root / "checksums.json", Path(local_dir) / prefix / "checksums.json")
        return str(local_dir)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return calls


def test_a_public_repo_downloads_without_a_token_and_prints_its_directory(tmp_path, monkeypatch, capsys):
    calls = _fake_hub(monkeypatch, FIXTURE)
    monkeypatch.delenv("RETROENV_BENCHMARK_DIR", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("RETROENV_TASKS_REPO", "LiteFold/RetroEnv@main")
    monkeypatch.setenv("RETROENV_PREPARED_DIR", str(tmp_path))
    assert prepare.main() == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(tmp_path)
    assert calls == [
        {
            "repo": "LiteFold/RetroEnv",
            "revision": "main",
            "token": None,
            "patterns": ["tasks-private/*", "stocks/*", "library/*", "checksums.json", "manifest.json"],
        }
    ]


def test_a_subdirectory_serves_another_benchmark_from_the_same_repo(tmp_path, monkeypatch, capsys):
    calls = _fake_hub(monkeypatch, FIXTURE)
    monkeypatch.delenv("RETROENV_BENCHMARK_DIR", raising=False)
    monkeypatch.setenv("RETROENV_TASKS_REPO", "LiteFold/RetroEnv")
    monkeypatch.setenv("RETROENV_TASKS_SUBDIR", "previous-release")
    monkeypatch.setenv("RETROENV_PREPARED_DIR", str(tmp_path))
    assert prepare.main() == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(tmp_path / "previous-release")
    assert all(pattern.startswith("previous-release/") for pattern in calls[0]["patterns"])


def test_without_a_source_it_says_what_to_set(monkeypatch):
    monkeypatch.delenv("RETROENV_BENCHMARK_DIR", raising=False)
    monkeypatch.delenv("RETROENV_TASKS_REPO", raising=False)
    with pytest.raises(SystemExit, match="RETROENV_TASKS_REPO"):
        prepare.main()
