from __future__ import annotations

import gzip
import importlib.util
import json
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


def test_without_a_source_it_says_what_to_set(monkeypatch, tmp_path):
    monkeypatch.delenv("RETROENV_BENCHMARK_DIR", raising=False)
    monkeypatch.delenv("RETROENV_TASKS_REPO", raising=False)
    monkeypatch.delenv("RETROENV_CORPUS_MANIFEST", raising=False)
    monkeypatch.setattr(prepare, "DEFAULT_MANIFEST", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="RETROENV_TASKS_REPO"):
        prepare.main()


def _fake_bucket(root: Path) -> tuple[Path, dict]:
    """A bucket directory laid out as publish_bucket writes it, and its manifest."""
    from retroenv.evalsets import make_evalset, select_tiered
    from retroenv.serving import build_index
    from retroenv.store import TaskStore

    bucket = root / "bucket"
    bucket.mkdir()
    for name in ("manifest.json", "checksums.json", "library", "stocks"):
        source = FIXTURE / name
        (shutil.copytree if source.is_dir() else shutil.copy2)(source, bucket / name)
    prefix = "openenv/indexes/snap"
    build_index(FIXTURE, bucket / prefix)
    store = TaskStore(FIXTURE / "tasks-private", FIXTURE / "stocks")
    chosen = select_tiered([t for t in store.tasks("test_hard") if t.variant == "standard"], {"hard": 2}, "s")
    record = make_evalset("final_eval", chosen, seed="s", design={})
    (bucket / "openenv/evalsets").mkdir(parents=True)
    (bucket / "openenv/evalsets/final_eval.json").write_text(json.dumps(record))

    def entry(path, local):
        file = bucket / path
        return {"path": path, "local": local, "size": file.stat().st_size, "sha256": prepare.sha256_file(file)}

    files = [
        entry(str(p.relative_to(bucket)), str(p.relative_to(bucket)))
        for p in sorted(bucket.rglob("*"))
        if p.is_file() and not str(p.relative_to(bucket)).startswith("openenv")
    ]
    # The index is stored gzipped, as publish_bucket stores it; prepare unpacks and checks both.
    index = bucket / prefix / "serving.sqlite"
    with index.open("rb") as source, gzip.open(index.with_suffix(".sqlite.gz"), "wb") as sink:
        shutil.copyfileobj(source, sink)
    unpacked = {"size": index.stat().st_size, "sha256": prepare.sha256_file(index)}
    index.unlink()
    files.append({**entry(f"{prefix}/serving.sqlite.gz", "serving/serving.sqlite"), "unpacked": unpacked})
    manifest = {
        "status": "ready",
        "storage": "bucket-sqlite",
        "snapshot_id": "snap",
        "bucket_id": "FineEnvs/fixture-bucket",
        "files": files,
        "evalsets": {"final_eval": entry("openenv/evalsets/final_eval.json", "evalsets/final_eval.json")},
    }
    path = root / "corpus-manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def _bucket_env(monkeypatch, manifest: Path, prepared: Path, mount: Path | None = None) -> None:
    for name in ("RETROENV_BENCHMARK_DIR", "RETROENV_TASKS_REPO", "RETROENV_BUCKET_ID", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RETROENV_CORPUS_MANIFEST", str(manifest))
    monkeypatch.setenv("RETROENV_PREPARED_DIR", str(prepared))
    if mount is None:
        monkeypatch.delenv("RETROENV_BUCKET_ROOT", raising=False)
    else:
        monkeypatch.setenv("RETROENV_BUCKET_ROOT", str(mount))


def _fake_bucket_http(monkeypatch, bucket: Path) -> list[str]:
    downloaded: list[str] = []

    class FakeApi:
        def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files, token):
            assert bucket_id == "FineEnvs/fixture-bucket" and raise_on_missing_files and token is None
            for remote, local in files:
                downloaded.append(remote)
                shutil.copyfile(bucket / remote, local)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    return downloaded


def test_a_mounted_bucket_prepares_a_servable_directory(tmp_path, monkeypatch, capsys):
    from retroenv.benchmark import Benchmark

    manifest, _ = _fake_bucket(tmp_path)
    prepared = tmp_path / "prepared"
    _bucket_env(monkeypatch, manifest, prepared, mount=tmp_path / "bucket")
    downloaded = _fake_bucket_http(monkeypatch, tmp_path / "bucket")
    assert prepare.main() == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(prepared)
    assert downloaded == []  # everything came from the mount
    benchmark = Benchmark.load(prepared)
    assert len(benchmark.store.tasks("final_eval")) == 2
    assert not (prepared / "tasks-private").exists()  # tasks are served from the index


def test_without_a_mount_it_downloads_once_then_reuses(tmp_path, monkeypatch):
    manifest, data = _fake_bucket(tmp_path)
    prepared = tmp_path / "prepared"
    _bucket_env(monkeypatch, manifest, prepared)
    downloaded = _fake_bucket_http(monkeypatch, tmp_path / "bucket")
    assert prepare.main() == 0
    assert sorted(downloaded) == sorted(e["path"] for e in prepare._entries(data))
    downloaded.clear()
    assert prepare.main() == 0
    assert downloaded == []
    assert not list(prepared.rglob("*.partial"))


def test_a_corrupted_bucket_file_stops_the_server(tmp_path, monkeypatch):
    manifest, _ = _fake_bucket(tmp_path)
    (tmp_path / "bucket/library/reagents.json").write_text("[]")
    prepared = tmp_path / "prepared"
    _bucket_env(monkeypatch, manifest, prepared)
    _fake_bucket_http(monkeypatch, tmp_path / "bucket")
    with pytest.raises(SystemExit, match="does not match its manifest digest"):
        prepare.main()
    assert not list(prepared.rglob("*.partial"))


def test_a_packed_file_that_unpacks_wrong_stops_the_server(tmp_path, monkeypatch):
    manifest, data = _fake_bucket(tmp_path)
    packed = next(e for e in data["files"] if e.get("unpacked"))
    packed["unpacked"]["sha256"] = "0" * 64  # the gzip is intact but expands to something else
    manifest.write_text(json.dumps(data))
    prepared = tmp_path / "prepared"
    _bucket_env(monkeypatch, manifest, prepared, mount=tmp_path / "bucket")
    _fake_bucket_http(monkeypatch, tmp_path / "bucket")
    with pytest.raises(SystemExit, match="serving.sqlite.gz"):
        prepare.main()
    assert not list(prepared.rglob("*.partial")) and not (prepared / "serving/serving.sqlite").exists()


def test_an_evalset_dropped_from_the_snapshot_is_no_longer_served(tmp_path, monkeypatch):
    manifest, _ = _fake_bucket(tmp_path)
    prepared = tmp_path / "prepared"
    (prepared / "evalsets").mkdir(parents=True)
    (prepared / "evalsets" / "retired.json").write_text("{}")
    _bucket_env(monkeypatch, manifest, prepared, mount=tmp_path / "bucket")
    _fake_bucket_http(monkeypatch, tmp_path / "bucket")
    assert prepare.main() == 0
    assert sorted(p.name for p in (prepared / "evalsets").glob("*.json")) == ["final_eval.json"]
