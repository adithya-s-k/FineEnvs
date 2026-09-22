import hashlib

import pytest
from multilingual_asr.data.catalog import Catalog
from multilingual_asr.data.schema import FAMILIES, SPLITS
from multilingual_asr.fixtures import make_fixture
from multilingual_asr.models import AsrAction
from multilingual_asr.server.environment import AsrEnvironment


@pytest.fixture
def snapshot(tmp_path):
    manifest = make_fixture(tmp_path / "snap")
    return Catalog(tmp_path / "snap"), manifest


def test_snapshot_covers_every_language_family_and_split(snapshot):
    catalog, manifest = snapshot
    assert manifest["status"] == "ready"
    assert manifest["source_license"] == "synthetic-fixture"
    groups = {(c["language"], c["family"]) for c in manifest["counts"]}
    assert groups == {(lang, fam) for lang in catalog.languages for fam in FAMILIES}
    assert all(catalog.count(split) for split in SPLITS)


def test_discovery_withholds_the_reference_and_the_audio(snapshot):
    catalog, _ = snapshot
    rows = catalog.task_range("test", 0, catalog.count("test"))
    assert rows and all("reference" not in row for row in rows)
    assert all(row["asset_path"].startswith("/assets/") for row in rows)
    # Positions are stable and addressable.
    assert catalog.at("test", 0)["task_id"] == rows[0]["task_id"]


def test_reset_and_step_grade_once_and_then_refuse(snapshot):
    catalog, _ = snapshot
    env = AsrEnvironment(catalog)
    task = catalog.at("test", 0)
    observation = env.reset(task_id=task["task_id"])
    assert observation.task_id == task["task_id"]
    assert observation.duration_seconds > 0 and observation.sampling_rate == 16_000
    result = env.step(AsrAction(transcript=task["reference"]))
    assert result.done and result.reward == 1.0
    # The episode is consumed; a second answer must not be graded against it.
    with pytest.raises(ValueError, match="Call reset before step"):
        env.step(AsrAction(transcript=task["reference"]))


def test_a_failed_reset_cannot_leave_a_previous_task_gradable(snapshot):
    catalog, _ = snapshot
    env = AsrEnvironment(catalog)
    env.reset(task_id=catalog.at("test", 0)["task_id"])
    with pytest.raises(KeyError):
        env.reset(task_id="fleurs-does-not-exist")
    with pytest.raises(ValueError, match="Call reset before step"):
        env.step(AsrAction(transcript="anything"))


def test_audio_assets_are_content_addressed_and_playable(snapshot):
    import wave

    catalog, _ = snapshot
    task = catalog.at("test", 0)
    path, mime = catalog.asset(task["asset_sha256"])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == task["asset_sha256"]
    assert mime == "audio/wav"
    with wave.open(str(path)) as handle:
        assert handle.getframerate() == 16_000
        assert handle.getnframes() == task["num_samples"]


def test_tasks_outside_the_duration_policy_are_excluded_not_truncated(tmp_path):
    from multilingual_asr.data.prepare import write_snapshot

    def rows():
        yield {
            "id": 1,
            "split": "test",
            "num_samples": 16,  # 0.001s: far below policy
            "audio": {"bytes": b"RIFF"},
            "transcription": "x",
            "raw_transcription": "X.",
            "language": "English",
        }

    manifest = write_snapshot(
        tmp_path / "s",
        [("en_us", rows())],
        source="synthetic-fixture",
        revision="r",
        languages=["en_us"],
    )
    assert manifest["counts"] == []
    assert manifest["skipped"]["duration_outside_policy"] == 1
