import hashlib
from collections import Counter

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
    import soundfile

    catalog, _ = snapshot
    task = catalog.at("test", 0)
    path, mime = catalog.asset(task["asset_sha256"])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == task["asset_sha256"]
    assert mime == "audio/wav"
    # Verified with the reader the environment uses, so a subtype it cannot decode
    # cannot pass here either.
    info = soundfile.info(str(path))
    assert info.samplerate == task["sampling_rate"] == 16_000
    assert info.channels == 1
    assert info.frames == task["num_samples"]


def test_tasks_outside_the_duration_policy_are_excluded_not_truncated(tmp_path):
    from multilingual_asr.data.prepare import write_snapshot

    def rows():
        yield {
            "id": 1,
            "path": "/cache/too-short.wav",
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


def test_two_recordings_of_one_sentence_are_two_tasks(tmp_path):
    """FLEURS' id is a sentence id: several speakers read the same sentence under it.

    hi_in test holds 418 recordings under 265 ids. Keying identity on the sentence id
    collapsed distinct clips into one and made two different recordings share a task id.
    """
    from multilingual_asr.data.prepare import write_snapshot
    from multilingual_asr.fixtures import tone

    def rows():
        for speaker in ("a", "b"):
            yield {
                "id": 7,  # the same sentence
                "path": f"/cache/{speaker}.wav",  # different recordings
                "split": "test",
                "num_samples": 16_000,
                "audio": {"bytes": tone(1.0, 220)},
                "transcription": "one sentence",
                "raw_transcription": "One sentence.",
                "language": "English",
            }

    manifest = write_snapshot(
        tmp_path / "s",
        [("en_us", rows())],
        source="synthetic-fixture",
        revision="r",
        languages=["en_us"],
    )
    catalog = Catalog(tmp_path / "s")
    tasks = catalog.task_range("test", 0, catalog.count("test"))
    per_family = Counter(t["family"] for t in tasks)
    # Both recordings survive, each with its own task per family.
    assert set(per_family.values()) == {2}, per_family
    assert len({t["task_id"] for t in tasks}) == len(tasks)
    assert {t["sample_id"] for t in tasks} == {7}
    assert len({t["recording"] for t in tasks}) == 2
    assert sum(c["tasks"] for c in manifest["counts"]) == len(tasks)


def test_group_range_addresses_tasks_without_listing_the_split(corpus_index):
    """Selecting a few tasks per language must not require paging the whole split."""
    catalog, _ = corpus_index
    language, family = "hi_in", "transcription"
    count = catalog.group_count("train", language, family)
    assert count

    rows = catalog.group_range("train", language, family, [0, count - 1])
    assert [row["language"] for row in rows] == [language] * 2
    assert [row["family"] for row in rows] == [family] * 2
    assert rows[0]["task_id"] != rows[1]["task_id"]
    # Discovery withholds the reference here exactly as the split listing does.
    assert all("reference" not in row for row in rows)

    # The same positions addressed one at a time give the same tasks.
    assert rows == [
        catalog.public(catalog.group_at("train", language, family, position))
        for position in (0, count - 1)
    ]

    with pytest.raises(IndexError):
        catalog.group_range("train", language, family, list(range(1001)))


def test_sampling_is_balanced_and_reproducible_without_listing(monkeypatch):
    """A seed must reproduce the draw, and no group may be listed to sample it."""
    from contextlib import contextmanager

    from multilingual_asr import training

    listed = []

    class FakeClient:
        def num_group_tasks(self, split, language, family):
            return 0 if language == "xx_xx" else 50

        def get_group_tasks(self, split, language, family, positions):
            listed.append((language, family, tuple(positions)))
            return [
                {
                    "task_id": f"{language}.{family}.{p}",
                    "language": language,
                    "family": family,
                }
                for p in positions
            ]

    @contextmanager
    def fake_connect(url):
        yield FakeClient()

    monkeypatch.setattr(training, "connect", fake_connect)
    languages, families = ["en_us", "hi_in"], ["transcription", "language_id"]
    rows = training.sampled_rows("http://x", "train", languages, families, 42, 3)

    assert len(rows) == len(languages) * len(families) * 3
    counts = Counter((row["language"], row["family"]) for row in rows)
    assert set(counts) == {(lang, fam) for lang in languages for fam in families}
    assert set(counts.values()) == {3}
    # Only the drawn positions are fetched, never the 50 in each group.
    assert all(len(positions) == 3 for _, _, positions in listed)

    again = training.sampled_rows("http://x", "train", languages, families, 42, 3)
    assert [row["task_id"] for row in rows] == [row["task_id"] for row in again]
    other = training.sampled_rows("http://x", "train", languages, families, 7, 3)
    assert [row["task_id"] for row in rows] != [row["task_id"] for row in other]

    with pytest.raises(ValueError, match="Missing language/task groups"):
        training.sampled_rows("http://x", "train", ["xx_xx"], ["transcription"], 42, 3)
