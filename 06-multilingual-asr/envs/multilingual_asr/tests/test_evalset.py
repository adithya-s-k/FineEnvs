from collections import Counter

import pytest
from multilingual_asr.data.evalset import (
    OCR_LANGUAGE_MAP,
    OCR_ONLY,
    allocate,
    load,
    overlap_languages,
    record,
    save,
    select,
    summarize,
)
from multilingual_asr.data.schema import FAMILIES
from multilingual_asr.data.tasks import SAMPLING_RATE

FAMS = list(FAMILIES)


def metadata(languages, per_language=60):
    return {
        language: [
            {
                "id": index,
                "split": "test",
                "num_samples": 5 * SAMPLING_RATE,
                "transcription": f"utterance {index} in {language}",
                "raw_transcription": f"Utterance {index} in {language}.",
                "language": language.upper(),
            }
            for index in range(per_language)
        ]
        for language in languages
    }


def test_allocation_totals_exactly_and_balances_families():
    languages = [f"l{i}" for i in range(21)]
    counts = allocate(languages, FAMS, 504)
    assert sum(counts.values()) == 504
    per_family, per_language = Counter(), Counter()
    for (language, family), n in counts.items():
        per_family[family] += n
        per_language[language] += n
    assert set(per_family.values()) == {168}
    assert set(per_language.values()) == {24}


def test_overlap_is_twentyone_because_fleurs_has_no_sanskrit():
    """The OCR corpus has 22 languages; FLEURS has no Sanskrit, so the overlap is 21."""
    assert len(OCR_LANGUAGE_MAP) == 21
    assert OCR_ONLY == ("sa",)
    assert "sa" not in OCR_LANGUAGE_MAP
    available = list(OCR_LANGUAGE_MAP.values()) + ["zu_za"]
    assert overlap_languages(available) == sorted(
        OCR_LANGUAGE_MAP.values(),
        key=lambda code: [iso for iso, c in OCR_LANGUAGE_MAP.items() if c == code][0],
    )


def test_overlap_refuses_to_pretend_a_mapped_language_exists():
    with pytest.raises(ValueError, match="FLEURS lacks mapped languages"):
        overlap_languages(["en_us", "hi_in"])


def test_selection_is_deterministic():
    data = metadata(["en_us", "hi_in"])
    first = select(data, ["en_us", "hi_in"], FAMS, 24)
    again = select(data, ["en_us", "hi_in"], FAMS, 24)
    assert [e["task_id"] for e in first] == [e["task_id"] for e in again]
    assert len({e["task_id"] for e in first}) == 24


def test_a_larger_set_is_a_superset_of_a_smaller_one_for_shared_languages():
    """The two published sets must agree on every task they share, not resample."""
    languages = ["en_us", "hi_in", "zu_za"]
    data = metadata(languages)
    broad = select(data, languages, FAMS, 9)  # 1 per language/family
    deep = select(data, ["en_us", "hi_in"], FAMS, 24)  # 4 per language/family
    shared = {e["task_id"] for e in broad if e["language"] in {"en_us", "hi_in"}}
    assert shared and shared <= {e["task_id"] for e in deep}


def test_families_draw_disjoint_utterances_within_a_language():
    """One bad recording must not dominate a language across every family."""
    data = metadata(["en_us"])
    selection = select(data, ["en_us"], FAMS, 9)
    per_family = {}
    for entry in selection:
        per_family.setdefault(entry["family"], set()).add(entry["sample_id"])
    picked = [ids for ids in per_family.values()]
    assert len(set.union(*picked)) == sum(len(ids) for ids in picked)


def test_selection_rejects_a_language_without_enough_eligible_audio():
    data = metadata(["en_us"], per_language=2)
    with pytest.raises(ValueError, match="eligible utterances, need"):
        select(data, ["en_us"], FAMS, 30)


def test_utterances_outside_the_duration_policy_are_not_selected():
    data = metadata(["en_us"])
    for row in data["en_us"][:50]:
        row["num_samples"] = 8  # far below the policy floor
    with pytest.raises(ValueError, match="eligible utterances, need"):
        select(data, ["en_us"], FAMS, 45)


def test_frozen_set_round_trips_and_rejects_tampering(tmp_path):
    data = metadata(["en_us", "hi_in"])
    body = record(
        select(data, ["en_us", "hi_in"], FAMS, 24),
        name="t",
        languages=["en_us", "hi_in"],
        families=FAMS,
        split="test",
        size=24,
        seed=42,
        validated=False,
    )
    path = save(body, tmp_path / "e.json")
    assert load(path)["evalset_id"] == body["evalset_id"]
    assert summarize(body)["languages"] == 2

    import json

    edited = json.loads(path.read_text())
    edited["tasks"] = edited["tasks"][::-1]
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="identity does not match"):
        load(path)


def test_recorded_verification_failures_block_use(tmp_path):
    data = metadata(["en_us"])
    body = record(
        select(data, ["en_us"], FAMS, 9),
        name="t",
        languages=["en_us"],
        families=FAMS,
        split="test",
        size=9,
        seed=42,
        validated=False,
        failures=[{"language": "en_us", "sample_id": 0, "error": "bad audio"}],
    )
    path = save(body, tmp_path / "e.json")
    with pytest.raises(ValueError, match="did not verify"):
        load(path)


def test_a_frozen_task_id_matches_the_one_prepare_derives():
    """The set names task IDs a snapshot must contain; both sides must derive them alike.

    A first version passed the repository id where the revision belongs, so every ID in
    the set differed from the one prepare wrote, and a correctly built snapshot reported
    504 of 504 tasks missing while holding exactly the right audio.
    """
    from multilingual_asr.data.schema import DEFAULT_REVISION
    from multilingual_asr.data.tasks import derive_tasks

    data = metadata(["en_us"])
    by_id = {row["id"]: row for row in data["en_us"]}
    frozen = select(data, ["en_us"], FAMS, 9)
    assert frozen
    for entry in frozen:
        row = dict(by_id[entry["sample_id"]], audio={"bytes": b"x"}, split="test")
        derived, _ = derive_tasks(row, "en_us", DEFAULT_REVISION)
        assert entry["task_id"] in {t["task_id"] for t in derived}, (
            f"{entry['family']} id differs between the frozen set and prepare"
        )


def test_the_frozen_set_records_the_revision_its_ids_were_derived_from():
    from multilingual_asr.data.schema import DEFAULT_REVISION

    data = metadata(["en_us"])
    body = record(
        select(data, ["en_us"], FAMS, 9),
        name="t",
        languages=["en_us"],
        families=FAMS,
        split="test",
        size=9,
        seed=42,
        validated=False,
    )
    assert body["revision"] == DEFAULT_REVISION
    # A set built under another revision is a different set, not the same one relabelled.
    other = record(
        select(data, ["en_us"], FAMS, 9, revision="other"),
        name="t",
        languages=["en_us"],
        families=FAMS,
        split="test",
        size=9,
        seed=42,
        validated=False,
        revision="other",
    )
    assert other["evalset_id"] != body["evalset_id"]
    assert {e["task_id"] for e in other["tasks"]} != {
        e["task_id"] for e in body["tasks"]
    }


def test_every_fleurs_language_code_round_trips_through_a_task_id():
    """es_419 is a real FLEURS code; a region segment is not always alphabetic.

    A stricter pattern rejected it, and the eval sets then looked as though 24 of their
    tasks were missing from a corpus that held every one of them.
    """
    import json
    import pathlib

    from multilingual_asr.data.schema import (
        DEFAULT_REVISION,
        language_code,
        parse_task_id,
        task_id,
    )

    manifest = pathlib.Path(__file__).resolve().parents[3] / "data/corpus-manifest.json"
    languages = json.loads(manifest.read_text())["config"]["languages"]
    assert len(languages) == 102 and "es_419" in languages
    for language in languages:
        assert language_code(language) == language
        for split in ("train", "validation", "test"):
            identifier = task_id(DEFAULT_REVISION, language, split, 7, "transcription")
            assert parse_task_id(identifier) == (language, split)
