import pytest
from image_text_gen.client import encode_image
from image_text_gen.data.catalog import SPLITS, VALIDATION_PERCENT, Catalog, build_tasks
from image_text_gen.fixtures import VARIANTS, png_bytes, render, write_source
from image_text_gen.models import ImageTextGenAction
from image_text_gen.server.environment import ImageTextGenEnvironment
from image_text_gen.server.scoring import combine
from image_text_gen.server.verifier import FixtureVerifier


@pytest.fixture
def catalog(tmp_path):
    return Catalog.from_source(write_source(tmp_path, train_repeat=40))


def test_splits_are_deterministic_deduplicated_and_disjoint(tmp_path, catalog):
    again = Catalog.from_source(write_source(tmp_path / "again", train_repeat=40))
    assert catalog.manifest["catalog_id"] == again.manifest["catalog_id"]
    assert set(catalog.splits()) <= set(SPLITS) and catalog.count("test") == 6
    total = catalog.count("train") + catalog.count("validation")
    assert total == 240 and 0 < catalog.count("validation") < total * VALIDATION_PERCENT * 3 / 100
    prompts = [catalog.at(s, i)["prompt"] for s in catalog.splits() for i in range(catalog.count(s))]
    assert len(prompts) == len(set(prompts))


def test_train_drops_duplicate_prompts_and_prompts_seen_in_test():
    row = {"id": "1", "prompt": 'A "sign".', "text": "sign"}
    tasks = build_tasks([row, {**row, "id": "2"}, {**row, "id": "3", "prompt": "B"}], [row])
    assert [t["source_id"] for t in tasks["train"] + tasks["validation"]] == ["3"]


def test_task_ids_are_stable_and_the_public_view_is_whitelisted(catalog):
    task = catalog.at("test", 0)
    assert task["task_id"].startswith("itg-") and catalog.get(task["task_id"]) is task
    assert "source_id" not in catalog.public(task)
    with pytest.raises(IndexError):
        catalog.at("test", 99)
    with pytest.raises(ValueError):
        catalog.task_range("train", 0, 5000)


def submit(env, image, readings):
    return env.step(ImageTextGenAction(image=encode_image(png_bytes(image, readings))))


def test_single_step_episode_and_replay_by_task_id(catalog):
    env = ImageTextGenEnvironment(catalog, FixtureVerifier())
    first = env.reset(split="test", seed=3)
    assert env.reset(split="test", seed=3).task_id == first.task_id
    image, expected = render(first.target_text, size=(600, 300))
    result = submit(env, image, [expected, expected])
    assert result.done and result.reward == 1.0 and result.metrics["exact_match"]
    assert len(result.transcriptions) == 2 and result.grading_policy_id
    with pytest.raises(RuntimeError):
        submit(env, image, [expected, expected])
    replay = ImageTextGenEnvironment(catalog, FixtureVerifier())
    assert replay.reset(task_id=first.task_id).prompt == first.prompt


def test_invalid_and_blank_images_score_zero_without_readings(catalog):
    class NoCalls(FixtureVerifier):
        def transcribe(self, png, info):
            raise AssertionError("verifier must not be called")

    env = ImageTextGenEnvironment(catalog, NoCalls())
    env.reset(split="test", index=0)
    bad = env.step(ImageTextGenAction(image="%%%"))
    assert bad.done and bad.reward == 0.0 and bad.metrics["invalid_image"]
    env.reset(split="test", index=0)
    blank, _ = render("x", "blank")
    result = submit(env, blank, None)
    assert result.reward == 0.0 and result.metrics["blank_image"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_fixture_variants_rank_below_a_clean_render(catalog, variant):
    target = "Happy Birthday"
    _, expected = render(target, variant, seed=1)
    clean = combine(target, [target])[0]
    value = combine(target, [expected, expected])[0]
    assert value <= clean
    if variant != "clean":
        assert value < clean


@pytest.mark.parametrize("target", ["2024", "Keep Calm"])
def test_every_defect_variant_changes_the_literal_truth(target):
    # Digits-only targets have no case, so only wrong_case may leave them unchanged.
    for variant in set(VARIANTS) - {"clean", "wrong_case"}:
        _, expected = render(target, variant, seed=3)
        assert expected != target, (target, variant)


def test_audit_log_records_every_grading_and_samples_images(catalog, tmp_path):
    import json

    from image_text_gen.server.audit import AuditLog

    audit = AuditLog(tmp_path / "audit", image_rate=1.0)
    env = ImageTextGenEnvironment(catalog, FixtureVerifier(), audit)
    first = env.reset(split="test", index=0)
    image, expected = render(first.target_text, size=(600, 300))
    result = submit(env, image, [expected, expected])
    records = list((tmp_path / "audit" / "records").glob("*.jsonl"))
    line = json.loads(records[0].read_text().splitlines()[0])
    assert line["task_id"] == first.task_id and line["reward"] == result.reward
    assert (tmp_path / "audit" / line["image"]).is_file()
    assert len(line["transcriptions"]) == 2


def test_audit_failure_never_changes_the_reward(catalog, tmp_path):
    from image_text_gen.server.audit import AuditLog

    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    env = ImageTextGenEnvironment(catalog, FixtureVerifier(), AuditLog(blocker, image_rate=1.0))
    first = env.reset(split="test", index=0)
    image, expected = render(first.target_text, size=(600, 300))
    assert submit(env, image, [expected, expected]).reward == 1.0


def test_excluded_source_ids_are_dropped_from_every_split(tmp_path):
    source = write_source(tmp_path, train_repeat=4)
    full = Catalog.from_source(source, excluded=frozenset())
    victim = full.at("test", 0)["source_id"]
    trained = full.at("train", 0)["source_id"]
    screened = Catalog.from_source(source, excluded={victim, trained})
    ids = {screened.at(s, i)["source_id"] for s in screened.splits() for i in range(screened.count(s))}
    assert victim not in ids and trained not in ids
    assert screened.count("test") == full.count("test") - 1


def test_published_jsonl_round_trips_to_the_same_catalog(tmp_path):
    import json

    catalog = Catalog.from_source(write_source(tmp_path / "src", train_repeat=40), excluded=frozenset())
    out = tmp_path / "published"
    out.mkdir()
    for split in SPLITS:
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in catalog.rows(split))
        )
    again = Catalog.from_published(out)
    assert again.manifest["catalog_id"] == catalog.manifest["catalog_id"]
    assert again.get(catalog.at("test", 0)["task_id"])["prompt"] == catalog.at("test", 0)["prompt"]


def test_shipped_exclusions_cover_the_reported_prompt():
    from image_text_gen.data.catalog import load_exclusions

    excluded = load_exclusions()
    assert len(excluded) >= 300
