import pytest
from multilingual_asr.data.schema import (
    CHARACTER_SCORED,
    character_scored,
    normalize_for_scoring,
    score_tokens,
)
from multilingual_asr.server.rewards import error_rate, score


def task(language, family, reference):
    return {"language": language, "family": family, "reference": reference}


def test_word_languages_are_scored_per_word():
    rate, unit = error_rate(
        "the quick brown cat", "the quick brown fox", "en_us", "transcription"
    )
    assert unit == "wer"
    assert rate == pytest.approx(1 / 4)


def test_spaceless_scripts_are_scored_per_character_not_per_word():
    """Word rate on an unspaced script is degenerate: one token, so any error scores 1."""
    reference, prediction = "这是一个句子", "这是一个团子"
    rate, unit = error_rate(prediction, reference, "cmn_hans_cn", "transcription")
    assert unit == "cer"
    assert rate == pytest.approx(1 / 6)
    # The same pair judged as words would be a total miss, which is the bug being avoided.
    assert error_rate(prediction, reference, "en_us", "transcription")[0] == 1.0


def test_every_character_scored_language_is_a_real_fleurs_code():
    from multilingual_asr.data.schema import language_code

    for language in CHARACTER_SCORED:
        assert language_code(language) == language
        assert character_scored(language)
    assert not character_scored("en_us")


def test_normalization_differs_by_family_because_the_targets_differ():
    raw = "Hello, World!"
    # The normalized family targets FLEURS' stripped field: case and punctuation are noise.
    assert normalize_for_scoring(raw, "transcription") == "hello world"
    # The verbatim family is graded on exactly those, so they must survive.
    assert normalize_for_scoring(raw, "verbatim_transcription") == "Hello, World!"


def test_verbatim_punctuation_is_an_error_but_normalized_punctuation_is_not():
    spoken = "Hello, world."
    without = "Hello world"
    assert score(task("en_us", "verbatim_transcription", spoken), without)[0] < 1.0
    assert score(task("en_us", "transcription", spoken), without)[0] == 1.0


def test_exact_answer_scores_one_and_empty_scores_zero():
    for language, reference in (("en_us", "a short sentence"), ("ja_jp", "短い文")):
        good, metrics = score(task(language, "transcription", reference), reference)
        assert good == 1.0 and metrics["exact_match"] is True
        empty, _ = score(task(language, "transcription", reference), "")
        assert empty == 0.0


def test_reward_is_clamped_when_a_model_pads_beyond_the_reference():
    reference = "two words"
    padded = reference + " " + " ".join(["filler"] * 50)
    reward, metrics = score(task("en_us", "transcription", reference), padded)
    # Insertions push the rate above 1; the reward floors at zero rather than going negative.
    assert metrics["wer"] > 1.0
    assert reward == 0.0


def test_metrics_are_numeric_and_keyed_by_the_unit_measured():
    """A trainer averages these, so a string in the dict would break the observation."""
    _, word = score(task("en_us", "transcription", "one two"), "one three")
    _, char = score(task("th_th", "transcription", "สวัสดี"), "สวัสดีครับ")
    assert "wer" in word and "cer" not in word
    assert "cer" in char and "wer" not in char
    for metrics in (word, char):
        assert all(isinstance(v, (int, float, bool)) for v in metrics.values())


def test_an_unscorable_reference_is_rejected_rather_than_scored():
    with pytest.raises(ValueError, match="empty after normalization"):
        error_rate("anything", "!!!", "en_us", "transcription")


def test_language_id_is_matched_exactly_and_not_normalized_as_text():
    good, metrics = score(task("hi_in", "language_id", "hi_in"), " HI_IN ")
    assert good == 1.0 and metrics["exact_match"] is True
    assert score(task("hi_in", "language_id", "hi_in"), "hi")[0] == 0.0


def test_score_tokens_drops_whitespace_only_for_character_scored_scripts():
    assert score_tokens("a b", "en_us") == ["a", "b"]
    assert score_tokens("我 们", "cmn_hans_cn") == ["我", "们"]
