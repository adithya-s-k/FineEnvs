import pytest
import requests
from image_text_gen.client import encode_image
from image_text_gen.fixtures import render
from image_text_gen.models import VERIFIER_FAILED, ImageTextGenAction
from image_text_gen.server.environment import ImageTextGenEnvironment
from image_text_gen.server.images import prepare
from image_text_gen.server.verifier import (
    DEFAULT_MODELS,
    NO_TEXT,
    Verifier,
    VerifierModel,
    VerifierUnavailable,
    clean_transcription,
    parse_models,
)


def fake_response(content="Keep Calm", status=200, finish="stop"):
    class Response:
        status_code = status
        headers = {}

        def json(self):
            return {
                "choices": [{"finish_reason": finish, "message": {"content": content}}],
                "usage": {"prompt_tokens": 1200, "completion_tokens": 5},
            }

        def close(self):
            pass

    return Response()


@pytest.fixture
def image():
    picture, _ = render("Keep Calm", size=(512, 256))
    return prepare(encode_image(picture))


def test_each_model_reads_blind_and_results_are_cached(monkeypatch, image):
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return fake_response()

    monkeypatch.setattr("image_text_gen.server.verifier.requests.post", post)
    verifier = Verifier(token="test-secret")
    readings = verifier.transcribe(*image)
    assert [r["text"] for r in readings] == ["Keep Calm", "Keep Calm"]
    assert sorted(c["model"] for c in calls) == sorted(m.route for m in DEFAULT_MODELS)
    for payload in calls:
        text = str(payload["messages"])
        assert "Keep Calm" not in text  # the target is never sent to a verifier
        assert payload["temperature"] == 0
    qwen36 = next(c for c in calls if "Qwen3.6" in c["model"])
    gemma = next(c for c in calls if "gemma" in c["model"])
    assert "chat_template_kwargs" not in gemma
    assert qwen36["chat_template_kwargs"] == {"enable_thinking": False}
    assert all(r["cached"] for r in verifier.transcribe(*image)) and len(calls) == 2


@pytest.mark.parametrize(
    "response",
    [fake_response(status=503), fake_response(status=400), fake_response(finish="content_filter"),
     fake_response(content=None)],
)
def test_any_failed_reading_assigns_no_reward_and_keeps_the_episode(monkeypatch, image, response):
    monkeypatch.setattr("image_text_gen.server.verifier.requests.post", lambda *a, **k: response)
    monkeypatch.setattr("image_text_gen.server.verifier.time.sleep", lambda s: None)
    verifier = Verifier(token="test-secret")
    with pytest.raises(VerifierUnavailable, match=VERIFIER_FAILED):
        verifier.transcribe(*image)
    assert not verifier.cache

    from image_text_gen.data.catalog import Catalog, build_tasks

    rows = [{"id": "1", "prompt": 'A sign "Keep Calm".', "text": "Keep Calm"}]
    env = ImageTextGenEnvironment(Catalog(build_tasks([], rows)), verifier)
    env.reset(split="test", index=0)
    picture, _ = render("Keep Calm", size=(512, 256))
    with pytest.raises(VerifierUnavailable):
        env.step(ImageTextGenAction(image=encode_image(picture)))
    assert env.state.step_count == 0


def test_truncated_reading_is_kept_not_failed(monkeypatch, image):
    monkeypatch.setattr(
        "image_text_gen.server.verifier.requests.post",
        lambda *a, **k: fake_response("Keep Calm " + "x" * 100, finish="length"),
    )
    readings = Verifier(token="test-secret").transcribe(*image)
    assert all(r["truncated"] for r in readings)


def test_transient_timeout_retries_once(monkeypatch, image):
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs["json"]["model"])
        if calls.count(kwargs["json"]["model"]) == 1:
            raise requests.Timeout("slow provider")
        return fake_response()

    monkeypatch.setattr("image_text_gen.server.verifier.requests.post", post)
    monkeypatch.setattr("image_text_gen.server.verifier.time.sleep", lambda s: None)
    verifier = Verifier(token="test-secret", concurrency=1)
    assert [r["text"] for r in verifier.transcribe(*image)] == ["Keep Calm", "Keep Calm"]
    assert len(calls) == 4


def test_configuration_is_explicit():
    for provider in ("auto", "fastest", "https://x", "Deep Infra"):
        with pytest.raises(ValueError):
            Verifier([VerifierModel("google/gemma-4-31B-it", provider)], token="t")
    with pytest.raises(ValueError):
        Verifier([DEFAULT_MODELS[0], DEFAULT_MODELS[0]], token="t")
    models = parse_models("Qwen/Qwen3.8-27B:deepinfra, google/gemma-4-31B-it:novita")
    assert models[0].extra == {"chat_template_kwargs": {"enable_thinking": False}}
    assert models[1] == VerifierModel("google/gemma-4-31B-it", "novita")
    a, b = Verifier(token="t"), Verifier(models, token="t")
    assert a.policy_id != b.policy_id


def test_clean_transcription_handles_fences_and_no_text():
    assert clean_transcription("```\nKeep Calm\n```") == "Keep Calm"
    assert clean_transcription(NO_TEXT) == "" and clean_transcription(" no text ") == ""
