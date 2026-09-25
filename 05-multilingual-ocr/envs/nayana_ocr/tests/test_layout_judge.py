import copy
import json

import pytest
import requests
from nayana_ocr.data.schema import REVISION
from nayana_ocr.data.tasks import derive_tasks
from nayana_ocr.fixtures import fixture_rows
from nayana_ocr.models import NayanaAction
from nayana_ocr.server.environment import NayanaEnvironment
from nayana_ocr.server.judge import CHECKS, GemmaJudge, JudgeUnavailable
from nayana_ocr.server.layout import matched_count, score_layout

REFERENCE = '[{"label":"text","bbox":[10,10,110,110]}]'


def test_layout_penalizes_duplicates_missing_regions_and_wrong_classes():
    assert score_layout(REFERENCE, REFERENCE, 200, 200)[0] == 1
    assert score_layout("[]", REFERENCE)[0] == 0
    duplicate = json.dumps(json.loads(REFERENCE) * 2)
    assert score_layout(duplicate, REFERENCE)[0] == pytest.approx(2 / 3)
    assert score_layout(REFERENCE.replace('"text"', '"title"'), REFERENCE)[0] == 0
    shifted = REFERENCE.replace("10,10,110,110", "30,10,130,110")
    assert 0 < score_layout(shifted, REFERENCE)[0] < 1
    # The matcher must reassign a previous edge when that increases true positives.
    assert matched_count([[0.9, 0.8], [0.7, 0.0]], 0.5) == 2


@pytest.mark.parametrize(
    "prediction",
    [
        "```json\n[]\n```",
        '[{"label":"text","bbox":[0,0,NaN,2]}]',
        '[{"label":"text","bbox":[false,0,10,20]}]',
        '[{"label":"text","bbox":[0,0,300,300]}]',
        '[{"label":"text","label":"title","bbox":[0,0,10,20]}]',
        '[{"label":"text","bbox":[0,0,10,20],"score":1}]',
        "{" * 10000,
    ],
)
def test_layout_rejects_malformed_or_unbounded_output(prediction):
    reward, metrics = score_layout(prediction, REFERENCE, 200, 200)
    assert reward == 0 and metrics["valid_format"] is False


def test_new_tasks_keep_nontext_regions_and_original_question_indices():
    row = next(fixture_rows("kn"))
    row["regions.json"].append(
        {
            "region_id": 18,
            "layout_type": "image",
            "bbox": {"xmin": 1, "ymin": 90, "xmax": 100, "ymax": 140},
        }
    )
    row["vqa.json"]["questions"].append(
        {"type": "descriptive", "question": "ಇದು ಏನು?", "answer": "ಇನ್ವಾಯ್ಸ್"}
    )
    tasks, excluded = derive_tasks(row, "kn", REVISION)
    layout = next(t for t in tasks if t["family"] == "layout_detection")
    vqa = next(t for t in tasks if t["family"] == "descriptive_vqa")
    assert len(json.loads(layout["reference"])) == 2
    assert layout["media"] == vqa["media"] == row["jpg"]["bytes"]
    assert vqa["unit"] == "1" and vqa["reference"] == "ಇನ್ವಾಯ್ಸ್"
    assert not any(t["family"] == "page_ocr" for t in tasks)
    assert excluded["page_ocr_incomplete_annotations"] == 1
    metadata, _ = derive_tasks(
        {k: v for k, v in row.items() if k != "jpg"}, "kn", REVISION, metadata_only=True
    )
    assert [(t["family"], t["unit"], t["reference"]) for t in metadata] == [
        (t["family"], t["unit"], t["reference"]) for t in tasks
    ]
    row["regions.json"][-1]["layout_type"] = "unsupported"
    tasks, excluded = derive_tasks(row, "kn", REVISION)
    assert not any(t["family"] == "layout_detection" for t in tasks)


def judge_task():
    return {
        "task_id": "fixture",
        "language": "en",
        "prompt": "What is the total?",
        "reference": "$42",
    }


def fake_response(verdict=None, status=200, finish="stop"):
    class Response:
        status_code = status

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": finish,
                        "message": {"content": json.dumps(verdict)},
                    }
                ]
            }

    return Response()


def test_judge_all_checks_required_cached_and_candidate_is_quoted(monkeypatch):
    calls = []
    verdict = {k: True for k in CHECKS}

    def post(url, **kwargs):
        calls.append(kwargs)
        return fake_response(verdict)

    monkeypatch.setattr("nayana_ocr.server.judge.requests.post", post)
    judge = GemmaJudge(token="test-secret")
    task = judge_task()
    assert judge.score(task, "$42")[0] == 1
    assert judge.score(task, "$42")[0] == 1 and len(calls) == 1
    for check in CHECKS:
        verdict = {k: k != check for k in CHECKS}
        assert judge.score(task, "$42 " + check)[0] == 0
    record = json.loads(calls[0]["json"]["messages"][1]["content"])
    assert record["candidate_answer"] == "$42" and record["reference_answer"] == "$42"
    changed = copy.deepcopy(task)
    changed["reference"] = "$43"
    judge.score(changed, "$42")
    assert (
        len(calls) == 8
    )  # The reference and policy participate in the cache identity.
    assert judge.score(task, " ")[0] == 0


@pytest.mark.parametrize(
    "response",
    [
        fake_response(status=503),
        fake_response({"correct": True}),
        fake_response({k: "true" for k in CHECKS}),
        fake_response({k: True for k in CHECKS}, finish="length"),
    ],
)
def test_judge_failures_never_consume_episode_or_assign_reward(monkeypatch, response):
    monkeypatch.setattr(
        "nayana_ocr.server.judge.requests.post", lambda *a, **kw: response
    )
    judge = GemmaJudge(token="test-secret")
    env = NayanaEnvironment(catalog=object(), judge=judge)
    env._task = {**judge_task(), "family": "descriptive_vqa"}
    with pytest.raises(JudgeUnavailable):
        env.step(NayanaAction(answer="$42"))
    assert env.state.step_count == 0 and not judge.cache


def test_judge_requires_explicit_provider_and_uses_only_hf_router():
    judge = GemmaJudge(token="test-secret")
    assert judge.url == "https://router.huggingface.co/v1/chat/completions"
    for provider in ("auto", "fastest", "https://example.com", "deepinfra:other"):
        with pytest.raises(ValueError):
            GemmaJudge(provider=provider, token="test-secret")


def test_transient_provider_timeout_retries_once_but_a_rejection_does_not(monkeypatch):
    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise requests.Timeout("provider timed out")
        return fake_response({k: False for k in CHECKS})

    monkeypatch.setattr("nayana_ocr.server.judge.requests.post", post)
    monkeypatch.setattr("nayana_ocr.server.judge.time.sleep", lambda seconds: None)
    judge = GemmaJudge(token="test-secret")
    assert judge.score(judge_task(), "$43")[0] == 0
    assert len(calls) == 2 and calls[0] == calls[1]
