import pytest
from nayana_ocr.data.catalog import Catalog
from nayana_ocr.fixtures import make_fixture
from nayana_ocr.models import NayanaAction
from nayana_ocr.server.environment import NayanaEnvironment
from nayana_ocr.server.rewards import score


@pytest.mark.parametrize(
    "text", ["ಕನ್ನಡ ಭಾಷೆ", "हिन्दी", "العَرَبِيَّة", "日本語", "é", "a\u200db"]
)
def test_unicode_exact_and_padding_guard(text):
    assert score("section_ocr", text, text)[0] == 1.0
    assert score("section_ocr", "", text)[0] == 0.0
    assert score("section_ocr", text + " " * 5000, text)[0] == 0.0


def test_normalization_preserves_script_distinctions():
    assert score("section_ocr", "e\u0301", "é")[0] == 1.0
    assert score("section_ocr", "क", "की")[0] < 1.0
    assert score("section_ocr", "عربية", "عَرَبِيَّة")[0] < 1.0
    assert score("section_ocr", "A", "a")[0] < 1.0
    assert score("section_ocr", " a\n b ", "a b")[0] == 1.0


def test_mcq_requires_one_unambiguous_letter():
    assert score("mcq_vqa", " B\n", "B")[0] == 1.0
    for prediction in ("A or B", "answer B", "b", "", "AB"):
        assert score("mcq_vqa", prediction, "B")[0] == 0.0


def test_same_task_independent_episodes_and_failed_reset_invalidates_state(tmp_path):
    make_fixture(tmp_path)
    catalog = Catalog(tmp_path)
    a, b = NayanaEnvironment(catalog), NayanaEnvironment(catalog)
    task = catalog.at("train", 0)
    first = a.reset(task_id=task["task_id"])
    second = b.reset(task_id=task["task_id"])
    assert first.model_dump() == second.model_dump()
    assert a.state.episode_id != b.state.episode_id
    assert a.step(NayanaAction(answer="")).reward == 0.0
    result = b.step(NayanaAction(answer=task["reference"]))
    assert result.reward == 1.0 and "reference" not in result.model_dump()
    with pytest.raises(RuntimeError, match="Reset"):
        b.step(NayanaAction(answer=task["reference"]))
    a.reset(task_id=task["task_id"])
    with pytest.raises(KeyError):
        a.reset(task_id="unknown")
    with pytest.raises(RuntimeError, match="Reset"):
        a.step(NayanaAction(answer=""))
