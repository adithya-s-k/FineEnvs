"""Check the scoring boundary without importing GPU dependencies."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[3] / "train" / "grpo_latex_ocr.py"
spec = importlib.util.spec_from_file_location("grpo_latex_ocr", path)
training = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = training
spec.loader.exec_module(training)


def test_reward_keeps_raw_output_for_length_guard():
    from latex_ocr_env.server.rubric import LatexOCRRubric

    def step(action):
        return SimpleNamespace(
            done=True, reward=LatexOCRRubric().grade(action.latex, "x").reward
        )

    environments = [SimpleNamespace(client=SimpleNamespace(step=step))] * 2
    completions = [
        [{"role": "assistant", "content": "x"}],
        [{"role": "assistant", "content": "x" + " " * 1000}],
    ]
    assert training.env_reward(completions, environments) == [1.0, 0.0]


def test_scoring_mismatched_groups_fails():
    with pytest.raises(ValueError):
        training.env_reward(["x"], [])


def test_invalid_training_sizes_fail_before_gpu_work():
    with pytest.raises(ValueError):
        training.Config(num_generations=1).validate()
