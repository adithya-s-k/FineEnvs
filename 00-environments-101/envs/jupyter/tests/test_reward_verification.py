"""Regression tests for Jupyter reward verification and anti-reward-hacking fixes."""

import sys
import types
from pathlib import Path

JUPYTER_ROOT = Path(__file__).resolve().parents[1]
if str(JUPYTER_ROOT) not in sys.path:
    sys.path.insert(0, str(JUPYTER_ROOT))

# Allow running unit tests without e2b_code_interpreter installed
sys.modules.setdefault("e2b_code_interpreter", types.SimpleNamespace(Sandbox=object))

from core.tasks import matches_expected
from gem.env import JupyterGemEnv
from skyrl_gym.env import JupyterSkyRLEnv


def test_matches_expected_rejects_substring_false_positives():
    # Single-char / short expected outputs must not match inside unrelated words/numbers
    assert not matches_expected("y", "Defaulting to user installation because normal site-packages is not writeable")
    assert not matches_expected("y", "python 3.13")
    assert not matches_expected("25", "250")
    assert not matches_expected("9", "95000")
    assert not matches_expected("3", "Python 3.13.1")


def test_matches_expected_accepts_valid_outputs():
    assert matches_expected("y", "The column with the highest sum is y. Its sum is 30")
    assert matches_expected("y", "y")
    assert matches_expected("25", "25")
    assert matches_expected("1540", "Sum: 1540")
    assert matches_expected("[1, 2, 3, 4, 5, 6, 7, 8, 9]", "[1, 2, 3, 4, 5, 6, 7, 8, 9]")


def test_skyrl_reward_requires_successful_code_execution():
    env = JupyterSkyRLEnv(expected_output="y")
    env.last_output = "y"
    env._last_code_success = False
    assert env._compute_reward() == 0.0

    env._last_code_success = True
    env.last_output = "python"
    assert env._compute_reward() == 0.0

    env.last_output = "y"
    assert env._compute_reward() == 1.0


def test_gem_reward_requires_successful_code_execution():
    env = JupyterGemEnv(task="Print highest column", expected_output="y")
    env.last_output = "y"
    env._last_code_success = False
    assert env._compute_reward() == 0.0

    env._last_code_success = True
    env.last_output = "Defaulting to user installation"
    assert env._compute_reward() == 0.0

    env.last_output = "y"
    assert env._compute_reward() > 0.0
