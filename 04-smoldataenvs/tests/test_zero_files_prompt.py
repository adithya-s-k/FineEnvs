from __future__ import annotations

import importlib.util
from pathlib import Path


def load_rollout():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rollout.py"
    spec = importlib.util.spec_from_file_location("test_rollout_zero_files", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def user_prompt(row):
    return load_rollout().build_prompt(row)[-1]["content"]


def test_prompt_is_unchanged_when_files_are_declared():
    prompt = user_prompt({"question": "Q?", "files": ["data.csv", "notes.txt"]})
    assert "The files are in /home/user/input and your program runs in that directory:\n- data.csv\n- notes.txt" in prompt
    assert "list /home/user/input" not in prompt


def test_prompt_tells_model_to_list_input_dir_when_file_names_are_missing():
    prompt = user_prompt({"question": "Q?", "files": []})
    assert "No file names were provided by the dataset." in prompt
    assert "list /home/user/input" in prompt
    assert "os.listdir('/home/user/input')" in prompt
