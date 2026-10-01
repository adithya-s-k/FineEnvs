import builtins
import runpy
import sys
from pathlib import Path

import pytest


@pytest.mark.contract
@pytest.mark.parametrize("mode", ["whitebox", "multi_harness"])
def test_setup_does_not_require_native_opencode(monkeypatch, mode):
    original_import = builtins.__import__

    def without_opencode(name, *args, **kwargs):
        if name == "opencode_env" or name.startswith("opencode_env."):
            raise ModuleNotFoundError("Native OpenCode is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_opencode)
    monkeypatch.setattr(sys, "argv", ["check_setup.py", "--mode", mode])
    runpy.run_path(
        str(Path(__file__).parents[1] / "check_setup.py"), run_name="__main__"
    )
