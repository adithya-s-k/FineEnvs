import importlib.util
import json
from pathlib import Path

import pytest

from airops_core.build import pack, windows

ROOT = Path(__file__).resolve().parents[3]
INPUTS = Path(__file__).resolve().parent / "fixtures" / "inputs"


def load_build_pack():
    spec = importlib.util.spec_from_file_location("build_pack", ROOT / "tools" / "build_pack.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(out: Path, derived: Path, inputs: Path = INPUTS, *extra: str) -> int:
    """The build on the fixture slice: one month, JFK, eight tails, so the size and disruption floors come down."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pack, "MIN_TAILS", 5)
        mp.setattr(pack, "MIN_DISRUPTION", 1)
        mp.setattr(pack, "EVAL", (("2023-06-26", "JFK"),))
        return load_build_pack().main(["--offline", "--months", "2023-06", "--hubs", "JFK", "--inputs", str(inputs),
                                       "--derived", str(derived), "--out", str(out), "--workers", "1", *extra])


@pytest.fixture(scope="session", autouse=True)
def solve_once(tmp_path_factory):
    """Every build of the session reads the window search's MIPs from one cache, whichever --derived it uses."""
    shared = tmp_path_factory.mktemp("leverage")
    solve = windows.leverage
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(windows, "leverage", lambda task, cache: solve(task, shared))
        yield


@pytest.fixture(scope="session")
def run_build():
    return run


@pytest.fixture(scope="session")
def built(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("pack")
    out = root / "a" / "b6-v1"
    assert run(out, root / "derived") == 0
    return out


@pytest.fixture(scope="session")
def task(built) -> dict:
    (line,) = (built / "tasks.jsonl").read_text().splitlines()
    return json.loads(line)
