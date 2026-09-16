import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec = importlib.util.spec_from_file_location('local_watch', Path(__file__).parents[1] / 'local_watch.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_hf_eval_startup_also_reserves_capacity():
    api = SimpleNamespace(list_jobs=lambda **_: [SimpleNamespace(status=SimpleNamespace(stage='STARTING'))])
    assert not module.admit_with_hf(lambda _: pytest.fail('local GPU allocated during HF eval startup'), {}, api)


def test_idle_hf_still_checks_local_and_sandbox_capacity():
    api = SimpleNamespace(list_jobs=lambda **_: [SimpleNamespace(status=SimpleNamespace(stage='COMPLETED'))])
    assert not module.admit_with_hf(lambda _: False, {}, api)
    assert module.admit_with_hf(lambda _: True, {}, api)


def test_unknown_hf_capacity_never_allocates():
    def unavailable(**kwargs):
        raise TimeoutError('HF unavailable')
    api = SimpleNamespace(list_jobs=unavailable)
    with pytest.raises(TimeoutError):
        module.admit_with_hf(lambda _: pytest.fail('unknown capacity'), {}, api)
