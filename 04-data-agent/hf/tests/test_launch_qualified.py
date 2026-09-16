import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('launch_qualified', Path(__file__).parents[1] / 'launch_qualified.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_ambiguous_hf_submission_cannot_allocate_twice(tmp_path):
    class API:
        calls = 0

        def run_job(self, **kwargs):
            self.calls += 1
            assert json.loads((tmp_path / 'intent.json').read_text())['owner'] == 'unique-owner'
            raise TimeoutError('Response lost after allocation')

    api = API()
    wrapped = module.RecordedAPI(api, tmp_path / 'intent.json')
    kwargs = {'env': {'RUN_OWNER': 'unique-owner', 'BUNDLE_SHA256': 'abc'},
              'namespace': 'test', 'labels': {'role': 'train'}}
    with pytest.raises(TimeoutError):
        wrapped.run_job(**kwargs)
    with pytest.raises(RuntimeError, match='intent already exists'):
        wrapped.run_job(**kwargs)
    assert api.calls == 1


def test_unqualified_native_plan_does_not_launch(tmp_path, monkeypatch):
    from argparse import Namespace
    (tmp_path / 'plan.json').write_text('{"checkpoint_eval_gpu_validation": "pending"}')
    monkeypatch.setattr(module, 'run', lambda _: pytest.fail('unqualified allocation'))
    with pytest.raises(ValueError, match='not complete'):
        module.opencode(Namespace(ready=tmp_path), {})
