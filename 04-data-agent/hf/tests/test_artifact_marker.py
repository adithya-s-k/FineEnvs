from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))


def test_final_marker_change_republishes_checkpoint(tmp_path, monkeypatch):
    import huggingface_hub
    import checkpoint_store
    from artifacts import Publisher

    for key, value in {'ARTIFACT_BUCKET': 'org/bucket', 'RUN_ID': 'test',
                       'RUN_OWNER': 'owner', 'COMPARISON_ARM': 'blackbox',
                       'BUNDLE_SHA256': 'sha'}.items():
        monkeypatch.setenv(key, value)
    calls = []
    class API:
        def sync_bucket(self, *args, **kwargs):
            calls.append(kwargs)
    monkeypatch.setattr(huggingface_hub, 'HfApi', API)
    monkeypatch.setattr(checkpoint_store, 'seal', lambda *args, **kwargs: None)
    folder = tmp_path / 'run/checkpoint-550'
    folder.mkdir(parents=True)
    marker = folder / 'checkpoint.saved.json'
    marker.write_text('{"final":false}')
    publisher = Publisher(tmp_path)
    publisher.sync()
    publisher.sync()
    assert sum('include' in call for call in calls) == 1
    marker.write_text('{"final":true}')
    publisher.sync()
    assert sum('include' in call for call in calls) == 2
    assert publisher.published == {'checkpoint-550'}
