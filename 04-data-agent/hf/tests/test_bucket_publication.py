"""An upload timeout must not lose a checkpoint or publish its marker early."""
import json
from unittest.mock import Mock, patch

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from artifacts import Publisher
from bucket_io import sync_with_retry
from checkpoint_store import READY


def http_error(code):
    return HfHubHTTPError("failure", response=httpx.Response(
        code, request=httpx.Request("POST", "https://example.test/bucket")))


@pytest.mark.parametrize("error", [TimeoutError("response decoding"),
    httpx.ReadError("connection interrupted"), http_error(503)])
def test_transfer_timeout_recovers_without_changing_filter(error):
    api = Mock()
    api.sync_bucket.side_effect = [error, None]
    with patch("bucket_io.time.sleep") as sleep:
        sync_with_retry(api, "checkpoint", "destination", include=[READY])
    assert api.sync_bucket.call_args_list[0] == api.sync_bucket.call_args_list[1]
    sleep.assert_called_once_with(30)


@pytest.mark.parametrize("error,attempts", [(TimeoutError(), 3), (http_error(403), 1),
                                          (ValueError("invalid data"), 1)])
def test_transfer_retries_are_bounded_and_do_not_hide_invalid_requests(error, attempts):
    api = Mock()
    api.sync_bucket.side_effect = error
    with patch("bucket_io.time.sleep"), pytest.raises(type(error)):
        sync_with_retry(api, "checkpoint", "destination")
    assert api.sync_bucket.call_count == attempts


def test_completion_marker_follows_retried_data_upload(tmp_path, monkeypatch):
    for key, value in {"ARTIFACT_BUCKET": "org/bucket", "RUN_ID": "run", "RUN_OWNER": "job",
                       "COMPARISON_ARM": "blackbox", "BUNDLE_SHA256": "bundle"}.items():
        monkeypatch.setenv(key, value)
    checkpoint = tmp_path / "run/checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.saved.json").write_text("{}")
    publisher = Publisher(tmp_path)
    api = Mock()
    operations = []
    failed = False

    def transfer(source, destination, **options):
        nonlocal failed
        assert not publisher.published
        operations.append((source, options))
        if source == str(checkpoint) and options.get("exclude") == [READY] and not failed:
            failed = True
            raise TimeoutError("committed data; lost response")

    api.sync_bucket.side_effect = transfer
    with patch("huggingface_hub.HfApi", return_value=api), patch("checkpoint_store.seal"), \
            patch("bucket_io.time.sleep"):
        publisher.sync()
    assert [options.get("include") for _, options in operations] == [None, None, None, [READY]]
    assert publisher.published == {"checkpoint-2"}
    assert json.loads((tmp_path / "upload_status.json").read_text())["published_checkpoints"] == ["checkpoint-2"]


def test_failed_data_upload_never_publishes_completion_marker(tmp_path, monkeypatch):
    for key, value in {"ARTIFACT_BUCKET": "org/bucket", "RUN_ID": "run", "RUN_OWNER": "job",
                       "COMPARISON_ARM": "blackbox", "BUNDLE_SHA256": "bundle"}.items():
        monkeypatch.setenv(key, value)
    checkpoint = tmp_path / "run/checkpoint-2"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.saved.json").write_text("{}")
    publisher = Publisher(tmp_path)
    api = Mock()
    api.sync_bucket.side_effect = [None, TimeoutError(), TimeoutError(), TimeoutError()]
    with patch("huggingface_hub.HfApi", return_value=api), patch("checkpoint_store.seal"), \
            patch("bucket_io.time.sleep"), pytest.raises(TimeoutError):
        publisher.sync()
    assert not publisher.published
    assert not any("include" in call.kwargs for call in api.sync_bucket.call_args_list)
    assert not (tmp_path / "upload_status.json").exists()
