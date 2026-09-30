import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from huggingface_hub.errors import HfHubHTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from coordinator import sync_decisions, verified_terminal_result


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.args = {
            "stage": "ERROR",
            "score": {"comparison_ready": True, "complete": True, "tito_pass": True,
                      "graded_cells": 250, "expected_cells": 250},
            "evidence": {"manifest_sha256": "hash", "step": 100, "bundle_sha256": "bundle", "source": "checkpoint"},
            "manifest": {"arm": "whitebox", "step": 100, "bundle_sha256": "bundle"},
            "sha": "hash", "source": "checkpoint",
            "status": {"passed": True, "finished_at": 123, "arm": "whitebox", "phase": "checkpoint"},
        }

    def test_uploaded_complete_result_survives_provider_failure(self):
        verified_terminal_result(**self.args)

    def test_failed_partial_wrong_checkpoint_and_unvalidated_results_rejected(self):
        changes = [("score", "graded_cells", 249), ("score", "tito_pass", False),
                   ("score", "comparison_ready", False), ("status", "passed", False),
                   ("status", "phase", "baseline"), ("status", "finished_at", None),
                   ("evidence", "manifest_sha256", "other"), ("evidence", "step", 200),
                   ("evidence", "source", "other"), ("evidence", "bundle_sha256", "other")]
        for group, field, value in changes:
            with self.subTest(field=field):
                args = copy.deepcopy(self.args)
                args[group][field] = value
                with self.assertRaises(ValueError):
                    verified_terminal_result(**args)
        for stage in ("RUNNING", "CANCELED", "DELETED"):
            with self.assertRaises(ValueError):
                verified_terminal_result(**{**self.args, "stage": stage})

    def error(self, code):
        return HfHubHTTPError("test failure", response=httpx.Response(
            code, request=httpx.Request("POST", "https://example.test/bucket")))

    @patch("coordinator.time.sleep")
    def test_transient_upload_retried(self, sleep):
        api = Mock()
        api.sync_bucket.side_effect = [self.error(500), None]
        sync_decisions(api, Path("decisions"), "destination")
        self.assertEqual(api.sync_bucket.call_count, 2)
        sleep.assert_called_once_with(30)

    @patch("coordinator.time.sleep")
    def test_retries_bounded_and_permission_errors_fail(self, sleep):
        for code, expected in ((503, 3), (403, 1)):
            api = Mock()
            api.sync_bucket.side_effect = self.error(code)
            with self.assertRaises(HfHubHTTPError):
                sync_decisions(api, Path("decisions"), "destination")
            self.assertEqual(api.sync_bucket.call_count, expected)


if __name__ == "__main__":
    unittest.main()
