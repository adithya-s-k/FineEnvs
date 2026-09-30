"""A late logger must never race a trainer or publish scores from other weights."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from late_hf_logging import replay, planned_stop_matches


class LateLoggingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.env = self.root / "env"
        self.env.write_text("HF_API_KEY=unused-test-token\n")
        self.plan = self.root / "plan.json"
        self.plan.write_text(json.dumps({"files": {}, "env_file": str(self.env),
            "training_job": "train", "namespace": "org", "bundle_sha256": "bundle"}))
        self.api = Mock()
        self.job = SimpleNamespace(status=SimpleNamespace(stage="COMPLETED"),
            environment={"BUNDLE_SHA256": "bundle", "ARTIFACT_BUCKET": "org/bucket",
                         "RUN_ID": "run", "RUN_OWNER": "owner"})
        self.api.inspect_job.return_value = self.job

    def tearDown(self):
        self.temporary.cleanup()

    def run_replay(self):
        with patch("huggingface_hub.HfApi", return_value=self.api), patch.dict("os.environ"):
            replay(self.plan)

    def test_active_trainer_rejected_before_any_bucket_sync(self):
        self.job.status.stage = "RUNNING"
        with self.assertRaisesRegex(ValueError, "completed trainer"):
            self.run_replay()
        self.api.sync_bucket.assert_not_called()

    def test_only_explicit_bound_stop_is_eligible_for_late_logging(self):
        self.job.status.stage = "CANCELED"
        plan = json.loads(self.plan.read_text())
        self.assertFalse(planned_stop_matches(self.job, plan))
        plan["final_checkpoint"] = {"training_job": "train", "step": 150,
            "checkpoint": "hf://buckets/org/bucket/run/jobs/owner/run/checkpoint-150",
            "manifest_sha256": "a" * 64, "user_requested_stop": True, "full_checkpoint_verified": True}
        self.assertTrue(planned_stop_matches(self.job, plan))
        for key, value in [("training_job", "other"), ("step", 100), ("manifest_sha256", ""),
                           ("full_checkpoint_verified", False), ("user_requested_stop", False)]:
            changed = {**plan, "final_checkpoint": {**plan["final_checkpoint"], key: value}}
            with self.subTest(key=key):
                self.assertFalse(planned_stop_matches(self.job, changed))
        self.job.status.stage = "RUNNING"
        self.assertFalse(planned_stop_matches(self.job, plan))

    def test_other_weights_cannot_be_logged_under_this_trainer(self):
        logs = self.root / "late-training-logs"
        logs.mkdir()
        (logs / "status.json").write_text(json.dumps({"passed": True, "finished_at": 123}))
        scores = self.root / "output/decisions/scores"
        scores.mkdir(parents=True)
        (scores / "step-000100.json").write_text(json.dumps({"step": 100,
            "source": {"bundle_sha256": "bundle", "step": 100,
                       "source": "hf://buckets/org/bucket/run/jobs/other-trainer/run/checkpoint-100"},
            "scores": {"comparison_ready": True, "tito_pass": True, "arm": "whitebox", "graded_cells": 250}}))
        with self.assertRaisesRegex(ValueError, "Unverified checkpoint"):
            self.run_replay()
        self.api.sync_bucket.assert_called_once()
        # The only operation was a remote-to-local read, never a bucket upload.
        self.assertEqual(self.api.sync_bucket.call_args.args[0], "hf://buckets/org/bucket/run/jobs/owner")


if __name__ == "__main__":
    unittest.main()
