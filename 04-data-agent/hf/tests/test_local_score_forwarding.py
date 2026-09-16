import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("local_watch", Path(__file__).resolve().parents[1] / "local_watch.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class ScoreForwardingTest(unittest.TestCase):
    def test_matching_baseline_and_checkpoint_forward_for_every_recipe(self):
        import hashlib
        for arm, cells in (("blackbox", 1000), ("opencode", 1000), ("whitebox", 250)):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                baseline = root / "baseline.json"
                baseline.write_text(json.dumps({"graded_cells": cells, "comparison_ready": True}))
                plan = root / "plan.json"
                plan.write_text(json.dumps({"root": str(root), "arm": arm, "training_job": "123",
                    "bundle_sha256": "bundle", "baseline_score": str(baseline),
                    "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()}))
                directory = root / "checkpoint-evals"
                directory.mkdir()
                (directory / "scores-000100.json").write_text(json.dumps({
                    "proof": {"passed": True, "training_arm": arm, "graded_cells": cells,
                              "bundle_sha256": "bundle", "manifest_sha256": "model", "step": 100},
                    "evaluation": {"manifest_sha256": "model", "step": 100, "job_id": "456"},
                    "scores": {"comparison_ready": True, "graded_cells": cells}}))
                _, training, changed = watch.forward_scores(plan)
                self.assertTrue(changed)
                for step in (0, 100):
                    record = json.loads((training / f"checkpoint-scores/step-{step:06d}.json").read_text())
                    self.assertEqual(record["scores"]["graded_cells"], cells)
                baseline.write_text('{}')
                with self.assertRaisesRegex(ValueError, "baseline score changed"):
                    watch.forward_scores(plan)

    def test_only_verified_full_scores_enter_training_curve_and_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = root / "plan.json"
            plan.write_text(json.dumps({"root": str(root), "arm": "opencode", "training_job": "123",
                                       "bundle_sha256": "bundle"}))
            (root / "checkpoint-evals").mkdir()
            source = root / "checkpoint-evals/scores-000100.json"
            record = {"proof": {"passed": True, "qualification_only": True, "training_arm": "opencode",
                "graded_cells": 1000, "bundle_sha256": "bundle", "manifest_sha256": "model", "step": 100},
                "evaluation": {"manifest_sha256": "model", "step": 100, "job_id": "456"},
                "scores": {"comparison_ready": True, "average_pass_at_1": 0.2}}
            source.write_text(json.dumps(record))
            self.assertFalse(watch.forward_scores(plan)[2])
            record["proof"]["qualification_only"] = False
            source.write_text(json.dumps(record))
            self.assertTrue(watch.forward_scores(plan)[2])
            self.assertFalse(watch.forward_scores(plan)[2])
            target = root / "outputs/local-train-opencode-123/checkpoint-scores/step-000100.json"
            event = json.loads(target.read_text())
            self.assertEqual(event["step"], 100)
            self.assertEqual(event["source"]["job_id"], "456")
            record["scores"]["average_pass_at_1"] = 0.4
            source.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                watch.forward_scores(plan)
            self.assertEqual(json.loads(target.read_text()), event)


if __name__ == "__main__":
    unittest.main()
