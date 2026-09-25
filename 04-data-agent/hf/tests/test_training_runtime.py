"""Failure-boundary checks for HF checkpoint dispatch, model loading and launch gates."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HF = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HF), str(HF / "runtime"), str(HF.parent / "train")]
from checkpoint_store import READY, digest, restore_model
from common import MODEL, REVISION
from coordinator import eligible, evaluation_key, submit_once
from launch_gates import validate_proofs, validate_checkpoint_eval


class DispatchTest(unittest.TestCase):
    def test_ready_steps_and_final(self):
        self.assertFalse(eligible({"step": 50}, 100, True))
        self.assertTrue(eligible({"step": 100}, 100, True))
        self.assertTrue(eligible({"step": 151, "final": True}, 100, True))
        self.assertFalse(eligible({"step": 0, "final": True}, 100, True))

    def test_manifest_and_protocol_bind_eval_identity(self):
        key = evaluation_key("train1", "hash1", {"temperature": 0.8})
        self.assertNotEqual(key, evaluation_key("train1", "hash2", {"temperature": 0.8}))
        self.assertNotEqual(key, evaluation_key("train1", "hash1", {"temperature": 0.7}))

    def test_ambiguous_submission_is_not_repeated(self):
        state, snapshots, calls = {}, [], []
        def persist(): snapshots.append(copy.deepcopy(state))
        def launch():
            calls.append(1)
            raise TimeoutError("server may already have accepted the job")
        with self.assertRaises(TimeoutError):
            submit_once(state, "key", [], persist, launch)
        self.assertEqual(snapshots[0]["key"]["status"], "submitting")
        with self.assertRaisesRegex(RuntimeError, "Unresolved"):
            submit_once(state, "key", [], persist, launch)
        self.assertEqual(len(calls), 1)

    def test_adopt_job_after_ambiguous_response(self):
        state = {"key": {"status": "submitting"}}
        job = SimpleNamespace(id="accepted", labels={"evaluation_key": "key"})
        with patch("builtins.print"):
            actual = submit_once(state, "key", [job], lambda: None,
                                 lambda: self.fail("must adopt, not submit"))
        self.assertIs(actual, job)
        self.assertEqual(state["key"]["job_id"], "accepted")

    def test_failed_intent_persistence_never_submits(self):
        def persist(): raise OSError("remote storage unavailable")
        with self.assertRaises(OSError):
            submit_once({}, "key", [], persist, lambda: self.fail("unsafe submission"))


class LaunchGateTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((HF / "configs/deployment.json").read_text())
        self.baseline = {"arm": "blackbox", "comparison_ready": True, "graded_cells": 1000, "tito_pass": True}
        self.smoke = {"arm": "blackbox", "passed": True, "bundle_sha256": "new", "remote_restore_verified": True,
                      "tito_pass": True, "weights_updated": True, "native_optimizer_state_verified": True}

    def check(self, config=None, baseline=None, smoke=None):
        validate_proofs(self.config, {"sha256": "new"}, "blackbox", config or self.config,
                        baseline or self.baseline, smoke or self.smoke)

    def test_matching_evidence(self): self.check()

    def test_native_diagnostic_cannot_be_a_four_harness_curve_baseline(self):
        from launch_gates import validate_comparison_baseline
        native = {"comparison_ready": True, "graded_cells": 250, "tito_pass": True,
                  "implementation": "standalone-opencode"}
        with self.assertRaisesRegex(ValueError, "four-harness"):
            validate_comparison_baseline(native)
        matching = {"comparison_ready": True, "graded_cells": 1000, "tito_pass": True,
                    "harnesses": {name: {"graded": 250} for name in self.config["harness_pins"]}}
        validate_comparison_baseline(matching)
        matching["harnesses"]["opencode"]["graded"] = 249
        with self.assertRaises(ValueError):
            validate_comparison_baseline(matching)

    def test_dataset_change_rejected(self):
        old = copy.deepcopy(self.config)
        old["data"]["manifest_sha256"]["test_manifest.json"] = "different"
        with self.assertRaises(ValueError): self.check(config=old)

    def test_incomplete_baseline_rejected(self):
        with self.assertRaises(ValueError): self.check(baseline={**self.baseline, "graded_cells": 999})

    def test_old_smoke_bundle_rejected(self):
        with self.assertRaises(ValueError): self.check(smoke={**self.smoke, "bundle_sha256": "old"})

    def test_checkpoint_gate_binds_complete_scores_to_actual_smoke_weights(self):
        job = SimpleNamespace(status=SimpleNamespace(stage="COMPLETED"),
            labels={"role": "eval", "phase": "checkpoint", "arm": "whitebox", "training_job": "smoke"},
            environment={"BUNDLE_SHA256": "runtime", "CHECKPOINT_MANIFEST_SHA": "manifest",
                         "CHECKPOINT_STEP": "4", "CHECKPOINT_PREFIX": "bucket/smoke/checkpoint-4"})
        score = {"comparison_ready": True, "tito_pass": True, "graded_cells": 250, "arm": "whitebox"}
        evidence = {"step": 4, "bundle_sha256": "runtime", "manifest_sha256": "manifest",
                    "source": "bucket/smoke/checkpoint-4"}
        validate_checkpoint_eval({"sha256": "runtime"}, "whitebox", "smoke", job, score, evidence)
        for changed in [{**score, "graded_cells": 249}, {**score, "tito_pass": False}]:
            with self.assertRaises(ValueError):
                validate_checkpoint_eval({"sha256": "runtime"}, "whitebox", "smoke", job, changed, evidence)
        with self.assertRaises(ValueError):
            validate_checkpoint_eval({"sha256": "runtime"}, "whitebox", "other-smoke", job, score, evidence)
        with self.assertRaises(ValueError):
            validate_checkpoint_eval({"sha256": "runtime"}, "whitebox", "smoke", job, score,
                                     {**evidence, "manifest_sha256": "other-weights"})

    def test_dry_run_validates_without_submitting_any_job(self):
        from deploy import submit
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "bundle_uploaded.json").write_text(json.dumps({"sha256": "runtime", "repo": "org/repro", "revision": "pin"}))
            config = copy.deepcopy(self.config)
            config.pop("training_launch_hold", None)
            args = SimpleNamespace(role="train", phase="long", arm="whitebox", baseline_job="baseline",
                smoke_job="smoke", checkpoint_eval_job="eval", flavor="h200x2", training_job=None,
                external_checkpoint_coordinator=True, dp=1, limit=0, resume_eval_owner=None,
                timeout="24h", dry_run=True)
            api = Mock()
            api.space_info.return_value = SimpleNamespace(host="https://example.hf.space")
            proof = {"smoke_prefix": "smoke", "baseline_prefix": "base", "space_bundle_sha256": "space",
                     "space_url": "https://example.hf.space"}
            with patch("launch_gates.verify", return_value=proof) as smoke_gate, \
                 patch("launch_gates.verify_checkpoint_eval", return_value={"passed": True}) as eval_gate, \
                 patch("builtins.print"):
                preview = submit(api, config, {"HF_TOKEN": "never-record-this"}, out, args)
            smoke_gate.assert_called_once()
            eval_gate.assert_called_once()
            api.run_job.assert_not_called()
            self.assertFalse(preview["submitted"])
            self.assertNotIn("never-record-this", (out / "launch-preview.json").read_text())

    def test_native_launch_logs_comparison_baseline_and_retains_diagnostic(self):
        from deploy import submit
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "bundle_uploaded.json").write_text(json.dumps({"sha256": "runtime", "repo": "org/repro", "revision": "pin"}))
            args = SimpleNamespace(role="train", phase="long", arm="opencode", baseline_job="native",
                comparison_baseline_job="four-harness", smoke_job="smoke", flavor="a100x4",
                training_job=None, external_checkpoint_coordinator=False, dp=1, limit=0,
                resume_eval_owner=None, timeout="24h", dry_run=True)
            api = Mock()
            api.space_info.return_value = SimpleNamespace(host="https://example.hf.space")
            proof = {"smoke_prefix": "smoke", "baseline_prefix": "native-prefix", "space_bundle_sha256": "space",
                     "space_url": "https://example.hf.space"}
            response = Mock()
            response.raise_for_status.return_value.json.return_value = {
                "arm": "blackbox", "test_tasks": 250, "bundle_sha256": "space"}
            with patch("launch_gates.verify", return_value=proof), \
                 patch("launch_gates.verify_comparison_baseline", return_value="comparison-prefix") as gate, \
                 patch("runtime.service_contract.check", return_value={"passed": True}), \
                 patch("httpx.get", return_value=response), patch("builtins.print"):
                result = submit(api, self.config, {"HF_TOKEN": "test-only"}, out, args)
            gate.assert_called_once_with(api, self.config, "four-harness", out)
            self.assertEqual(result["environment"]["BASELINE_PREFIX"], "comparison-prefix")
            self.assertEqual(result["environment"]["BASELINE_JOB"], "four-harness")
            self.assertEqual(result["environment"]["NATIVE_BASELINE_PREFIX"], "native-prefix")
            api.run_job.assert_not_called()


class ModelRestoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "remote"
        self.source.mkdir()
        import torch
        from safetensors.torch import save_file
        save_file({"weight": torch.tensor([1.0, 2.0])}, self.source / "model.safetensors")
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
            (self.source / name).write_text("{}")
        (self.source / "optimizer.pt").write_bytes(b"optimizer must not be fetched for inference")
        manifest = {"arm": "blackbox", "bundle_sha256": "bundle", "step": 100,
                    "base_model": MODEL, "base_revision": REVISION,
                    "files": {p.name: digest(p) for p in self.source.iterdir()}}
        (self.source / READY).write_text(json.dumps(manifest))
        self.sha = digest(self.source / READY)
        self.downloaded = []
        case = self
        class API:
            def download_bucket_files(self, bucket, *, files, **kwargs):
                for name, target in files:
                    case.downloaded.append(Path(name).name)
                    shutil.copy2(case.source / Path(name).name, target)
        self.api = API

    def tearDown(self): self.tmp.cleanup()

    def run_restore(self):
        with patch("huggingface_hub.HfApi", self.api):
            return restore_model("hf://buckets/org/bucket/run/checkpoint-100", self.root / "model",
                arm="blackbox", bundle_sha256="bundle", manifest_sha256=self.sha)

    def test_model_only_restore_has_exact_hashes(self):
        self.assertEqual(self.run_restore()["step"], 100)
        self.assertNotIn("optimizer.pt", self.downloaded)
        self.assertEqual(digest(self.root / "model/model.safetensors"), digest(self.source / "model.safetensors"))

    def test_changed_manifest_is_rejected_before_weights(self):
        (self.source / READY).write_text((self.source / READY).read_text() + " ")
        with self.assertRaisesRegex(ValueError, "manifest changed"): self.run_restore()
        self.assertNotIn("model.safetensors", self.downloaded)

    def test_changed_model_is_rejected(self):
        (self.source / "model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "model hash mismatch"): self.run_restore()


class CoordinatorLifecycleTest(unittest.TestCase):
    def test_saved50_eval100_and_off_interval_final150(self):
        self.exercise_lifecycle()

    def test_user_stopped_job_evaluates_only_verified_final_checkpoint(self):
        self.exercise_lifecycle(stage="CANCELED", requested=True, expected=[150])

    def test_normal_cancellation_does_not_launch_an_evaluation(self):
        self.exercise_lifecycle(stage="CANCELED", expected=[])

    def exercise_lifecycle(self, stage="RUNNING", requested=False, expected=None):
        from huggingface_hub.errors import EntryNotFoundError
        import coordinator
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads((HF / "configs/deployment.json").read_text())
            (root / "hf/configs").mkdir(parents=True)
            (root / "hf/configs/deployment.json").write_text(json.dumps(config))
            remote = {}
            prefix = "run/jobs/train-owner/run"
            folders = [prefix + f"/checkpoint-{n}" for n in [50, 100, 150]]
            for n, folder in zip([50, 100, 150], folders):
                remote[folder + "/" + READY] = json.dumps({"arm": "blackbox", "bundle_sha256": "bundle", "step": n})
            submitted, snapshots = [], []
            trainer = SimpleNamespace(labels={"role": "train", "arm": "blackbox"},
                environment={"RUN_OWNER": "train-owner", "BUNDLE_SHA256": "bundle", "SPACE_URL": "https://environment.invalid",
                             "SPACE_BUNDLE_SHA256": "qualified-environment-bundle"},
                status=SimpleNamespace(stage=stage))
            peer = SimpleNamespace(id="coord1", status=SimpleNamespace(stage="RUNNING"), environment={"RUN_OWNER": "coord-owner"})
            class API:
                def inspect_job(self, **kwargs):
                    if submitted and stage == "RUNNING": trainer.status.stage = "COMPLETED"
                    return trainer
                def list_jobs(self, **kwargs):
                    return [peer] if kwargs["labels"]["role"] == "coordinator" else submitted
                def list_bucket_tree(self, bucket, **kwargs):
                    return [SimpleNamespace(path=f) for f in folders]
                def download_bucket_files(self, bucket, *, files, **kwargs):
                    for source, dest in files:
                        if source not in remote: raise EntryNotFoundError(source)
                        Path(dest).write_text(remote[source])
                def sync_bucket(self, source, dest, **kwargs):
                    snapshots.append(json.loads((Path(source) / "state.json").read_text()))
                def run_job(self, **kwargs):
                    self_case.assertTrue(snapshots[-1][kwargs["labels"]["evaluation_key"]]["status"] == "submitting")
                    self_case.assertEqual(kwargs["flavor"], "a100-large")
                    self_case.assertEqual(kwargs["env"]["SPACE_BUNDLE_SHA256"], "qualified-environment-bundle")
                    job = SimpleNamespace(id=f"eval{len(submitted)}", labels=kwargs["labels"],
                        environment=kwargs["env"], status=SimpleNamespace(stage="COMPLETED"))
                    owner = kwargs["env"]["RUN_OWNER"]
                    remote[f"run/jobs/{owner}/canonical_scores.json"] = json.dumps({"comparison_ready": True})
                    remote[f"run/jobs/{owner}/checkpoint_evaluation.json"] = json.dumps({"manifest_sha256": kwargs["env"]["CHECKPOINT_MANIFEST_SHA"]})
                    submitted.append(job)
                    return job
            self_case = self
            clock = [0]
            def sleep(seconds):
                clock[0] += seconds
                if clock[0] > 300: self.fail("coordinator failed to drain")
            env = {"TRAINING_JOB": "train1", "ARTIFACT_BUCKET": "org/bucket", "RUN_ID": "run",
                   "RUN_OWNER": "coord-owner", "BUNDLE_SHA256": "bundle", "BUNDLE_REPO": "org/bundle",
                   "BUNDLE_REVISION": "revision", "HF_TOKEN": "test-token"}
            request = None
            if requested:
                import hashlib
                request = {"training_job": "train1", "step": 150,
                           "checkpoint": "hf://buckets/org/bucket/" + folders[-1],
                           "manifest_sha256": hashlib.sha256(remote[folders[-1] + "/" + READY].encode()).hexdigest(),
                           "user_requested_stop": True, "full_checkpoint_verified": True}
            with patch.dict(os.environ, env), patch("coordinator.ROOT", root), patch("huggingface_hub.HfApi", API), \
                 patch("coordinator.time.sleep", sleep), patch("coordinator.time.monotonic", lambda: clock[0]):
                coordinator.run(root / "output", "blackbox", final_checkpoint=request)
            expected = [100, 150] if expected is None else expected
            self.assertEqual([int(j.environment["CHECKPOINT_STEP"]) for j in submitted], expected)
            self.assertEqual(len(list((root / "output/decisions/scores").glob("*.json"))), len(expected))

    def test_stop_authorization_rejects_changed_checkpoint_or_job(self):
        from coordinator import validate_requested_final
        request = {"training_job": "train1", "step": 150, "checkpoint": "bucket/cp150",
                   "manifest_sha256": "hash", "user_requested_stop": True, "full_checkpoint_verified": True}
        self.assertTrue(validate_requested_final(request, "train1", "bucket/cp150", {"step": 150}, "hash"))
        for key, value in [("training_job", "train2"), ("step", 100), ("checkpoint", "bucket/cp100"),
                           ("manifest_sha256", "changed"), ("full_checkpoint_verified", False),
                           ("user_requested_stop", False)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_requested_final({**request, key: value}, "train1", "bucket/cp150", {"step": 150}, "hash")


if __name__ == "__main__":
    unittest.main()
