import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HF = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, HF / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


followup = module("local_followup")
long = module("local_long")
gates = module("launch_gates")


class LocalQualificationTests(unittest.TestCase):
    def test_eval_admission_preserves_fifty_slots_with_valid_reservation(self):
        spec = importlib.util.spec_from_file_location("tested_service_policy", HF / "runtime/service_policy.py")
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        admission = policy.Admission(int(followup.EVAL_ADMISSION["SANDBOX_CAPACITY"]),
                                     int(followup.EVAL_ADMISSION["TRAIN_RESERVED_SANDBOXES"]))
        for _ in range(50):
            admission.acquire("eval", timeout=0)
        with self.assertRaises(TimeoutError):
            admission.acquire("eval", timeout=0)
        admission.acquire("train", timeout=0)
        for _ in range(50):
            admission.release("eval")
        admission.release("train")
        self.assertEqual(admission.active, {"train": 0, "eval": 0})

    def test_checkpoint_smoke_requires_each_task_harness_and_pinned_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads((HF / "configs/deployment.json").read_text())
            pins = config["harness_pins"]
            (root / "hf/configs").mkdir(parents=True)
            (root / "hf/configs/deployment.json").write_text(json.dumps(config))
            tasks = root / "experiments/daytona_harness_comparison/logs/20260915"
            tasks.mkdir(parents=True)
            (tasks / "test_indices.txt").write_text("17,42,8")
            output = root / "eval"
            output.mkdir()
            scores = {"graded_cells": 8, "comparison_ready": False,
                "harness_versions": {h: {v: 2} for h, v in pins.items()}}
            (output / "canonical_scores.json").write_text(json.dumps(scores))
            reports = [{"harness": h, "index": i, "tito_pass": True} for h in pins for i in (17, 42)]
            audit = output / "final_tito.json"
            audit.write_text(json.dumps({"reports": reports}))
            plan = {"root": str(root), "arm": "opencode", "checkpoint_eval_qualification": True,
                    "bundle_sha256": "runtime", "controller_sha256": "controller"}
            record = {"step": 4, "manifest_sha256": "checkpoint"}
            proof = followup.validate_evaluation(plan, output, record)
            self.assertTrue(proof["qualification_only"])
            audit.write_text(json.dumps({"reports": reports[:-1]}))
            with self.assertRaises(ValueError):
                followup.validate_evaluation(plan, output, record)
            audit.write_text(json.dumps({"reports": reports}))
            scores["harness_versions"]["opencode"] = {"unknown": 2}
            (output / "canonical_scores.json").write_text(json.dumps(scores))
            with self.assertRaises(ValueError):
                followup.validate_evaluation(plan, output, record)
            plan["checkpoint_eval_qualification"] = False
            with self.assertRaises(ValueError):
                followup.validate_evaluation(plan, output, record)

    def test_admission_changes_do_not_hide_score_protocol_changes(self):
        import copy
        config = json.loads((HF / "configs/deployment.json").read_text())
        changed = copy.deepcopy(config)
        changed["evaluation"]["concurrency_per_arm"]["opencode"] = 50
        self.assertEqual(gates.protocol_identity(config), gates.protocol_identity(changed))
        changed["evaluation"]["max_output_tokens_per_call"] = 8192
        self.assertNotEqual(gates.protocol_identity(config), gates.protocol_identity(changed))

    def test_only_hundred_intervals_and_final_are_evaluated(self):
        manifests = [{"step": n} for n in [0, 50, 100, 150, 200, 250]]
        self.assertEqual(followup.eligible_steps(manifests, False), {100, 200})
        self.assertEqual(followup.eligible_steps(manifests, True), {100, 200, 250})
        manifests[-1]["final"] = True
        self.assertEqual(followup.eligible_steps(manifests, False), {100, 200, 250})
        self.assertEqual(followup.eligible_steps([{"step": 2, "final": True}], True, qualification=True), set())
        self.assertEqual(followup.eligible_steps([{"step": 2, "final": True}, {"step": 4}], True,
                                               qualification=True), {4})

    def test_partial_or_wrong_implementation_cannot_qualify_native_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scores.json"
            valid = {"comparison_ready": True, "graded_cells": 250, "tito_pass": True,
                     "implementation": "standalone-opencode"}
            path.write_text(json.dumps({"daytona": valid}))
            long.validate_baseline(path, "opencode")
            for key, value in [("graded_cells", 249), ("tito_pass", False),
                               ("implementation", "harbor")]:
                path.write_text(json.dumps({**valid, key: value}))
                with self.assertRaises(ValueError):
                    long.validate_baseline(path, "opencode")

    def test_passing_proof_cannot_hide_source_changes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = root / "outputs/local-train-whitebox-123"
            smoke.mkdir(parents=True)
            source = root / "trainer.py"
            source.write_text("original trainer")
            encoded = json.dumps({"files": {"trainer.py": hashlib.sha256(source.read_bytes()).hexdigest()}}).encode()
            sha = hashlib.sha256(encoded).hexdigest()
            (root / "bundle_manifest.json").write_bytes(encoded)
            (root / "local_manifest.json").write_text(json.dumps({"sha256": sha}))
            proof = {"arm": "whitebox", "bundle_sha256": sha, "passed": True,
                     "remote_restore_verified": True, "tito_pass": True, "weights_updated": True,
                     "native_optimizer_state_verified": True}
            (smoke / "training_smoke_verified.json").write_text(json.dumps(proof))
            self.assertEqual(long.proof_for(smoke, "whitebox")[0], root)
            source.write_text("changed trainer")
            with self.assertRaises(ValueError):
                long.proof_for(smoke, "whitebox")

    def test_native_grading_gate_rejects_the_old_zero_tolerance_fallback(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "experiments/daytona_harness_comparison/logs/20260915/source/packages/data_agent_env"
            package.mkdir(parents=True)
            hashes = {}
            for name in ("task.py", "tasks.py", "verifier.py", "grader.py"):
                path = package / name
                path.write_text("explicit zero tolerance preserved")
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            baseline = root / "canonical_scores.json"
            (root / "verification.json").write_text(json.dumps({"passed": True,
                "parameters_verified": 1250, "original_graded_records_preserved": True,
                "reports": [{}] * 250, "runtime_files": hashes}))
            long.validate_native_grading(root, baseline)
            (package / "verifier.py").write_text("task.atol or 1e-3")
            with self.assertRaisesRegex(ValueError, "Native grader differs"):
                long.validate_native_grading(root, baseline)


if __name__ == "__main__":
    unittest.main()
