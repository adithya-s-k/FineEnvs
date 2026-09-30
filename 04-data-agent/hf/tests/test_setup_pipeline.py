"""Verify deployment/training ordering and replay without duplicate allocations."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

HF = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HF), str(HF / "runtime")]
import setup_pipeline


class PipelineTest(unittest.TestCase):
    def execute(self, fail_baseline=False):
        from huggingface_hub.errors import EntryNotFoundError
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads((HF / "configs/deployment.json").read_text())
            config["run_id"] = "test-run"
            config["pipeline"]["baseline_jobs"] = {"blackbox": "bb", "whitebox": "wb"}
            (root / "hf/configs").mkdir(parents=True)
            (root / "hf/configs/deployment.json").write_text(json.dumps(config))
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "bundle.json").write_text(json.dumps({"sha256": "new"}))
            (bundle / "bundle.tar.gz").write_bytes(b"fixture bundle")
            jobs = []
            baselines = {job_id: NS(id=job_id, status=NS(stage="ERROR" if fail_baseline and arm == "whitebox" else "COMPLETED"),
                        environment={"RUN_OWNER": "base-" + arm}) for arm, job_id in config["pipeline"]["baseline_jobs"].items()}
            remote = {f"test-run/jobs/base-{arm}/canonical_scores.json": json.dumps({"arm": arm, "comparison_ready": True})
                      for arm in ["blackbox", "whitebox"]}
            events = []
            deployed = [False]
            class API:
                def inspect_job(self, *, job_id, **kwargs):
                    return baselines.get(job_id) or next(j for j in jobs if j.id == job_id)
                def list_jobs(self, *, labels, **kwargs):
                    return [j for j in jobs if all(j.labels.get(k) == v for k, v in labels.items())]
                def download_bucket_files(self, bucket, *, files, **kwargs):
                    for name, dest in files:
                        if name not in remote: raise EntryNotFoundError(name)
                        Path(dest).write_text(remote[name])
                def sync_bucket(self, source, dest, **kwargs):
                    remote[dest.removeprefix("hf://buckets/org/bucket/") + "/pipeline.json"] = (Path(source) / "pipeline.json").read_text()
                def space_info(self, repo): return NS(host="https://" + repo.replace("/", "-"))
            def spaces(*args):
                events.append("deploy")
                deployed[0] = True
            def get(*args, **kwargs):
                return NS(raise_for_status=lambda: None, json=lambda: {"admission": {"active": {"train": 0, "eval": 0}},
                          "bundle_sha256": "new" if deployed[0] else "old"})
            def start(*args):
                events.append("ui")
                return NS(wait=lambda **kwargs: 0)
            def submit(api, config, secret, out, args):
                events.append(f"{args.role}:{args.arm}:{args.phase}")
                job_id = "created" + str(len(jobs))
                labels = {"role": args.role, "arm": args.arm, "phase": args.phase, "run": "test-run"}
                job = NS(id=job_id, labels=labels, status=NS(stage="COMPLETED"),
                         environment={"BUNDLE_SHA256": "new", "RUN_OWNER": job_id})
                jobs.append(job)
                remote[f"test-run/jobs/{job_id}/" + ("audit/metrics.jsonl" if args.arm == "blackbox" else "run/metrics.jsonl")] = json.dumps(
                    {"step": 12, "loss": 0.0, "grad_norm": 1.0, "reward": 0.5}) + "\n"
                return {"id": job_id, "stage": "COMPLETED"}
            env = {"ARTIFACT_BUCKET": "org/bucket", "RUN_ID": "test-run", "BUNDLE_SHA256": "new",
                   "BUNDLE_REPO": "org/repro", "BUNDLE_REVISION": "revision", "RUN_OWNER": "setup-owner"}
            with patch.dict(os.environ, env), patch("setup_pipeline.ROOT", root), patch("setup_pipeline.BUNDLE", bundle), \
                 patch("huggingface_hub.HfApi", API), patch("httpx.get", get), patch("deploy.spaces", spaces), \
                 patch("deploy.submit", submit), patch("setup_pipeline.start", start):
                if fail_baseline:
                    with self.assertRaisesRegex(RuntimeError, "ERROR"):
                        setup_pipeline.run(root / "output")
                    self.assertEqual(events, [])
                    self.assertEqual(jobs, [])
                    return
                setup_pipeline.run(root / "output")
                self.assertEqual(events[:2], ["deploy", "ui"])
                self.assertEqual(events[2:6], ["train:blackbox:smoke", "train:whitebox:smoke", "train:blackbox:long", "train:whitebox:long"])
                self.assertEqual(len(jobs), 6)
                setup_pipeline.run(root / "replayed")
                self.assertEqual(len(jobs), 6, "Replay must adopt all existing jobs")
                self.assertEqual(len(events), 8, "Replay must not rebuild Spaces or rerun the UI")
                state = json.loads((root / "replayed/pipeline.json").read_text())
                self.assertEqual(state["phase"], "completed")
                self.assertTrue(state["passed"])

    def test_complete_baselines_smoke_before_long_and_replay(self): self.execute()
    def test_failed_baseline_cannot_deploy_or_allocate_training(self): self.execute(fail_baseline=True)


if __name__ == "__main__":
    unittest.main()
