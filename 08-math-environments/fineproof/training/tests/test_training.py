"""The training pieces that need no GPU, TRL or OpenEnv: the objective's advantage, the cut-off check and ungraded
rollouts, the rollout metrics, the eval summary, task preparation, checkpoint copies and restores, and which
checkpoints the eval launcher sends.

    python -m unittest discover -s tests
"""

import asyncio
import contextlib
import io
import json
import math
import os
import shutil
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "jobs")]
sys.modules.setdefault("httpx", types.ModuleType("httpx"))  # jobs/run.py imports it for its health checks only


class _StubHarnessLoop:
    """TRL's `_HarnessRolloutLoop._score_group` as far as objective.py relies on it: one sample per training row, in
    rollout order, each with the rollout's std-divided advantage. The std is reported under TRL main's metric name,
    which the pinned TRL calls `reward_std`."""

    reward_funcs = []

    async def _score_group(self, group):
        rewards = [math.nan if r is None else r for r in group.rollout_rewards]
        scored = [r for r in rewards if not math.isnan(r)]
        mean = sum(scored) / len(scored)
        std = (sum((r - mean) ** 2 for r in scored) / len(scored)) ** 0.5
        return [types.SimpleNamespace(advantage=0.0 if math.isnan(r) else (r - mean) / (std + 1e-8),
                                      metrics={"reward_kept": r, "reward_group_std": std})
                for r, sequences in zip(rewards, group.completions_sequences) for _ in sequences]


def _stub_trl():
    """objective.py subclasses TRL classes at import; stand-ins are enough for its own logic."""
    base = types.ModuleType("trl.experimental.async_grpo")
    base.AsyncGRPOTrainer = type("AsyncGRPOTrainer", (), {})
    harness = types.ModuleType("trl.experimental.async_grpo.openenv_harness")
    harness.HarnessRolloutWorker = type("HarnessRolloutWorker", (), {})
    harness._HarnessRolloutLoop = _StubHarnessLoop
    for name, module in (("trl", types.ModuleType("trl")), ("trl.experimental", types.ModuleType("trl.experimental")),
                         ("trl.experimental.async_grpo", base),
                         ("trl.experimental.async_grpo.openenv_harness", harness)):
        sys.modules.setdefault(name, module)


class Objective(unittest.TestCase):
    def test_mean_centered(self):
        _stub_trl()
        from objective import mean_centered

        advantages = mean_centered([0.5, 0.52, None, 0.48, float("nan"), 0.5])
        for got, want in zip(advantages, [0.0, 0.02, 0.0, -0.02, 0.0, 0.0]):
            self.assertAlmostEqual(got, want, places=12)  # unscored rollouts get 0 and stay out of the mean
        self.assertEqual(mean_centered([None, None]), [0.0, 0.0])

    def test_the_loop_never_divides_by_the_std(self):
        """Would fail if the std division came back: TRL's advantages here are +-1.41, the mean-centred ones +-0.02."""
        _stub_trl()
        from objective import _MeanCenteredLoop

        # Five rollouts: one gave two training rows, the unscored one none.
        group = types.SimpleNamespace(rollout_rewards=[0.5, 0.52, None, 0.48, 0.5],
                                      completions_sequences=[["a"], ["b", "c"], [], ["d"], ["e"]])
        samples = asyncio.run(_MeanCenteredLoop()._score_group(group))
        for got, want in zip([s.advantage for s in samples], [0.0, 0.02, 0.02, -0.02, 0.0], strict=True):
            self.assertAlmostEqual(got, want, places=12)

    def test_a_changed_sample_layout_fails_loudly(self):
        _stub_trl()
        from objective import _MeanCenteredLoop

        loop = _MeanCenteredLoop()
        group = types.SimpleNamespace(rollout_rewards=[0.1, 0.9], completions_sequences=[["a"], ["b"]])
        with mock.patch.object(_StubHarnessLoop, "_score_group", mock.AsyncMock(return_value=[mock.Mock()] * 3)):
            with self.assertRaises(RuntimeError):
                asyncio.run(loop._score_group(group))
        loop.reward_funcs = [lambda **kwargs: None]  # another reward source: the rollout reward is no longer the sum
        with self.assertRaises(RuntimeError):
            asyncio.run(loop._score_group(group))


@dataclass
class _Verdict:
    env_reward: float | None
    done: bool = True


class _StubHarborSession:
    """OpenEnv's HarborSession.verify, as far as tasks.py relies on it: the verifier's reward from the result."""

    def verify(self, transcript, final_state=None):
        return _Verdict(env_reward=self.result.reward)


def _turn(finish):
    return types.SimpleNamespace(finish_reason=finish, prompt_token_ids=[1], completion_token_ids=[2])


def _result(reward, finishes, ok=True, exception=None, trial="trial-1"):
    """An OpenEnv HarborRolloutResult with the fields rollout_log records, one captured turn per finish reason."""
    return types.SimpleNamespace(
        trial_name=trial, task_name="task", reward=reward, ok=ok, error=None, exception_type=exception,
        rollout_type="train", capture_level="tokens", n_turns=len(finishes), n_roots=1, n_trainable_tokens=2,
        wall_s=1.0, phase_timings={}, findings=[], budget_stop_count=0, param_fixes=[],
        turns=[_turn(f) for f in finishes], agent_log_tail="")


class CutOffAndUngraded(unittest.TestCase):
    def test_the_submitted_call_counts(self):
        from tasks import submitted_proof_truncated

        result = _result(0.5, ["stop", "length"])
        # A rewrite cut off before its heading: the complete first proof was submitted.
        self.assertFalse(submitted_proof_truncated(result, {"submitted_call": 0, "finish_reasons": ["stop", "length"]}))
        # A cut-off first proof, then a rewrite without the heading: the cut-off proof was submitted.
        result = _result(0.5, ["length", "stop"])
        self.assertTrue(submitted_proof_truncated(result, {"submitted_call": 0, "finish_reasons": ["length", "stop"]}))
        self.assertFalse(submitted_proof_truncated(result, {"submitted_call": 1, "finish_reasons": ["length", "stop"]}))
        # No record from the agent: the last captured call stands in (right for one call per proof).
        self.assertTrue(submitted_proof_truncated(_result(0.5, ["length"]), {}))
        self.assertFalse(submitted_proof_truncated(_result(0.5, []), {}))

    def verify(self, result, record=None, zero_truncated=True):
        """Run FineProofSession.verify on `result`, with `record` as the agent's metadata in the trial's result.json.
        Returns the reward GRPO sees and the rollouts.jsonl line."""
        harbor_env = types.ModuleType("harbor_env")
        harness = types.ModuleType("harbor_env.harness")
        harness.HarborSession = _StubHarborSession
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.dict(sys.modules, {"harbor_env": harbor_env, "harbor_env.harness": harness}):
            from tasks import _session_class

            trials = Path(directory) / "trials"
            (trials / result.trial_name).mkdir(parents=True)
            if record is not None:
                (trials / result.trial_name / "result.json").write_text(
                    json.dumps({"agent_result": {"metadata": {"fineproof": record}}}))
            session = _session_class().__new__(_session_class())
            session.result, session.trial_root, session.zero_truncated = result, trials, zero_truncated
            reward = session.verify([]).env_reward
            lines = (Path(directory) / "rollouts.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 1)
        return reward, json.loads(lines[0])

    def test_a_complete_proof_is_not_zeroed_for_a_cut_off_rewrite(self):
        reward, line = self.verify(_result(0.6, ["stop", "length"]),
                                   {"submitted_call": 0, "finish_reasons": ["stop", "length"]})
        self.assertEqual((reward, line["reward"], line["truncated"]), (0.6, 0.6, False))

    def test_a_cut_off_proof_is_zeroed_whichever_call_came_last(self):
        reward, line = self.verify(_result(0.3, ["length", "stop"]),
                                   {"submitted_call": 0, "finish_reasons": ["length", "stop"]})
        self.assertEqual((reward, line["reward"], line["judge_reward"], line["truncated"]), (0.0, 0.0, 0.3, True))

    def test_an_agent_timeout_is_logged_ungraded(self):
        import rollout_log

        reward, line = self.verify(_result(0.0, ["stop"], exception="AgentTimeoutError"))
        self.assertIsNone(reward)
        self.assertIsNone(line["reward"])  # not the verifier's 0 for the missing proof
        metrics = rollout_log.summarize([line, {"reward": 0.5, "ok": True, "calls": []}])
        self.assertEqual((metrics["rollouts/graded_frac"], metrics["rollouts/agent_timeouts"]), (0.5, 1))
        self.assertEqual(metrics["rollouts/zero_frac"], 0.0)


class RolloutMetrics(unittest.TestCase):
    def test_zeroed_proofs_keep_their_judge_score(self):
        import rollout_log

        records = [{"reward": 0.0, "judge_reward": 0.4, "truncated": True, "ok": True, "calls": [{"finish": "length", "prompt_tokens": 9, "completion_tokens": 30}]},
                   {"reward": 0.8, "truncated": False, "ok": True, "calls": [{"finish": "stop", "prompt_tokens": 9, "completion_tokens": 20}]}]
        metrics = rollout_log.summarize(records)
        self.assertAlmostEqual(metrics["rollouts/reward_mean"], 0.4)
        self.assertAlmostEqual(metrics["rollouts/judge_reward_mean"], 0.6)
        self.assertEqual(metrics["rollouts/zeroed_truncated"], 1)
        self.assertAlmostEqual(metrics["rollouts/truncated_frac"], 0.5)


class EvalLauncher(unittest.TestCase):
    def make(self, root: Path, steps, submitted=()):
        for step in steps:
            path = root / f"checkpoint-{step}"
            path.mkdir()
            for name in ("trainer_state.json", "config.json"):
                (path / name).write_text("{}")
            (path / "model.safetensors").write_text("")
        with (root / "checkpoint_evals.jsonl").open("w") as handle:
            for step in submitted:
                handle.write(json.dumps({"step": step, "dry_run": False}) + "\n")

    def launched(self, root: Path, final: bool):
        from checkpoint_evals import CheckpointEvals

        evals = CheckpointEvals(root, bucket="o/b", model="m", sandbox="hf-sandbox", rounds=0, max_tokens=8,
                                space_id=None, trackio_bucket=None, dry_run=True, eval_every=20)
        sent = []
        evals.launch = lambda path, step: sent.append(step)
        evals.scan(final=final)
        return sent

    def test_eval_every_and_the_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make(root, (10, 20, 30, 40, 50))
            self.assertEqual(self.launched(root, final=False), [20, 40])
            self.assertEqual(self.launched(root, final=True), [20, 40, 50])

    def test_a_resumed_run_does_not_resend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make(root, (20, 40), submitted=(20,))
            self.assertEqual(self.launched(root, final=False), [40])


class EvalSummary(unittest.TestCase):
    def test_a_problem_without_a_grade_stays_in_pass_at_k(self):
        from evaluate import summarize

        rows = [{"task": "a", "sample": 0, "reward": 1.0}, {"task": "a", "sample": 1, "reward": 0.2},
                {"task": "b", "sample": 0, "reward": 0.5}, {"task": "b", "sample": 1, "reward": None},
                {"task": "c", "sample": 0, "reward": None}, {"task": "c", "sample": 1, "reward": None}]
        summary = summarize(rows, expected=6, tasks=["a", "b", "c"], samples=2)
        self.assertEqual((summary["graded"], summary["problems"], summary["problems_expected"]), (3, 2, 3))
        self.assertAlmostEqual(summary["pass_at_k_full_marks"], 1 / 3)  # not 1/2: c counts, as a failure
        self.assertAlmostEqual(summary["best_of_k_mean"], (1.0 + 0.5 + 0.0) / 3)
        self.assertAlmostEqual(summary["mean_reward"], (1.0 + 0.2 + 0.5) / 3)  # over graded attempts, with coverage

    def test_a_partial_eval_is_not_charted(self):
        import evaluate

        args = types.SimpleNamespace()
        for graded, charted in ((3, False), (4, True)):
            with tempfile.TemporaryDirectory() as directory, mock.patch.object(evaluate, "report") as report, \
                    contextlib.redirect_stdout(io.StringIO()):
                evaluate.publish(args, Path(directory), {"expected": 4, "graded": graded})
                written = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual((report.called, written["partial"]), (charted, not charted))


def _stub_fineproof_harbor(train, removed, test, raw):
    """fineproof_harbor's data and adapter as prepare.py uses them, over in-memory rows."""
    package = types.ModuleType("fineproof_harbor")
    data = types.ModuleType("fineproof_harbor.data")
    data.fetch = lambda cache: {"train": "train.parquet", "sft": "sft.parquet"}
    data.load_all = lambda paths: {"train": train, "test": test, "removed": removed}
    data.reference_proofs = lambda path: {}
    data.fineproofs = lambda path, split, refs: raw
    adapter = types.ModuleType("fineproof_harbor.adapter")

    def write_task(row, out):
        task = out / "tasks" / row["id"]
        (task / "environment").mkdir(parents=True)
        (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
        (task / "task.toml").write_text('[agent]\nnetwork_mode = "no-network"\n\n[environment]\ncpus = 1\n')

    adapter.write_task = write_task
    package.data, package.adapter = data, adapter
    return {"fineproof_harbor": package, "fineproof_harbor.data": data, "fineproof_harbor.adapter": adapter}


class Prepare(unittest.TestCase):
    row = staticmethod(lambda name, mean=0.5: {"id": name, "reward_mean": mean})

    def test_listed_train_falls_back_to_raw_rows(self):
        from prepare import listed_train

        train, raw = [self.row("a"), self.row("b")], [self.row("a"), self.row("b"), self.row("dropped")]
        loads = []
        rows, fallback = listed_train(["b", "a"], train, [], {"t"}, lambda: loads.append(1) or raw)
        self.assertEqual(([r["id"] for r in rows], fallback, loads), (["b", "a"], [], []))  # raw rows never read
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rows, fallback = listed_train(["a", "dropped"], train,
                                          [{"id": "dropped", "reason": "near-duplicate of a train problem",
                                            "matches": "a"}], {"t"}, lambda: raw)
        self.assertEqual(([r["id"] for r in rows], fallback), (["a", "dropped"], ["dropped"]))
        self.assertIn("dropped", err.getvalue())
        for names in (["t"], ["a", "a"], ["nowhere"]):  # a test problem, a repeat, an unknown id
            with self.assertRaises(ValueError), contextlib.redirect_stderr(io.StringIO()):
                listed_train(names, train, [], {"t"}, lambda: raw)

    def test_train_list_and_the_hf_sandbox_warning(self):
        import prepare

        train = [self.row("a"), self.row("b"), self.row("c", 0.95)]
        raw = train + [self.row("dropped")]
        removed = [{"id": "dropped", "reason": "near-duplicate of a train problem", "matches": "a"}]
        with tempfile.TemporaryDirectory() as directory:
            out, listing = Path(directory) / "prepared", Path(directory) / "list.txt"
            listing.write_text("# the run's problems\nc\n\ndropped\n")
            argv = ["prepare.py", "--out", str(out), "--train-list", str(listing), "--sandbox", "hf-sandbox"]
            with mock.patch.dict(sys.modules, _stub_fineproof_harbor(train, removed, [self.row("t")], raw)), \
                    mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                prepare.main()
            ready = json.loads((out / "ready.json").read_text())
            built = sorted(p.name for p in (out / "datasets" / "train" / "tasks").iterdir())
            toml = (out / "datasets" / "train" / "tasks" / "c" / "task.toml").read_text()
        self.assertEqual(built, ["c", "dropped"])  # exactly the list, outside the reward_mean filter too
        self.assertEqual((ready["selection"]["raw_fallback"], ready["agent_network"]), (["dropped"], "public"))
        self.assertIn('network_mode = "public"', toml)
        self.assertIn("WARNING: dropped was dropped from train", err.getvalue())
        self.assertIn("UNSAFE to reuse with an agent installed in the sandbox", err.getvalue())


class CheckpointCopies(unittest.TestCase):
    """jobs/run.py: what --resume-from copies, what a resumed Job restores, and which secrets each child gets."""

    def fake_sync(self, bucket_root: Path, calls: list):
        """`hf buckets sync` from a local folder standing in for the bucket, honouring --exclude like the CLI."""

        def run(command, **kwargs):
            calls.append(command)
            if command[:3] != ["hf", "buckets", "sync"]:
                return types.SimpleNamespace(returncode=0, stdout="")
            source = bucket_root / command[3].removeprefix("hf://buckets/o/b/").strip("/")
            target = Path(command[4])
            excludes = [command[i + 1] for i, arg in enumerate(command) if arg == "--exclude"]
            for path in source.rglob("*"):
                relative = path.relative_to(source).as_posix()
                if path.is_file() and not any(fnmatch(relative, pattern) for pattern in excludes):
                    (target / relative).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target / relative)
            return types.SimpleNamespace(returncode=0, stdout="")

        return run

    def checkpoint(self, folder: Path, state=("optimizer.pt", "scheduler.pt", "rng_state.pth")):
        folder.mkdir(parents=True)
        for name in ("config.json", "model.safetensors", "trainer_state.json", *state):
            (folder / name).write_text("{}")

    def test_resume_from_keeps_the_training_state(self):
        import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.checkpoint(root / "bucket" / "old" / "checkpoint-40")
            self.checkpoint(root / "bucket" / "weights" / "checkpoint-40", state=())
            (root / "out").mkdir()
            (root / "eval").mkdir()
            calls = []
            with mock.patch.dict(os.environ, {"FINEPROOF_OUTPUT_BUCKET": "o/b"}), \
                    mock.patch.object(run, "CHECKPOINTS_ROOT", root / "local"), \
                    mock.patch.object(run.subprocess, "run", self.fake_sync(root / "bucket", calls)):
                local = Path(run.local_checkpoint("/outputs/old/checkpoint-40", root / "out", training=True))
                self.assertTrue((local / "optimizer.pt").is_file() and (local / "rng_state.pth").is_file())
                self.assertNotIn("--exclude", calls[-1])
                evaluated = Path(run.local_checkpoint("/outputs/old/checkpoint-40", root / "eval"))
                self.assertIn("--exclude", calls[-1])  # an eval copies the weights only
                self.assertTrue((evaluated / "model.safetensors").is_file())
                with self.assertRaisesRegex(RuntimeError, "optimizer, scheduler, rng_state"):
                    run.local_checkpoint("/outputs/weights/checkpoint-40", root / "out", training=True)
            with self.assertRaisesRegex(RuntimeError, "optimizer"):  # a local folder (Slurm) is checked too
                run.local_checkpoint(str(root / "bucket" / "weights" / "checkpoint-40"), root / "out", training=True)

    def test_a_resumed_job_keeps_its_own_dependencies(self):
        import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "bucket" / "fp-run"
            self.checkpoint(previous / "checkpoint-20")
            for name, text in (("dependencies.json", "first Job"), ("packages.txt", "first Job"),
                               ("rollouts.jsonl", "earlier rollouts"), ("trials/t/result.json", "{}")):
                (previous / name).parent.mkdir(parents=True, exist_ok=True)
                (previous / name).write_text(text)
            output = root / "outputs" / "fp-run"
            output.mkdir(parents=True)
            for name in ("dependencies.json", "packages.txt"):
                (output / name).write_text("this Job")
            with mock.patch.dict(os.environ, {"FINEPROOF_OUTPUT_BUCKET": "o/b"}), \
                    mock.patch.object(run, "OUTPUTS_ROOT", root / "outputs"), \
                    mock.patch.object(run, "remote_checkpoint", lambda url: "checkpoint-20"), \
                    mock.patch.object(run.subprocess, "run", self.fake_sync(root / "bucket", [])):
                restored = run.restore_run(output)
            self.assertEqual(restored, output / "checkpoint-20")
            self.assertEqual((output / "rollouts.jsonl").read_text(), "earlier rollouts")
            self.assertFalse((output / "trials").exists())
            for name in ("dependencies.json", "packages.txt"):
                self.assertEqual((output / name).read_text(), "this Job")

    def test_children_get_only_their_secrets(self):
        import run

        env = {"PATH": "/bin", "HF_TOKEN": "h", "JUDGE_API_KEY": "j", "DAYTONA_API_KEY": "d", "JUDGE_SAMPLES": "2"}
        self.assertEqual(run.child_env(env, "harbor"), env)
        for child in ("vllm", "train", "eval"):
            self.assertEqual(run.child_env(env, child), {"PATH": "/bin", "HF_TOKEN": "h", "JUDGE_SAMPLES": "2"})
        self.assertEqual(run.child_env(env, "other"), {"PATH": "/bin", "JUDGE_SAMPLES": "2"})


if __name__ == "__main__":
    unittest.main()
