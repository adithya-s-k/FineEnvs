"""The training pieces that need no GPU, TRL or OpenEnv: the objective's advantage, the cut-off check, the rollout
metrics, and which checkpoints the eval launcher sends.

    python -m unittest discover -s tests
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "jobs")]


def _stub_trl():
    """objective.py subclasses TRL classes at import; stand-ins are enough for its pure function."""
    base = types.ModuleType("trl.experimental.async_grpo")
    base.AsyncGRPOTrainer = type("AsyncGRPOTrainer", (), {})
    harness = types.ModuleType("trl.experimental.async_grpo.openenv_harness")
    harness.HarnessRolloutWorker = type("HarnessRolloutWorker", (), {})
    harness._HarnessRolloutLoop = type("_HarnessRolloutLoop", (), {})
    for name, module in (("trl", types.ModuleType("trl")), ("trl.experimental", types.ModuleType("trl.experimental")),
                         ("trl.experimental.async_grpo", base),
                         ("trl.experimental.async_grpo.openenv_harness", harness)):
        sys.modules.setdefault(name, module)


class Objective(unittest.TestCase):
    def test_mean_centered_undoes_the_std_division(self):
        _stub_trl()
        from objective import mean_centered

        rewards = [0.5, 0.52, 0.48, 0.5]
        mean = sum(rewards) / 4
        std = (sum((r - mean) ** 2 for r in rewards) / 4) ** 0.5
        for r in rewards:
            trl_advantage = (r - mean) / (std + 1e-8)
            self.assertAlmostEqual(mean_centered(trl_advantage, std), r - mean, places=9)
        self.assertEqual(mean_centered(0.0, float("nan")), 0.0)  # an unscored rollout keeps TRL's 0
        self.assertEqual(mean_centered(0.0, None), 0.0)


class CutOff(unittest.TestCase):
    def test_only_the_final_call_counts(self):
        from tasks import final_call_truncated

        turn = lambda reason: types.SimpleNamespace(finish_reason=reason)  # noqa: E731
        self.assertTrue(final_call_truncated(types.SimpleNamespace(turns=[turn("length")])))
        self.assertTrue(final_call_truncated(types.SimpleNamespace(turns=[turn("stop"), turn("length")])))
        self.assertFalse(final_call_truncated(types.SimpleNamespace(turns=[turn("length"), turn("stop")])))
        self.assertFalse(final_call_truncated(types.SimpleNamespace(turns=[])))
        self.assertFalse(final_call_truncated(types.SimpleNamespace(turns=None)))


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


if __name__ == "__main__":
    unittest.main()
