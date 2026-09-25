"""Explicit zero tolerances must survive task parsing and actual native grading."""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest

PACKAGE = "comparison_native_tolerance"
package = ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[2] / "envs/blackbox-opencode")]
sys.modules[PACKAGE] = package
Task = importlib.import_module(PACKAGE + ".task").DataAgentTask
grade_rollout = importlib.import_module(PACKAGE + ".verifier").grade_rollout


class NativeToleranceTest(unittest.TestCase):
    def row(self, **values):
        return {"instruction": "Calculate the value", "answer": "1", "reward_mode": "numeric",
                "hf_bucket": "org/test", "bucket_prefix": "task", **values}

    def test_explicit_zero_is_strict_but_omitted_tolerance_defaults(self):
        strict = Task.from_row(self.row(atol=0.0, rtol="0.0"))
        default = Task.from_row(self.row(atol=None, rtol=""))
        self.assertEqual((strict.atol, strict.rtol), (0.0, 0.0))
        self.assertEqual((default.atol, default.rtol), (1e-3, 1e-3))
        read = lambda _: "1.0005"
        self.assertEqual(grade_rollout(strict, read, ("/answer",)).correctness, 0.0)
        self.assertEqual(grade_rollout(default, read, ("/answer",)).correctness, 1.0)

    def test_absolute_and_relative_tolerances_are_preserved_independently(self):
        narrow = Task.from_row(self.row(atol=1e-5, rtol=0))
        relative = Task.from_row(self.row(atol=0, rtol=1e-3))
        read = lambda _: "1.0005"
        self.assertEqual(grade_rollout(narrow, read, ("/answer",)).correctness, 0.0)
        self.assertEqual(grade_rollout(relative, read, ("/answer",)).correctness, 1.0)


if __name__ == "__main__":
    unittest.main()
