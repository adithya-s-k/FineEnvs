"""Task-record helpers that do not need the datasets.

    python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qed_nano_harbor.data import normalize, problem_id, strip_self_evaluation  # noqa: E402


class StripSelfEvaluation(unittest.TestCase):
    def test_cuts_deepseek_self_evaluation(self):
        proof = ("Solution text.\n\n\\[\\boxed{1}\\]\n\n---\n\n## Self Evaluation  \n\n"
                 "Here is my evaluation of the solution: ... the final overall score should be: \\boxed{1}")
        self.assertEqual(strip_self_evaluation(proof), "Solution text.\n\n\\[\\boxed{1}\\]")

    def test_heading_variants(self):
        for heading in ("## Self-Evaluation", "### self evaluation", "# Self Evaluation:"):
            self.assertEqual(strip_self_evaluation(f"Proof.\n{heading}\nGreat proof."), "Proof.")

    def test_leaves_other_proofs_alone(self):
        proof = "We evaluate the sum. A self evaluation in prose is not a heading.\n## Conclusion\nDone."
        self.assertEqual(strip_self_evaluation(proof), proof)


class Ids(unittest.TestCase):
    def test_id_ignores_whitespace_and_case(self):
        self.assertEqual(problem_id("aops", "Prove  that X."), problem_id("aops", "prove that x."))
        self.assertEqual(normalize(" A\n b "), "a b")


if __name__ == "__main__":
    unittest.main()
