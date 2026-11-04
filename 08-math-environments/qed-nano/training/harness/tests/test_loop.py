"""The qed-proof loop against a stub model.

    python -m unittest discover -s harness/tests
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qed_proof_agent import loop  # noqa: E402


def stub(replies):
    calls = []

    async def complete(messages):
        calls.append([dict(m) for m in messages])
        return replies[len(calls) - 1]

    return complete, calls


class Loop(unittest.TestCase):
    def test_final_proof_comes_from_the_last_heading(self):
        complete, calls = stub(["First proof.", "Gap in step 2.\n\n## Final proof\n\nCorrected proof."])
        proof, messages = asyncio.run(loop.run(complete, "Prove X.", rounds=1))
        self.assertEqual(proof, "Corrected proof.")
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user", "assistant"])

    def test_conversation_is_append_only(self):
        complete, calls = stub(["a", "b\n## Final proof\nc", "d\n## Final proof\ne"])
        asyncio.run(loop.run(complete, "Prove X.", rounds=2))
        for earlier, later in zip(calls, calls[1:]):
            self.assertEqual(later[: len(earlier)], earlier)  # every call's prompt extends the previous one

    def test_round_without_heading_keeps_previous_proof(self):
        complete, _ = stub(["First proof.", "Only a critique, no heading."])
        proof, _ = asyncio.run(loop.run(complete, "Prove X.", rounds=1))
        self.assertEqual(proof, "First proof.")

    def test_zero_rounds_grades_the_first_reply(self):
        complete, calls = stub(["Just a proof."])
        proof, _ = asyncio.run(loop.run(complete, "Prove X.", rounds=0))
        self.assertEqual((proof, len(calls)), ("Just a proof.", 1))

    def test_heading_variants(self):
        for heading in ("## Final proof", "### Final Proof:", "# FINAL PROOF"):
            self.assertEqual(loop.extract_final(f"x\n{heading}\nbody"), "body")
        self.assertIsNone(loop.extract_final("The final proof is below."))
        self.assertIsNone(loop.extract_final("## Final proof\n   "))


if __name__ == "__main__":
    unittest.main()
