"""Decontamination of the train set: solution text in statements, test and benchmark overlap, train duplicates."""

import unittest

from fineproof_harbor import data


def row(id_, problem, set_="train"):
    return {"id": id_, "set": set_, "problem": problem}


STATEMENT = ("Let P be a polynomial in three variables with real coefficients such that P(x, y, z) = P(x, y, xy - z) "
             "for all real x, y and z. Prove that there exists a polynomial F in one variable with the stated form.")


class StripSolutionLeak(unittest.TestCase):
    def test_cuts_at_common_remarks(self):
        self.assertEqual(data.strip_solution_leak(STATEMENT + " (Russia) Common remarks. The polynomial works."),
                         STATEMENT + " (Russia)")

    def test_cuts_at_a_solution_heading(self):
        text = "a) Prove that x > 8.\n\nb) Prove the second part.\n\n## a) Solution 1\n\nLet x be ..."
        self.assertEqual(data.strip_solution_leak(text), "a) Prove that x > 8.\n\nb) Prove the second part.")

    def test_leaves_a_statement_that_mentions_solutions(self):
        text = "Find all solutions of the equation x^2 = 2^x in positive integers, and prove there are no others."
        self.assertEqual(data.strip_solution_leak(text), text)


class Decontaminate(unittest.TestCase):
    def test_drops_test_ids_near_copies_and_duplicates(self):
        test = [row("t-1", STATEMENT, "test")]
        bench = [row("pb-x", "For nonnegative integers a and b denote their bitwise xor by a xor b and find every a "
                             "such that x xor ax differs from y xor ay for all x > y >= 0.", "proofbench")]
        train = [
            row("t-1", STATEMENT),                                                   # the test problem itself
            row("near-test", STATEMENT + " Show also that F is unique."),            # contains the test problem
            row("aops-dup", "Prove that the sum of the first n odd numbers is n squared, for every n >= 1."),
            row("olympiads-dup", "Prove that the sum of the first n odd numbers is n squared, for every n >= 1."),
            row("keep", "Prove that among any five points in a unit square two lie within distance one of each other."),
        ]
        kept, removed = data.decontaminate(train, test, bench)
        self.assertEqual([r["id"] for r in kept], ["aops-dup", "keep"])
        reasons = {r["id"]: r["reason"] for r in removed}
        self.assertEqual(reasons["t-1"], "test problem")
        self.assertEqual(reasons["near-test"], "near-copy of a test problem")
        self.assertEqual(reasons["olympiads-dup"], "duplicate of a train problem")

    def test_known_paraphrase_is_dropped(self):
        kept, removed = data.decontaminate([row("aops-c8f456a645", "Bitwise xor, written differently.")], [], [])
        self.assertEqual(kept, [])
        self.assertEqual(removed[0]["matches"], "pb-tst-2023-4")

    def test_near_identical_train_pair_keeps_the_smaller_id(self):
        base = "Prove that for every positive integer n the number n^5 - n is divisible by 30, using Fermat's theorem."
        kept, removed = data.decontaminate([row("b-2", base + " Justify every step."), row("a-1", base)], [], [])
        self.assertEqual([r["id"] for r in kept], ["a-1"])
        self.assertEqual(removed[0]["reason"], "near-duplicate of a train problem")


if __name__ == "__main__":
    unittest.main()
