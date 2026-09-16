import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from whitebox_tito import audit_rows


class WhiteboxProvenanceTest(unittest.TestCase):
    def test_identical_tokens_with_distinct_observed_logprobs(self):
        # Repeated GRPO samples can emit identical tokens with slightly different
        # floating-point logprobs. Matching the first token-identical call is wrong.
        calls = [{'prompt_ids': [1], 'completion_ids': [2], 'logprobs': [p]} for p in [-.1, -.10001]]
        result = audit_rows([[1], [1]], [[2], [2]], [[1], [1]], [[-.10001], [-.1]], calls)
        self.assertTrue(all(row['tito_pass'] for row in result))

    def test_one_call_cannot_prove_two_occurrences(self):
        calls = [{'prompt_ids': [1], 'completion_ids': [2], 'logprobs': [-.1]}]
        with self.assertRaisesRegex(AssertionError, 'distinct engine call'):
            audit_rows([[1], [1]], [[2], [2]], [[1], [1]], [[-.1], [-.1]], calls)

    def test_changed_probability_is_rejected(self):
        calls = [{'prompt_ids': [1], 'completion_ids': [2], 'logprobs': [-.1]}]
        with self.assertRaisesRegex(AssertionError, 'provenance'):
            audit_rows([[1]], [[2]], [[1]], [[-.2]], calls)

    def test_tool_context_and_budget_trim_preserve_exact_prefix(self):
        calls = [{'prompt_ids': [1], 'completion_ids': [2], 'logprobs': [-.1]},
                 {'prompt_ids': [1, 2, 3], 'completion_ids': [4, 5], 'logprobs': [-.2, -.3]}]
        result = audit_rows([[1]], [[2, 3, 4]], [[1, 0, 1]], [[-.1, 0., -.2]], calls)
        self.assertEqual(result[0]['supervised'], 2)

    def test_ambiguous_truncation_preserves_occurrences(self):
        calls = [{'prompt_ids': [1], 'completion_ids': [2, 3], 'logprobs': [-.1, -.2]},
                 {'prompt_ids': [1], 'completion_ids': [2], 'logprobs': [-.1]}]
        self.assertEqual(len(audit_rows([[1], [1]], [[2], [2, 3]], [[1], [1, 1]],
                                       [[-.1], [-.1, -.2]], calls)), 2)


if __name__ == '__main__': unittest.main()
