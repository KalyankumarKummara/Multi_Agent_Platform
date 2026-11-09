"""Deterministic tests for opt-in intent evaluation metrics."""

import unittest

from groq_intent_evaluation_metrics import (
    calculate_classifier_metrics,
    classification_matches_expectation,
    matching_arguments,
)


class TestExactArgumentMatching(unittest.TestCase):
    def test_exact_matching_arguments(self):
        self.assertTrue(matching_arguments({"owner": "octo", "repo": "demo"},
                                           {"owner": "octo", "repo": "demo"}))

    def test_missing_arguments_do_not_match(self):
        self.assertFalse(matching_arguments(
            {"owner": "octo", "repo": "demo"}, {"owner": "octo"}
        ))

    def test_unexpected_extra_arguments_do_not_match(self):
        self.assertFalse(matching_arguments(
            {"owner": "octo", "repo": "demo"},
            {"owner": "octo", "repo": "demo", "debug": True},
        ))

    def test_incorrect_argument_values_do_not_match(self):
        self.assertFalse(matching_arguments(
            {"pull_number": 6}, {"pull_number": 7}
        ))

    def test_argument_values_must_have_the_expected_type(self):
        self.assertFalse(matching_arguments({"pull_number": 1}, {"pull_number": True}))

    def test_empty_arguments_match_only_empty_arguments(self):
        self.assertTrue(matching_arguments({}, {}))
        self.assertFalse(matching_arguments({}, {"unexpected": "value"}))
        self.assertFalse(matching_arguments({"required": 1}, {}))

    def test_correct_arguments_with_wrong_tool_fail_overall_match(self):
        expected = {
            "intent": "github_read", "tool_id": "get_repository",
            "arguments": {"owner": "octo", "repo": "demo"},
            "clarification": False,
        }
        actual = {
            "parse_status": "valid", "intent": "github_read",
            "tool_id": "list_branches",
            "arguments": {"owner": "octo", "repo": "demo"},
            "clarification": None,
        }
        self.assertFalse(classification_matches_expectation(expected, actual))

    def test_tool_and_argument_accuracy_are_separate_and_wrong_tool_gets_no_credit(self):
        results = [{
            "expected": {
                "intent": "github_read", "tool_id": "get_repository",
                "arguments": {"owner": "octo", "repo": "demo"},
                "clarification": False,
            },
            "classifier_output": {
                "parse_status": "valid", "intent": "github_read",
                "tool_id": "list_branches",
                "arguments": {"owner": "octo", "repo": "demo"},
                "clarification": None,
            },
        }]
        metrics = calculate_classifier_metrics(results)
        self.assertEqual(metrics["tool_selection_accuracy"], {
            "numerator": 0, "denominator": 1, "rate": 0.0,
        })
        self.assertEqual(metrics["exact_argument_set_accuracy"], {
            "numerator": 0, "denominator": 1, "rate": 0.0,
        })
        self.assertEqual(metrics["exact_argument_accuracy_given_correct_tool"], {
            "numerator": 0, "denominator": 0, "rate": None,
        })


if __name__ == "__main__":
    unittest.main()
