import unittest

from github_agent.event_processor import GitHubEventProcessor
from github_agent.webhook import NormalizedGitHubEvent


def make_event(event_type, action=None, metadata=None):
    return NormalizedGitHubEvent(
        event_id="delivery-test",
        source="github",
        event_type=event_type,
        action=action,
        repository={"name": "demo", "full_name": "octo/demo"},
        actor={"login": "octocat"},
        occurred_at=None,
        metadata={} if metadata is None else metadata,
    )


class TestGitHubEventProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = GitHubEventProcessor()

    def assert_decision(self, event_type, action, metadata, significance, reason):
        result = self.processor.process(make_event(event_type, action, metadata))
        self.assertEqual(result.significance, significance)
        self.assertEqual(result.reason, reason)
        self.assertEqual(result.status, "processed")
        self.assertFalse(result.requires_action)
        self.assertIsNone(result.recommended_action)

    def test_repository_significance_rules(self):
        self.assert_decision("repository", "created", {}, "medium", "Repository created.")
        self.assert_decision("repository", "deleted", {}, "high", "Repository deleted.")
        self.assert_decision("repository", "archived", {}, "high", "Repository archived.")

    def test_push_is_low_significance(self):
        self.assert_decision("push", None, {}, "low", "Push activity recorded.")

    def test_create_branch_and_tag_are_medium(self):
        branch = self.processor.process(make_event("create", metadata={"ref_type": "branch"}))
        tag = self.processor.process(make_event("create", metadata={"ref_type": "tag"}))

        self.assertEqual(branch.classification, "branch_or_tag")
        self.assertEqual(branch.significance, "medium")
        self.assertEqual(branch.reason, "Branch created.")
        self.assertEqual(tag.classification, "branch_or_tag")
        self.assertEqual(tag.significance, "medium")
        self.assertEqual(tag.reason, "Tag created.")

    def test_pull_request_significance_rules(self):
        self.assert_decision("pull_request", "opened", {}, "high", "Pull request opened.")
        self.assert_decision("pull_request", "reopened", {}, "high", "Pull request reopened.")
        self.assert_decision(
            "pull_request", "closed", {"pull_request": {"merged": True}},
            "high", "Pull request merged.",
        )
        self.assert_decision(
            "pull_request", "closed", {"pull_request": {"merged": False}},
            "medium", "Pull request closed without merging.",
        )

    def test_review_and_review_comment_rules(self):
        self.assert_decision(
            "pull_request_review", "submitted", {},
            "high", "Pull request review submitted.",
        )
        self.assert_decision(
            "pull_request_review_comment", "created", {},
            "medium", "Pull request review comment created.",
        )

    def test_issue_and_issue_comment_rules(self):
        self.assert_decision("issues", "opened", {}, "high", "Issue opened.")
        self.assert_decision("issues", "reopened", {}, "high", "Issue reopened.")
        self.assert_decision(
            "issue_comment", "created", {},
            "medium", "Issue comment created.",
        )

    def test_all_supported_classifications_and_result_fields(self):
        expected = {
            "repository": "repository",
            "push": "push",
            "create": "branch_or_tag",
            "pull_request": "pull_request",
            "pull_request_review": "pull_request_review",
            "pull_request_review_comment": "pull_request_review_comment",
            "issues": "issue",
            "issue_comment": "issue_comment",
        }
        for event_type, classification in expected.items():
            with self.subTest(event_type=event_type):
                action = "opened" if event_type in {"pull_request", "issues"} else None
                result = self.processor.process(make_event(event_type, action))
                self.assertEqual(result.event_id, "delivery-test")
                self.assertEqual(result.event_type, event_type)
                self.assertEqual(result.classification, classification)
                self.assertEqual(result.status, "processed")
                self.assertIn(result.significance, {"low", "medium", "high"})
                self.assertFalse(result.requires_action)
                self.assertIsNone(result.recommended_action)
                self.assertTrue(result.reason)
                self.assertLess(len(result.reason), 80)

    def test_missing_optional_metadata_falls_back_safely(self):
        closed_pr = self.processor.process(make_event("pull_request", "closed", {}))
        create_without_ref_type = self.processor.process(make_event("create", metadata=None))

        self.assertEqual(closed_pr.significance, "low")
        self.assertEqual(closed_pr.reason, "Pull request activity recorded.")
        self.assertEqual(create_without_ref_type.significance, "low")
        self.assertEqual(create_without_ref_type.reason, "Reference creation activity recorded.")

    def test_unsupported_event_is_rejected(self):
        with self.assertRaises(ValueError):
            self.processor.process(make_event("ping"))


if __name__ == "__main__":
    unittest.main()
