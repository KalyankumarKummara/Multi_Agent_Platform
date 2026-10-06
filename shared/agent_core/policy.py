from dataclasses import dataclass
from enum import Enum
from typing import Any


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class PolicyResult:
    decision: PolicyDecision
    reason: str


class PolicyEngine:

    def evaluate(self, action: str, context: Any) -> PolicyResult:

        github_read_actions = {
            "get_repository",
            "list_repositories",
            "get_repository_activity",
            "list_pull_requests",
            "get_pull_request",
            "get_pull_request_files",
            "get_pull_request_commits",
            "get_pull_request_reviews",
            "get_pull_request_review_comments",
            "list_issues",
            "get_issue",
            "get_issue_comments",
            "list_commits",
            "get_commit",
        }

        if action == "read_repository" or action in github_read_actions:
            return PolicyResult(
                decision=PolicyDecision.ALLOW,
                reason="Reading GitHub repository information is allowed."
            )

        if action == "merge_pull_request":
            return PolicyResult(
                decision=PolicyDecision.REQUIRE_APPROVAL,
                reason="Merging a pull request requires human approval."
            )

        if action == "delete_repository":
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="Repository deletion is not allowed."
            )

        return PolicyResult(
            decision=PolicyDecision.DENY,
            reason="Action is not recognized by the policy engine."
        )