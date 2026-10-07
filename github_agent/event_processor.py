"""Deterministic, informational interpretation of normalized GitHub events."""

from dataclasses import dataclass
from typing import Literal

from .webhook import NormalizedGitHubEvent


Significance = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class EventProcessingResult:
    """Read-only decision summary for one normalized GitHub event."""

    event_id: str
    event_type: str
    classification: str
    significance: Significance
    requires_action: bool
    recommended_action: str | None
    status: str
    reason: str


class GitHubEventProcessor:
    """Classify events and assess significance without executing actions."""

    CLASSIFICATIONS = {
        "repository": "repository",
        "push": "push",
        "create": "branch_or_tag",
        "pull_request": "pull_request",
        "pull_request_review": "pull_request_review",
        "pull_request_review_comment": "pull_request_review_comment",
        "issues": "issue",
        "issue_comment": "issue_comment",
    }

    def process(self, event: NormalizedGitHubEvent) -> EventProcessingResult:
        """Return a deterministic decision using normalized fields only."""
        if not isinstance(event, NormalizedGitHubEvent):
            raise TypeError("GitHubEventProcessor requires a NormalizedGitHubEvent.")

        classification = self.CLASSIFICATIONS.get(event.event_type)
        if classification is None:
            raise ValueError("Unsupported normalized GitHub event type.")

        action = event.action if isinstance(event.action, str) else None
        metadata = event.metadata if isinstance(event.metadata, dict) else {}
        significance: Significance = "low"
        reason = "GitHub activity recorded."

        if event.event_type == "repository":
            if action == "deleted":
                significance, reason = "high", "Repository deleted."
            elif action == "archived":
                significance, reason = "high", "Repository archived."
            elif action == "created":
                significance, reason = "medium", "Repository created."
            else:
                reason = "Repository activity recorded."

        elif event.event_type == "push":
            significance, reason = "low", "Push activity recorded."

        elif event.event_type == "create":
            ref_type = metadata.get("ref_type")
            if ref_type == "branch":
                significance, reason = "medium", "Branch created."
            elif ref_type == "tag":
                significance, reason = "medium", "Tag created."
            else:
                reason = "Reference creation activity recorded."

        elif event.event_type == "pull_request":
            pull_request = metadata.get("pull_request")
            merged = pull_request.get("merged") if isinstance(pull_request, dict) else None
            if action == "opened":
                significance, reason = "high", "Pull request opened."
            elif action == "reopened":
                significance, reason = "high", "Pull request reopened."
            elif action == "closed" and merged is True:
                significance, reason = "high", "Pull request merged."
            elif action == "closed" and merged is False:
                significance, reason = "medium", "Pull request closed without merging."
            else:
                reason = "Pull request activity recorded."

        elif event.event_type == "pull_request_review":
            if action == "submitted":
                significance, reason = "high", "Pull request review submitted."
            else:
                reason = "Pull request review activity recorded."

        elif event.event_type == "pull_request_review_comment":
            if action == "created":
                significance, reason = "medium", "Pull request review comment created."
            else:
                reason = "Pull request review comment activity recorded."

        elif event.event_type == "issues":
            if action == "opened":
                significance, reason = "high", "Issue opened."
            elif action == "reopened":
                significance, reason = "high", "Issue reopened."
            else:
                reason = "Issue activity recorded."

        elif event.event_type == "issue_comment":
            if action == "created":
                significance, reason = "medium", "Issue comment created."
            else:
                reason = "Issue comment activity recorded."

        return EventProcessingResult(
            event_id=event.event_id,
            event_type=event.event_type,
            classification=classification,
            significance=significance,
            requires_action=False,
            recommended_action=None,
            status="processed",
            reason=reason,
        )
