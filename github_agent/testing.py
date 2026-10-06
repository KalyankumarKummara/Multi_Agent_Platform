"""Deterministic fake client for GitHub Agent tests and local examples."""

from typing import Any


class MockGitHubAuthProvider:
    """Supplies a harmless test marker, never a real credential."""

    def __init__(self, token: str | None = "test-token") -> None:
        self.token = token

    def get_token(self) -> str | None:
        return self.token


class MockGitHubClient:
    """Small deterministic fake implementing the explicit client operations."""

    def __init__(self, data: dict[str, Any] | None = None, fail_operations: set[str] | None = None) -> None:
        self.data = data or {
            "get_repository": {"name": "demo", "full_name": "octo/demo"},
            "list_repositories": [{"name": "demo", "full_name": "octo/demo"}],
            "get_repository_activity": [],
            "list_pull_requests": [],
            "get_pull_request": {"number": 7, "title": "Example PR"},
            "get_pull_request_files": [],
            "get_pull_request_commits": [],
            "get_pull_request_reviews": [{"state": "APPROVED"}],
            "get_pull_request_review_comments": [],
            "list_issues": [],
            "get_issue": {"number": 8, "title": "Example issue"},
            "get_issue_comments": [],
            "list_commits": [],
            "get_commit": {"sha": "abc123", "message": "Example commit"},
        }
        self.fail_operations = fail_operations or set()
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, operation: str):
        if operation not in self.data:
            raise AttributeError(operation)

        def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((operation, args, kwargs))
            if operation in self.fail_operations:
                raise RuntimeError("simulated GitHub client failure")
            return self.data[operation]

        return call
