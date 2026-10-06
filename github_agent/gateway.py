"""Narrow domain gateway between GitHub tools and the injected API client."""

from typing import Any

from shared.agent_core.errors import AgentError, ErrorCategory

from .github_client import GitHubClient


class GitHubToolGateway:
    """Expose only the approved, read-only GitHub operations.

    Purpose-specific methods keep tools from constructing arbitrary URLs and
    provide one place to normalize inputs and remove sensitive response keys.
    """

    _SENSITIVE_KEYS = {"authorization", "access_token", "token", "password", "client_secret"}

    def __init__(self, client: GitHubClient) -> None:
        self._client = client

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: cls._sanitize(item)
                for key, item in value.items()
                if str(key).lower() not in cls._SENSITIVE_KEYS
            }
        if isinstance(value, list):
            return [cls._sanitize(item) for item in value]
        return value

    @staticmethod
    def _name(value: str, field: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise AgentError(
                code="INVALID_GITHUB_INPUT",
                message=f"{field} must not be empty.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": field},
            )
        return normalized

    @staticmethod
    def _number(value: int, field: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise AgentError(
                code="INVALID_GITHUB_INPUT",
                message=f"{field} must be a positive integer.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": field},
            )
        return value

    @staticmethod
    def _options(options: dict[str, Any], *allowed: str) -> dict[str, Any]:
        unexpected = set(options) - set(allowed)
        if unexpected:
            field = sorted(unexpected)[0]
            raise AgentError(
                code="UNEXPECTED_FIELD",
                message=f"Unexpected GitHub read option: {field}",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": field},
            )
        normalized = dict(options)
        for key in ("page", "per_page"):
            if key in normalized:
                normalized[key] = GitHubToolGateway._number(normalized[key], key)
                if key == "per_page" and normalized[key] > 100:
                    raise AgentError(
                        code="INVALID_GITHUB_INPUT",
                        message="per_page must not exceed 100.",
                        category=ErrorCategory.VALIDATION,
                        retryable=False,
                        details={"field": key},
                    )
        for key, value in normalized.items():
            if key not in ("page", "per_page"):
                if not isinstance(value, str):
                    raise AgentError(
                        code="INVALID_FIELD_TYPE",
                        message=f"{key} must be a string.",
                        category=ErrorCategory.VALIDATION,
                        retryable=False,
                        details={"field": key, "expected_type": "string"},
                    )
                normalized[key] = GitHubToolGateway._name(value, key)
        return normalized

    def _repo_args(self, owner: str, repo: str) -> tuple[str, str]:
        return self._name(owner, "owner"), self._name(repo, "repo")

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._sanitize(getattr(self._client, method)(*args, **kwargs))
        except AgentError:
            raise
        except Exception as error:
            # Do not echo exception text: HTTP libraries may include request data.
            raise AgentError(
                code="GITHUB_CLIENT_ERROR",
                message="The GitHub client failed to complete the read operation.",
                category=ErrorCategory.EXTERNAL_SERVICE,
                retryable=True,
                details={"operation": method},
            ) from None

    def get_repository(self, owner: str, repo: str) -> Any:
        return self._call("get_repository", *self._repo_args(owner, repo))

    def list_repositories(self, username: str | None = None, organization: str | None = None, **options: Any) -> Any:
        if username and organization:
            raise AgentError("INVALID_GITHUB_INPUT", "Specify username or organization, not both.", ErrorCategory.VALIDATION, False)
        username = self._name(username, "username") if username else None
        organization = self._name(organization, "organization") if organization else None
        options = self._options(options, "type", "sort", "per_page", "page")
        return self._call("list_repositories", username, organization, **options)

    def get_repository_activity(self, owner: str, repo: str, **options: Any) -> Any:
        options = self._options(options, "per_page", "page")
        return self._call("get_repository_activity", *self._repo_args(owner, repo), **options)

    def list_pull_requests(self, owner: str, repo: str, **options: Any) -> Any:
        options = self._options(options, "state", "per_page", "page", "base", "head", "sort", "direction")
        return self._call("list_pull_requests", *self._repo_args(owner, repo), **options)

    def get_pull_request(self, owner: str, repo: str, pull_number: int) -> Any:
        owner, repo = self._repo_args(owner, repo)
        return self._call("get_pull_request", owner, repo, self._number(pull_number, "pull_number"))

    def get_pull_request_files(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any:
        owner, repo = self._repo_args(owner, repo)
        options = self._options(options, "per_page", "page")
        return self._call("get_pull_request_files", owner, repo, self._number(pull_number, "pull_number"), **options)

    def get_pull_request_commits(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any:
        owner, repo = self._repo_args(owner, repo)
        options = self._options(options, "per_page", "page")
        return self._call("get_pull_request_commits", owner, repo, self._number(pull_number, "pull_number"), **options)

    def get_pull_request_reviews(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any:
        owner, repo = self._repo_args(owner, repo)
        options = self._options(options, "per_page", "page")
        return self._call("get_pull_request_reviews", owner, repo, self._number(pull_number, "pull_number"), **options)

    def get_pull_request_review_comments(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any:
        owner, repo = self._repo_args(owner, repo)
        options = self._options(options, "per_page", "page")
        return self._call("get_pull_request_review_comments", owner, repo, self._number(pull_number, "pull_number"), **options)

    def list_issues(self, owner: str, repo: str, **options: Any) -> Any:
        options = self._options(options, "state", "per_page", "page", "labels", "sort", "direction", "since")
        return self._call("list_issues", *self._repo_args(owner, repo), **options)

    def get_issue(self, owner: str, repo: str, issue_number: int) -> Any:
        owner, repo = self._repo_args(owner, repo)
        return self._call("get_issue", owner, repo, self._number(issue_number, "issue_number"))

    def get_issue_comments(self, owner: str, repo: str, issue_number: int, **options: Any) -> Any:
        owner, repo = self._repo_args(owner, repo)
        options = self._options(options, "per_page", "page")
        return self._call("get_issue_comments", owner, repo, self._number(issue_number, "issue_number"), **options)

    def list_commits(self, owner: str, repo: str, **options: Any) -> Any:
        options = self._options(options, "sha", "path", "author", "since", "until", "per_page", "page")
        return self._call("list_commits", *self._repo_args(owner, repo), **options)

    def get_commit(self, owner: str, repo: str, commit_sha: str) -> Any:
        owner, repo = self._repo_args(owner, repo)
        return self._call("get_commit", owner, repo, self._name(commit_sha, "commit_sha"))
