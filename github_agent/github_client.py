"""Explicit read-only GitHub client contract and standard-library adapter."""

from abc import ABC, abstractmethod
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from shared.agent_core.errors import AgentError, ErrorCategory

from .authentication import GitHubAuthProvider


class GitHubClient(ABC):
    """Domain operations supported by GitHub tools; no arbitrary API call."""

    @abstractmethod
    def get_repository(self, owner: str, repo: str) -> Any: ...

    @abstractmethod
    def list_repositories(self, username: str | None = None, organization: str | None = None, **options: Any) -> Any: ...

    @abstractmethod
    def get_repository_activity(self, owner: str, repo: str, **options: Any) -> Any: ...

    @abstractmethod
    def list_pull_requests(self, owner: str, repo: str, **options: Any) -> Any: ...

    @abstractmethod
    def get_pull_request(self, owner: str, repo: str, pull_number: int) -> Any: ...

    @abstractmethod
    def get_pull_request_files(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any: ...

    @abstractmethod
    def get_pull_request_commits(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any: ...

    @abstractmethod
    def get_pull_request_reviews(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any: ...

    @abstractmethod
    def get_pull_request_review_comments(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any: ...

    @abstractmethod
    def list_issues(self, owner: str, repo: str, **options: Any) -> Any: ...

    @abstractmethod
    def get_issue(self, owner: str, repo: str, issue_number: int) -> Any: ...

    @abstractmethod
    def get_issue_comments(self, owner: str, repo: str, issue_number: int, **options: Any) -> Any: ...

    @abstractmethod
    def list_commits(self, owner: str, repo: str, **options: Any) -> Any: ...

    @abstractmethod
    def get_commit(self, owner: str, repo: str, commit_sha: str) -> Any: ...


class GitHubRESTClient(GitHubClient):
    """Read-only GitHub REST adapter; all URLs are built by named methods."""

    API_ROOT = "https://api.github.com"

    def __init__(
        self,
        auth_provider: GitHubAuthProvider,
        *,
        timeout_seconds: float = 10.0,
        opener=urlopen,
        retry_delay_seconds: float = 0.2,
        sleeper=time.sleep,
    ) -> None:
        self._auth_provider = auth_provider
        self._timeout_seconds = timeout_seconds
        self._opener = opener
        self._retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._sleeper = sleeper

    @staticmethod
    def _segment(value: str) -> str:
        return quote(value, safe="")

    @staticmethod
    def _query(options: dict[str, Any]) -> str:
        filtered = {key: value for key, value in options.items() if value is not None}
        return "?" + urlencode(filtered) if filtered else ""

    def _get(self, path: str, **options: Any) -> Any:
        # This private transport is reachable only through fixed read methods.
        url = f"{self.API_ROOT}{path}{self._query(options)}"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Multi-Agent-Platform-GitHub-Agent",
        }
        token = self._auth_provider.get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(url, headers=headers, method="GET")
        # This is a bounded retry for idempotent GETs only. Authentication and
        # rate-limit failures are surfaced for the caller to retry later.
        for attempt in range(2):
            try:
                with self._opener(request, timeout=self._timeout_seconds) as response:
                    try:
                        return json.loads(response.read().decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        raise AgentError(
                            code="GITHUB_INVALID_RESPONSE",
                            message="GitHub API returned an invalid JSON response.",
                            category=ErrorCategory.EXTERNAL_SERVICE,
                            retryable=True,
                        ) from None
            except AgentError as error:
                if error.code == "GITHUB_INVALID_RESPONSE" and attempt == 0:
                    self._sleeper(self._retry_delay_seconds)
                    continue
                raise
            except HTTPError as error:
                status = error.code
                response_headers = error.headers or {}
                rate_limited = status == 429 or (
                    status == 403
                    and (
                        response_headers.get("X-RateLimit-Remaining") == "0"
                        or response_headers.get("Retry-After") is not None
                    )
                )
                error.close()
                retryable = rate_limited or status == 408 or status >= 500
                if status in (401, 403) and not rate_limited:
                    category = ErrorCategory.AUTHENTICATION
                    code = "GITHUB_AUTHENTICATION_FAILED"
                else:
                    category = ErrorCategory.EXTERNAL_SERVICE
                    code = "GITHUB_API_ERROR"
                # Honor server-directed backoff by surfacing rate limits rather
                # than retrying them immediately inside this synchronous client.
                if (status == 408 or status >= 500) and attempt == 0:
                    self._sleeper(self._retry_delay_seconds)
                    continue
                raise AgentError(
                    code=code,
                    message=f"GitHub API returned HTTP {status}.",
                    category=category,
                    retryable=retryable,
                    details={"status": status},
                ) from None
            except (URLError, TimeoutError, OSError):
                if attempt == 0:
                    self._sleeper(self._retry_delay_seconds)
                    continue
                raise AgentError(
                    code="GITHUB_CONNECTION_FAILED",
                    message="Could not connect to the GitHub API.",
                    category=ErrorCategory.EXTERNAL_SERVICE,
                    retryable=True,
                ) from None

        raise AgentError(
            code="GITHUB_CONNECTION_FAILED",
            message="Could not connect to the GitHub API.",
            category=ErrorCategory.EXTERNAL_SERVICE,
            retryable=True,
        ) from None

    def get_repository(self, owner: str, repo: str) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}")

    def list_repositories(self, username: str | None = None, organization: str | None = None, **options: Any) -> Any:
        path = f"/users/{self._segment(username)}/repos" if username else (
            f"/orgs/{self._segment(organization)}/repos" if organization else "/user/repos"
        )
        return self._get(path, **options)

    def get_repository_activity(self, owner: str, repo: str, **options: Any) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/events", **options)

    def list_pull_requests(self, owner: str, repo: str, **options: Any) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/pulls", **options)

    def get_pull_request(self, owner: str, repo: str, pull_number: int) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/pulls/{pull_number}")

    def get_pull_request_files(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/pulls/{pull_number}/files", **options)

    def get_pull_request_commits(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/pulls/{pull_number}/commits", **options)

    def get_pull_request_reviews(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/pulls/{pull_number}/reviews", **options)

    def get_pull_request_review_comments(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/pulls/{pull_number}/comments", **options)

    def list_issues(self, owner: str, repo: str, **options: Any) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/issues", **options)

    def get_issue(self, owner: str, repo: str, issue_number: int) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/issues/{issue_number}")

    def get_issue_comments(self, owner: str, repo: str, issue_number: int, **options: Any) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/issues/{issue_number}/comments", **options)

    def list_commits(self, owner: str, repo: str, **options: Any) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/commits", **options)

    def get_commit(self, owner: str, repo: str, commit_sha: str) -> Any:
        return self._get(f"/repos/{self._segment(owner)}/{self._segment(repo)}/commits/{self._segment(commit_sha)}")
