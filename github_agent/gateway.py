"""Narrow domain gateway between GitHub tools and the injected API client."""

import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from shared.agent_core.errors import AgentError, ErrorCategory

from .github_client import GitHubClient


logger = logging.getLogger(__name__)


class GitHubToolGateway:
    """Expose only the approved, read-only GitHub operations.

    Purpose-specific methods keep tools from constructing arbitrary URLs and
    provide one place to normalize inputs and remove sensitive response keys.
    """

    _SENSITIVE_KEY_PARTS = ("authorization", "token", "password", "secret")
    DEFAULT_ISSUE_STATISTICS_MAX_PAGES = 10
    MAX_ISSUE_STATISTICS_MAX_PAGES = 100
    ISSUE_STATISTICS_PER_PAGE = 100
    DEFAULT_STALE_ISSUE_DAYS = 7

    def __init__(self, client: GitHubClient) -> None:
        self._client = client

    @classmethod
    def _sanitize(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: cls._sanitize(item)
                for key, item in value.items()
                if not any(part in str(key).casefold() for part in cls._SENSITIVE_KEY_PARTS)
            }
        if isinstance(value, list):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, str):
            value = re.sub(
                r"(?i)\b(?:authorization|proxy-authorization|x-api-key|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|password|secret)\b\s*[:=]\s*(?:bearer\s+)?[^\s,;]+",
                "[redacted credential]",
                value,
            )
            value = re.sub(
                r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*",
                "Bearer [redacted]",
                value,
            )
            return re.sub(
                r"\b(?:gsk_[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,})\b",
                "[redacted token]",
                value,
                flags=re.IGNORECASE,
            )
        return value

    @staticmethod
    def _name(value: str, field: str) -> str:
        if not isinstance(value, str):
            raise AgentError(
                code="INVALID_FIELD_TYPE",
                message=f"{field} must be a string.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": field, "expected_type": "string"},
            )
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
        except AgentError as error:
            details = error.details if isinstance(error.details, dict) else {}
            status = details.get("status")
            if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status <= 599:
                status = None
            code = error.code
            if not isinstance(code, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code) is None:
                code = "unknown"
            category = (
                error.category.value
                if isinstance(error.category, ErrorCategory)
                else "unknown"
            )
            retryable = error.retryable if isinstance(error.retryable, bool) else "unknown"
            operation = method if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", method) else "unknown"
            logger.error(
                "GitHub client operation failed operation=%s error_code=%s category=%s retryable=%s http_status=%s",
                operation,
                code,
                category,
                retryable,
                status if status is not None else "unknown",
            )
            raise
        except Exception as error:
            # Do not echo exception text: HTTP libraries may include request data.
            operation = method if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", method) else "unknown"
            exception_type = type(error).__name__
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", exception_type) is None:
                exception_type = "unknown"
            logger.error(
                "GitHub client operation failed operation=%s exception_type=%s",
                operation,
                exception_type,
            )
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

    def list_branches(self, owner: str, repo: str, **options: Any) -> Any:
        options = self._options(options, "per_page", "page")
        return self._call("list_branches", *self._repo_args(owner, repo), **options)

    def list_contributors(
        self,
        owner: str,
        repo: str,
        limit: int = 10,
        page: int = 1,
    ) -> Any:
        """List one bounded contributor page sorted by contribution count."""
        limit = self._number(limit, "limit")
        if limit > 100:
            raise AgentError(
                code="INVALID_GITHUB_INPUT",
                message="limit must not exceed 100.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": "limit"},
            )
        options = self._options({"page": page}, "page")
        result = self._call(
            "list_contributors",
            *self._repo_args(owner, repo),
            per_page=limit,
            **options,
        )
        if isinstance(result, list):
            def contribution_count(item: Any) -> tuple[int, int]:
                count = item.get("contributions") if isinstance(item, dict) else None
                if type(count) is int and count >= 0:
                    return 0, -count
                return 1, 0

            return sorted(result, key=contribution_count)[:limit]
        return result

    def list_pull_requests(
        self,
        owner: str,
        repo: str,
        merged_only: bool = False,
        unmerged_only: bool = False,
        **options: Any,
    ) -> Any:
        if not isinstance(merged_only, bool) or not isinstance(unmerged_only, bool):
            raise AgentError(
                code="INVALID_FIELD_TYPE",
                message="merged_only and unmerged_only must be booleans.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": "merged_only or unmerged_only", "expected_type": "boolean"},
            )
        if merged_only and unmerged_only:
            raise AgentError(
                code="CONFLICTING_PULL_REQUEST_FILTERS",
                message="Choose either merged or closed unmerged pull requests.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
            )
        options = self._options(options, "state", "per_page", "page", "base", "head", "sort", "direction")
        if merged_only or unmerged_only:
            # GitHub exposes both classes through the closed state; merged_at
            # distinguishes merged from closed, unmerged pull requests.
            options["state"] = "closed"
        elif "state" not in options:
            # The GitHub API otherwise defaults this endpoint to open pull requests.
            options["state"] = "all"
        results = self._call("list_pull_requests", *self._repo_args(owner, repo), **options)
        if merged_only and isinstance(results, list):
            return [
                pull_request
                for pull_request in results
                if isinstance(pull_request, dict) and pull_request.get("merged_at") is not None
            ]
        if unmerged_only and isinstance(results, list):
            return [
                pull_request
                for pull_request in results
                if isinstance(pull_request, dict)
                and pull_request.get("state") == "closed"
                and pull_request.get("merged_at") is None
            ]
        return results

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
        results = self._call("get_pull_request_commits", owner, repo, self._number(pull_number, "pull_number"), **options)
        if isinstance(results, list):
            for item in results:
                commit = item.get("commit") if isinstance(item, dict) else None
                if isinstance(commit, dict):
                    for identity in ("author", "committer"):
                        metadata = commit.get(identity)
                        if isinstance(metadata, dict):
                            metadata.pop("email", None)
        return results

    def get_pull_request_reviews(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any:
        owner, repo = self._repo_args(owner, repo)
        options = self._options(options, "per_page", "page")
        return self._call("get_pull_request_reviews", owner, repo, self._number(pull_number, "pull_number"), **options)

    def get_pull_request_review_comments(self, owner: str, repo: str, pull_number: int, **options: Any) -> Any:
        owner, repo = self._repo_args(owner, repo)
        options = self._options(options, "per_page", "page")
        return self._call("get_pull_request_review_comments", owner, repo, self._number(pull_number, "pull_number"), **options)

    def list_issues(self, owner: str, repo: str, **options: Any) -> Any:
        owner, repo = self._repo_args(owner, repo)
        stale_days = options.pop("stale_days", None)
        created_after = options.pop("created_after", None)
        if created_after is not None:
            if not isinstance(created_after, str):
                raise AgentError(
                    code="INVALID_FIELD_TYPE",
                    message="created_after must be a YYYY-MM-DD date.",
                    category=ErrorCategory.VALIDATION,
                    retryable=False,
                    details={"field": "created_after"},
                )
            try:
                parsed_created_after = date.fromisoformat(created_after)
            except ValueError:
                raise AgentError(
                    code="INVALID_GITHUB_INPUT",
                    message="created_after must be a YYYY-MM-DD date.",
                    category=ErrorCategory.VALIDATION,
                    retryable=False,
                    details={"field": "created_after"},
                ) from None
            if parsed_created_after.isoformat() != created_after:
                raise AgentError(
                    code="INVALID_GITHUB_INPUT",
                    message="created_after must be a YYYY-MM-DD date.",
                    category=ErrorCategory.VALIDATION,
                    retryable=False,
                    details={"field": "created_after"},
                )

        options = self._options(
            options, "state", "per_page", "page", "labels", "sort", "direction",
            "since", "assignee", "creator",
        )
        state = options.get("state")
        if state is not None and state.casefold() not in {"open", "closed", "all"}:
            raise AgentError(
                code="INVALID_GITHUB_INPUT",
                message="state must be open, closed, or all.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": "state"},
            )
        if state is not None:
            options["state"] = state.casefold()
        sort = options.get("sort")
        if sort is not None and sort.casefold() not in {"created", "updated", "comments"}:
            raise AgentError(
                code="INVALID_GITHUB_INPUT",
                message="sort must be created, updated, or comments.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": "sort"},
            )
        if sort is not None:
            options["sort"] = sort.casefold()
        direction = options.get("direction")
        if direction is not None and direction.casefold() not in {"asc", "desc"}:
            raise AgentError(
                code="INVALID_GITHUB_INPUT",
                message="direction must be asc or desc.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": "direction"},
            )
        if direction is not None:
            options["direction"] = direction.casefold()

        if stale_days is not None:
            stale_days = self._number(stale_days, "stale_days")
            if stale_days > 36500:
                raise AgentError(
                    code="INVALID_GITHUB_INPUT",
                    message="stale_days must not exceed 36500.",
                    category=ErrorCategory.VALIDATION,
                    retryable=False,
                    details={"field": "stale_days"},
                )
            return self._stale_issue_report(owner, repo, options, stale_days, created_after)

        search_for_current_assignee = options.get("assignee", "").casefold() == "@me"
        if created_after is not None or search_for_current_assignee:
            query_parts = [f"repo:{owner}/{repo}", "is:issue"]
            if created_after is not None:
                query_parts.append(f"created:>={created_after}")
            if state in {"open", "closed"}:
                query_parts.append(f"is:{state}")
            if options.get("assignee"):
                query_parts.append(f"assignee:{self._search_qualifier(options['assignee'])}")
            if options.get("creator"):
                query_parts.append(f"author:{self._search_qualifier(options['creator'])}")
            if options.get("labels"):
                for label in options["labels"].split(","):
                    query_parts.append(f'label:"{self._search_qualifier(label.strip())}"')
            if options.get("since"):
                updated_since = options["since"][:10]
                query_parts.append(f"updated:>={updated_since}")
            search_options = {
                "per_page": options.get("per_page", 30),
                "page": options.get("page", 1),
            }
            if sort:
                search_options["sort"] = sort.casefold()
            if direction:
                search_options["order"] = direction.casefold()
            response = self._call("search_issues", " ".join(query_parts), **search_options)
            records = response.get("items") if isinstance(response, dict) else None
        else:
            records = self._call("list_issues", owner, repo, **options)

        if not isinstance(records, list):
            return records
        return [
            issue for issue in records
            if isinstance(issue, dict) and "pull_request" not in issue
        ]

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    def _stale_issue_report(
        self, owner: str, repo: str, options: dict[str, Any], stale_days: int,
        created_after: str | None,
    ) -> dict[str, Any]:
        """Collect a bounded issue report and classify inactivity from updated_at."""
        now = self._now_utc()
        cutoff = now - timedelta(days=stale_days)
        max_pages = self.DEFAULT_ISSUE_STATISTICS_MAX_PAGES
        page_size = self.ISSUE_STATISTICS_PER_PAGE
        search_mode = created_after is not None or options.get("assignee", "").casefold() == "@me"
        items: list[dict[str, Any]] = []
        pages_retrieved = 0
        unclassifiable_count = 0
        total_count = None
        incomplete_reason = None

        for page in range(1, max_pages + 1):
            try:
                if search_mode:
                    query_parts = [f"repo:{owner}/{repo}", "is:issue"]
                    if created_after is not None:
                        query_parts.append(f"created:>={created_after}")
                    state = options.get("state", "open")
                    if state in {"open", "closed"}:
                        query_parts.append(f"is:{state}")
                    if options.get("assignee"):
                        query_parts.append(f"assignee:{self._search_qualifier(options['assignee'])}")
                    if options.get("creator"):
                        query_parts.append(f"author:{self._search_qualifier(options['creator'])}")
                    if options.get("labels"):
                        for label in options["labels"].split(","):
                            query_parts.append(f'label:"{self._search_qualifier(label.strip())}"')
                    if options.get("since"):
                        query_parts.append(f"updated:>={options['since'][:10]}")
                    response = self._call(
                        "search_issues", " ".join(query_parts),
                        per_page=page_size, page=page, sort="updated", order="asc",
                    )
                    raw_records = response.get("items") if isinstance(response, dict) else None
                    if isinstance(response, dict):
                        count = response.get("total_count")
                        if type(count) is int and count >= 0:
                            total_count = count
                        if response.get("incomplete_results") is True:
                            incomplete_reason = "GitHub Search returned incomplete results."
                        elif response.get("incomplete_results") is not False:
                            incomplete_reason = "GitHub Search did not report whether the results are complete."
                else:
                    request_options = {
                        key: value for key, value in options.items()
                        if key not in {"per_page", "page", "sort", "direction"}
                    }
                    request_options.update({
                        "per_page": page_size, "page": page,
                        "sort": "updated", "direction": "asc",
                    })
                    raw_records = self._call("list_issues", owner, repo, **request_options)
            except AgentError:
                incomplete_reason = f"GitHub could not retrieve page {page}; the stale report is incomplete."
                break

            if not isinstance(raw_records, list):
                incomplete_reason = f"Page {page} had an invalid response; the stale report is incomplete."
                break
            pages_retrieved += 1
            for issue in raw_records:
                if not isinstance(issue, dict) or "pull_request" in issue:
                    continue
                raw_updated = issue.get("updated_at")
                if not isinstance(raw_updated, str):
                    unclassifiable_count += 1
                    continue
                try:
                    updated = datetime.fromisoformat(raw_updated.strip().replace("Z", "+00:00"))
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                    updated = updated.astimezone(timezone.utc)
                except (ValueError, OverflowError):
                    unclassifiable_count += 1
                    continue
                inactive_days = max(0, (now - updated).days)
                if updated > cutoff:
                    continue
                if inactive_days >= 30:
                    classification = "Critical stale"
                elif inactive_days >= 14:
                    classification = "Very stale"
                elif inactive_days >= 7:
                    classification = "Stale"
                else:
                    classification = "Below standard stale threshold"
                items.append({
                    **issue,
                    "inactive_days": inactive_days,
                    "stale_classification": classification,
                })

            if search_mode and total_count is not None and total_count > 1000:
                incomplete_reason = "GitHub Search limits accessible results to 1,000 for this query."
                break
            if incomplete_reason:
                break
            if len(raw_records) < page_size:
                break
            if page == max_pages:
                incomplete_reason = f"The {max_pages}-page stale-report safety limit was reached."
        else:
            incomplete_reason = f"The {max_pages}-page stale-report safety limit was reached."
        return {
            "stale_days": stale_days,
            "items": items,
            "pages_retrieved": pages_retrieved,
            "max_pages": max_pages,
            "complete": incomplete_reason is None,
            "incomplete_reason": incomplete_reason,
            "unclassifiable_count": unclassifiable_count,
            "search_total_count": total_count,
        }

    @staticmethod
    def _search_qualifier(value: str) -> str:
        """Escape a value embedded in a quoted GitHub search qualifier."""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def get_issue_statistics(
        self,
        owner: str,
        repo: str,
        max_pages: int = DEFAULT_ISSUE_STATISTICS_MAX_PAGES,
    ) -> dict[str, Any]:
        """Count repository issues across bounded pages, excluding pull requests."""
        owner, repo = self._repo_args(owner, repo)
        max_pages = self._number(max_pages, "max_pages")
        if max_pages > self.MAX_ISSUE_STATISTICS_MAX_PAGES:
            raise AgentError(
                code="INVALID_GITHUB_INPUT",
                message=f"max_pages must not exceed {self.MAX_ISSUE_STATISTICS_MAX_PAGES}.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": "max_pages"},
            )

        open_count = 0
        closed_count = 0
        pages_retrieved = 0
        incomplete_reason = None
        for page in range(1, max_pages + 1):
            try:
                # Count raw endpoint records for pagination before list_issues
                # filters PRs; otherwise a page with PRs could look short.
                records = self._call(
                    "list_issues",
                    owner,
                    repo,
                    state="all",
                    per_page=self.ISSUE_STATISTICS_PER_PAGE,
                    page=page,
                )
            except AgentError as error:
                status = error.details.get("status") if isinstance(error.details, dict) else None
                if status == 401:
                    why = "GitHub authentication prevented retrieving a page"
                elif status == 403 or status == 429:
                    why = "GitHub permissions or rate limits prevented retrieving a page"
                elif type(status) is int and status >= 500:
                    why = "GitHub service failure prevented retrieving a page"
                elif error.code == "GITHUB_CONNECTION_FAILED":
                    why = "a temporary connection problem prevented retrieving a page"
                else:
                    why = "GitHub could not retrieve a page"
                incomplete_reason = f"Statistics are incomplete because {why} (page {page})."
                break

            if not isinstance(records, list):
                incomplete_reason = f"Statistics are incomplete because page {page} had an invalid response."
                break

            pages_retrieved += 1
            malformed_record = False
            for record in records:
                if not isinstance(record, dict):
                    malformed_record = True
                    break
                if "pull_request" in record:
                    continue
                state = record.get("state")
                if state == "open":
                    open_count += 1
                elif state == "closed":
                    closed_count += 1
                else:
                    malformed_record = True
                    break
            if malformed_record:
                incomplete_reason = f"Statistics are incomplete because page {page} contained an issue with missing or invalid state data."
                break
            if len(records) < self.ISSUE_STATISTICS_PER_PAGE:
                break
        else:
            incomplete_reason = (
                f"Statistics are incomplete because the {max_pages}-page safety limit was reached."
            )

        complete = incomplete_reason is None
        total_count = open_count + closed_count
        return {
            "repository": f"{owner}/{repo}",
            "open_count": open_count,
            "closed_count": closed_count,
            "total_count": total_count,
            "open_percentage": round(open_count * 100 / total_count, 2) if complete and total_count else (0.0 if complete else None),
            "closed_percentage": round(closed_count * 100 / total_count, 2) if complete and total_count else (0.0 if complete else None),
            "complete": complete,
            "pages_retrieved": pages_retrieved,
            "max_pages": max_pages,
            "incomplete_reason": incomplete_reason,
        }

    def search_issues(
        self,
        owner: str,
        repo: str,
        keyword: str,
        state: str = "all",
        per_page: int = 30,
        page: int = 1,
    ) -> Any:
        """Search one repository's issue titles and bodies on a bounded page."""
        owner, repo = self._repo_args(owner, repo)
        keyword = self._name(keyword, "keyword")
        if len(keyword) > 256:
            raise AgentError(
                code="INVALID_GITHUB_INPUT",
                message="keyword must not exceed 256 characters.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": "keyword"},
            )
        if not isinstance(state, str) or state.strip().casefold() not in {"open", "closed", "all"}:
            raise AgentError(
                code="INVALID_GITHUB_INPUT",
                message="state must be open, closed, or all.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": "state"},
            )
        state = state.strip().casefold()
        per_page = self._number(per_page, "per_page")
        if per_page > 100:
            raise AgentError(
                code="INVALID_GITHUB_INPUT",
                message="per_page must not exceed 100.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
                details={"field": "per_page"},
            )
        options = self._options({"page": page}, "page")
        escaped_keyword = keyword.replace("\\", "\\\\").replace('"', '\\"')
        query_parts = [f"repo:{owner}/{repo}", "is:issue", "in:title,body"]
        if state in {"open", "closed"}:
            query_parts.append(f"is:{state}")
        query_parts.append(f'"{escaped_keyword}"')
        response = self._call(
            "search_issues",
            " ".join(query_parts),
            per_page=per_page,
            **options,
        )
        if not isinstance(response, dict):
            return {"total_count": None, "incomplete_results": None, "items": [], "page": page, "per_page": per_page}

        raw_items = response.get("items")
        items = []
        if isinstance(raw_items, list):
            for issue in raw_items:
                if not isinstance(issue, dict) or "pull_request" in issue:
                    continue
                item: dict[str, Any] = {}
                for field in ("number", "title", "state", "created_at", "updated_at", "html_url"):
                    value = issue.get(field)
                    if isinstance(value, (str, int)) and not isinstance(value, bool):
                        item[field] = value
                user = issue.get("user")
                if isinstance(user, dict) and isinstance(user.get("login"), str):
                    item["author"] = {"login": user["login"]}
                labels = issue.get("labels")
                if isinstance(labels, list):
                    item["labels"] = [
                        {"name": label["name"]}
                        for label in labels
                        if isinstance(label, dict) and isinstance(label.get("name"), str)
                    ]
                items.append(self._sanitize(item))
        total_count = response.get("total_count")
        if type(total_count) is not int or total_count < 0:
            total_count = None
        incomplete = response.get("incomplete_results")
        if type(incomplete) is not bool:
            incomplete = None
        return {
            "total_count": total_count,
            "incomplete_results": incomplete,
            "items": items,
            "page": page,
            "per_page": per_page,
        }

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
