"""Purpose-specific read-only GitHub tools."""

from typing import Any

from shared.agent_core.tools.tool import Tool

from ..gateway import GitHubToolGateway


_REPO = {
    "owner": {"type": "string", "required": True},
    "repo": {"type": "string", "required": True},
}
_PAGINATION = {
    "state": {"type": "string"},
    "per_page": {"type": "integer"},
    "page": {"type": "integer"},
}
_OPERATIONS: list[tuple[str, str, str, dict[str, dict[str, Any]], str, dict[str, dict[str, Any]]]] = [
    ("get_repository", "Get repository", "Read a repository overview including owner, description, visibility, default branch, dates, stars, forks, GitHub open_issues_count, license, archived status, and disabled status. Use for general repository metadata questions; GitHub's open_issues_count can include pull requests. Use get_issue_statistics for issue-only counts, status breakdowns, and percentages.", _REPO, "object", {}),
    ("list_repositories", "List repositories", "List repositories for a user, organization, or authenticated user.", {
        "username": {"type": "string"}, "organization": {"type": "string"},
        "type": {"type": "string"}, "sort": {"type": "string"},
        "per_page": {"type": "integer"}, "page": {"type": "integer"},
    }, "array", {}),
    ("get_repository_activity", "Get repository activity", "List recent repository events.", {**_REPO, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("list_branches", "List branches", "List repository branches.", {**_REPO, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("list_contributors", "List contributors", "List repository contributors with GitHub login, contribution count, profile URL, and avatar URL when available. Returns up to 10 by default and at most 100 per page. Use the requested limit for top N contributor questions; counts and top contributors refer only to the returned page, not necessarily the whole repository.", {**_REPO, "limit": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("list_pull_requests", "List pull requests", "List repository pull requests, optionally filtering to merged or closed unmerged pull requests.", {**_REPO, **_PAGINATION, "merged_only": {"type": "boolean"}, "unmerged_only": {"type": "boolean"}, "base": {"type": "string"}, "head": {"type": "string"}, "sort": {"type": "string"}, "direction": {"type": "string"}}, "array", {}),
    ("get_pull_request", "Get pull request", "Read pull request details including requested individual reviewers and requested review teams, which are pending requests and distinct from submitted reviews.", {**_REPO, "pull_number": {"type": "integer", "required": True}}, "object", {}),
    ("get_pull_request_files", "Get pull request files", "List changed files in a pull request with status, line counts, and available patches.", {**_REPO, "pull_number": {"type": "integer", "required": True}, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("get_pull_request_commits", "Get pull request commits", "List commits in a pull request with SHA, message, author association, date, and URL when available.", {**_REPO, "pull_number": {"type": "integer", "required": True}, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("get_pull_request_reviews", "Get pull request reviews", "List submitted reviews, reviewer logins, decisions, submission dates, and available review bodies for a pull request. Requested reviewers are returned by pull request details and are distinct from submitted reviews.", {**_REPO, "pull_number": {"type": "integer", "required": True}, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("get_pull_request_review_comments", "Get pull request review comments", "List review comments with author logins, file paths, creation dates, and comment bodies for a pull request.", {**_REPO, "pull_number": {"type": "integer", "required": True}, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("list_issues", "List repository issues", "List issues with state open, closed, or all; when omitted, GitHub defaults to open. For all-issue requests, explicitly pass state=all. Supports labels, assignee, creator, updated-since timestamp, and created_after (inclusive YYYY-MM-DD date). Sort by created, updated, or comments; direction asc/desc; per_page 1-100 and page pagination. Pull requests are excluded. For stale issue requests, set stale_days to the inactivity threshold (defaults to 7 days when the request says stale without a number); this scans up to 10 pages and reports if results may be incomplete. Stale classifications are Stale (7+ days), Very stale (14+ days), and Critical stale (30+ days). Use assignee @me for issues assigned to the authenticated account. Use get_issue_statistics for counts and percentages, and search_issues for keyword searches.", {**_REPO, "state": {"type": "string", "enum": ["open", "closed", "all"]}, "per_page": {"type": "integer"}, "page": {"type": "integer"}, "labels": {"type": "string"}, "assignee": {"type": "string"}, "creator": {"type": "string"}, "created_after": {"type": "string"}, "sort": {"type": "string", "enum": ["created", "updated", "comments"]}, "direction": {"type": "string", "enum": ["asc", "desc"]}, "since": {"type": "string"}, "stale_days": {"type": "integer"}}, "array", {}),
    ("get_issue", "Get issue", "Read one issue.", {**_REPO, "issue_number": {"type": "integer", "required": True}}, "object", {}),
    ("get_issue_statistics", "Get issue statistics", "Calculate repository-wide open, closed, and total issue counts and percentages. Excludes pull requests and paginates the Issues API. Use for issue statistics questions instead of repository overview metadata or keyword search. Results explicitly report when pagination is incomplete.", {**_REPO, "max_pages": {"type": "integer"}}, "object", {}),
    ("search_issues", "Search issues", "Search words or phrases in issue titles and bodies in the selected repository. Supports open, closed, or all states and pagination. Always searches issues only and excludes pull requests. Use get_issue when the user names a specific issue number.", {**_REPO, "keyword": {"type": "string", "required": True}, "state": {"type": "string"}, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "object", {}),
    ("get_issue_comments", "Get issue comments", "List comments on an issue.", {**_REPO, "issue_number": {"type": "integer", "required": True}, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("list_commits", "List commits", "List repository commits.", {**_REPO, "sha": {"type": "string"}, "path": {"type": "string"}, "author": {"type": "string"}, "since": {"type": "string"}, "until": {"type": "string"}, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("get_commit", "Get commit", "Read one commit by SHA.", {**_REPO, "commit_sha": {"type": "string", "required": True}}, "object", {}),
]


def create_github_tools(gateway: GitHubToolGateway) -> list[Tool]:
    """Bind explicit tool definitions to gateway methods.

    Separate tool IDs let the common executor validate, authorize, apply
    policy, audit, and execute each read operation through its normal path.
    """
    tools = []
    for tool_id, name, description, input_schema, output_type, _ in _OPERATIONS:
        operation = getattr(gateway, tool_id)
        tools.append(
            Tool(
                tool_id=tool_id,
                name=name,
                description=description,
                input_schema=input_schema,
                output_schema=(
                    {"anyOf": [{"type": "array"}, {"type": "object"}]}
                    if tool_id == "list_issues"
                    else {"type": output_type}
                ),
                permissions=["github.read"],
                risk_level="low",
                execute=operation,
            )
        )
    return tools


__all__ = ["create_github_tools"]
