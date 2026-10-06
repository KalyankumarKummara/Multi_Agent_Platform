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
    ("get_repository", "Get repository", "Read repository details.", _REPO, "object", {}),
    ("list_repositories", "List repositories", "List repositories for a user, organization, or authenticated user.", {
        "username": {"type": "string"}, "organization": {"type": "string"},
        "type": {"type": "string"}, "sort": {"type": "string"}, **_PAGINATION,
    }, "array", {}),
    ("get_repository_activity", "Get repository activity", "List recent repository events.", {**_REPO, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("list_pull_requests", "List pull requests", "List repository pull requests.", {**_REPO, **_PAGINATION, "base": {"type": "string"}, "head": {"type": "string"}, "sort": {"type": "string"}, "direction": {"type": "string"}}, "array", {}),
    ("get_pull_request", "Get pull request", "Read one pull request.", {**_REPO, "pull_number": {"type": "integer", "required": True}}, "object", {}),
    ("get_pull_request_files", "Get pull request files", "List files changed by a pull request.", {**_REPO, "pull_number": {"type": "integer", "required": True}, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("get_pull_request_commits", "Get pull request commits", "List commits in a pull request.", {**_REPO, "pull_number": {"type": "integer", "required": True}, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("get_pull_request_reviews", "Get pull request reviews", "List reviews for a pull request.", {**_REPO, "pull_number": {"type": "integer", "required": True}, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("get_pull_request_review_comments", "Get pull request review comments", "List review comments for a pull request.", {**_REPO, "pull_number": {"type": "integer", "required": True}, "per_page": {"type": "integer"}, "page": {"type": "integer"}}, "array", {}),
    ("list_issues", "List issues", "List repository issues.", {**_REPO, **_PAGINATION, "labels": {"type": "string"}, "sort": {"type": "string"}, "direction": {"type": "string"}, "since": {"type": "string"}}, "array", {}),
    ("get_issue", "Get issue", "Read one issue.", {**_REPO, "issue_number": {"type": "integer", "required": True}}, "object", {}),
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
                output_schema={"type": output_type},
                permissions=["github.read"],
                risk_level="low",
                execute=operation,
            )
        )
    return tools


__all__ = ["create_github_tools"]
