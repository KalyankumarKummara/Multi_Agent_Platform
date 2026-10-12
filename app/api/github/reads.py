"""Read-only GitHub data routes backed by named agent tools."""

import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from app.composition import GitHubRuntime, create_github_runtime
from shared.agent_core.errors import AgentError, ErrorCategory
from shared.agent_core.security import Principal
from shared.agent_core.tools.execution import ToolExecutionContext
from shared.agent_core.tracing import create_trace_context


_READ_PRINCIPAL = Principal(
    principal_id="github-demo-api",
    principal_type="service",
    permissions=frozenset({"github.read"}),
)


def _runtime(request: Request, injected: GitHubRuntime | None) -> GitHubRuntime:
    selected = injected or getattr(request.app.state, "github_runtime", None)
    if selected is None:
        selected = create_github_runtime()
        request.app.state.github_runtime = selected
    return selected


def _run_read(
    runtime: GitHubRuntime,
    tool_id: str,
    arguments: dict[str, Any],
) -> Any:
    tool = runtime.agent.tool_registry.get(tool_id)
    if tool is None:
        raise HTTPException(status_code=503, detail="GitHub read service is unavailable.")

    context = ToolExecutionContext(
        principal=_READ_PRINCIPAL,
        trace_context=create_trace_context(),
        permissions=["github.read"],
        metadata={"source": "github-demo-api"},
    )
    try:
        return runtime.tool_executor.execute(tool, arguments, context)
    except AgentError as error:
        upstream_status = re.fullmatch(
            r"Tool execution failed: GitHub API returned HTTP (401|403|404)\.",
            error.message,
        )
        if upstream_status:
            status = int(upstream_status.group(1))
            if status in {401, 403}:
                raise HTTPException(
                    status_code=502,
                    detail="GitHub denied this read. Check the configured token's repository access.",
                ) from None
            if status == 404:
                raise HTTPException(
                    status_code=404,
                    detail="The GitHub repository or requested data was not found.",
                ) from None
        if error.category in {ErrorCategory.AUTHORIZATION, ErrorCategory.POLICY}:
            status_code, message = 403, "This GitHub read operation is not permitted."
        elif error.category == ErrorCategory.VALIDATION:
            status_code, message = 400, "GitHub read parameters were rejected."
        elif error.category in {ErrorCategory.EXTERNAL_SERVICE, ErrorCategory.TIMEOUT, ErrorCategory.TOOL}:
            status_code, message = 502, "GitHub could not complete the read request."
        elif error.category == ErrorCategory.APPROVAL:
            status_code, message = 403, "This GitHub read operation requires approval."
        else:
            status_code, message = 503, "GitHub read service is unavailable."
        raise HTTPException(status_code=status_code, detail=message) from None


def _optional(**values: Any) -> dict[str, Any]:
    return {name: value for name, value in values.items() if value is not None}


def create_github_read_router(runtime: GitHubRuntime | None = None) -> APIRouter:
    """Create fixed-operation read routes using the composed agent runtime."""
    router = APIRouter(prefix="/github", tags=["github"])

    @router.get("/repository")
    async def repository(request: Request, owner: str, repo: str) -> Any:
        return _run_read(_runtime(request, runtime), "get_repository", {"owner": owner, "repo": repo})

    @router.get("/repositories")
    async def repositories(
        request: Request,
        username: str | None = None,
        organization: str | None = None,
        per_page: int = Query(default=20, ge=1, le=100),
        page: int = Query(default=1, ge=1),
    ) -> Any:
        if bool(username) == bool(organization):
            raise HTTPException(
                status_code=400,
                detail="Specify exactly one GitHub username or organization.",
            )
        arguments = {"per_page": per_page, "page": page}
        arguments.update(_optional(username=username, organization=organization))
        return _run_read(_runtime(request, runtime), "list_repositories", arguments)

    @router.get("/commits")
    async def commits(
        request: Request,
        owner: str,
        repo: str,
        sha: str | None = None,
        path: str | None = None,
        author: str | None = None,
        since: str | None = None,
        until: str | None = None,
        per_page: int = Query(default=20, ge=1, le=100),
        page: int = Query(default=1, ge=1),
    ) -> Any:
        arguments = {"owner": owner, "repo": repo, "per_page": per_page, "page": page}
        arguments.update(_optional(sha=sha, path=path, author=author, since=since, until=until))
        return _run_read(_runtime(request, runtime), "list_commits", arguments)

    @router.get("/branches")
    async def branches(
        request: Request,
        owner: str,
        repo: str,
        per_page: int = Query(default=20, ge=1, le=100),
        page: int = Query(default=1, ge=1),
    ) -> Any:
        arguments = {"owner": owner, "repo": repo, "per_page": per_page, "page": page}
        return _run_read(_runtime(request, runtime), "list_branches", arguments)

    @router.get("/pull-requests")
    async def pull_requests(
        request: Request,
        owner: str,
        repo: str,
        state: str | None = None,
        base: str | None = None,
        head: str | None = None,
        sort: str | None = None,
        direction: str | None = None,
        per_page: int = Query(default=20, ge=1, le=100),
        page: int = Query(default=1, ge=1),
    ) -> Any:
        arguments = {"owner": owner, "repo": repo, "per_page": per_page, "page": page}
        arguments.update(_optional(state=state, base=base, head=head, sort=sort, direction=direction))
        return _run_read(_runtime(request, runtime), "list_pull_requests", arguments)

    @router.get("/issues")
    async def issues(
        request: Request,
        owner: str,
        repo: str,
        state: str | None = None,
        labels: str | None = None,
        sort: str | None = None,
        direction: str | None = None,
        since: str | None = None,
        per_page: int = Query(default=20, ge=1, le=100),
        page: int = Query(default=1, ge=1),
    ) -> Any:
        arguments = {"owner": owner, "repo": repo, "per_page": per_page, "page": page}
        arguments.update(_optional(state=state, labels=labels, sort=sort, direction=direction, since=since))
        results = _run_read(_runtime(request, runtime), "list_issues", arguments)
        if isinstance(results, list):
            return [item for item in results if not (isinstance(item, dict) and "pull_request" in item)]
        return results

    return router
