"""FastAPI read API for GitHub activity history."""

from fastapi import APIRouter, HTTPException, Query, Request

from app.composition import GitHubRuntime, create_github_runtime
from github_agent.activity import ActivityRecord
from github_agent.activity_query import ActivityQueryError
from infrastructure.database.mssql.activity_store import ActivityPersistenceError


def create_github_activity_router(runtime: GitHubRuntime | None = None) -> APIRouter:
    """Create read-only GitHub activity routes using the shared runtime."""
    router = APIRouter(prefix="/github", tags=["github"])

    @router.get("/activities")
    async def list_github_activities(
        request: Request,
        limit: int = Query(default=20, ge=1, le=100),
        repository: str | None = None,
        event_type: str | None = None,
        status: str | None = None,
        significance: str | None = None,
    ) -> list[ActivityRecord]:
        selected_runtime = runtime or getattr(
            request.app.state,
            "github_runtime",
            None,
        )
        if selected_runtime is None:
            selected_runtime = create_github_runtime()
            request.app.state.github_runtime = selected_runtime
        try:
            return selected_runtime.activity_query.list_recent_activity(
                limit=limit,
                repository=repository,
                event_type=event_type,
                status=status,
                significance=significance,
            )
        except ActivityQueryError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ActivityPersistenceError:
            raise HTTPException(
                status_code=503,
                detail="Persisted GitHub activity storage is unavailable.",
            ) from None

    return router
