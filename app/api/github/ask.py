"""Natural-language endpoint for supported read-only GitHub requests."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.github_request_service import GitHubRequestServiceError


class GitHubAskRequest(BaseModel):
    """Request body for the GitHub-only natural-language endpoint."""

    message: str = Field(min_length=1, max_length=4000)


def create_github_ask_router() -> APIRouter:
    """Create the route that uses the runtime initialized by app lifespan."""
    router = APIRouter(prefix="/github", tags=["github"])

    @router.post("/ask")
    async def ask_github(request: Request, body: GitHubAskRequest) -> dict[str, Any]:
        if not body.message.strip():
            raise HTTPException(status_code=422, detail="The message must not be empty.")

        runtime = getattr(request.app.state, "github_runtime", None)
        if runtime is None:
            raise HTTPException(status_code=503, detail="The GitHub request service is unavailable.")

        service = getattr(runtime, "github_request_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="The GitHub request service is unavailable.")

        try:
            return service.process(body.message)
        except GitHubRequestServiceError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=error.message,
            ) from None
        except Exception:
            raise HTTPException(
                status_code=502,
                detail="The GitHub request could not be completed.",
            ) from None

    return router
