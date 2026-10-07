"""FastAPI boundary for GitHub webhook requests."""

import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.composition import GitHubRuntime, create_github_runtime
from shared.agent_core.errors import AgentError, ErrorCategory


def _error_response(error: AgentError) -> JSONResponse:
    """Translate domain errors without returning internal details."""
    if error.category == ErrorCategory.AUTHENTICATION:
        status_code, message = 401, "Webhook signature was rejected."
    elif error.category == ErrorCategory.VALIDATION:
        status_code, message = 400, "Webhook request is invalid."
    elif error.category == ErrorCategory.CONFIGURATION:
        status_code, message = 503, "Webhook processing is not configured."
    elif error.category in (ErrorCategory.AUTHORIZATION, ErrorCategory.POLICY):
        status_code, message = 403, "Webhook request is not permitted."
    elif error.retryable:
        status_code, message = 503, "Webhook processing failed; retry later."
    else:
        status_code, message = 500, "Webhook processing failed."

    return JSONResponse(status_code=status_code, content={"detail": message})


def create_github_webhook_router(runtime: GitHubRuntime | None = None) -> APIRouter:
    """Create the route with a composed runtime, injectable for local tests."""
    selected_runtime = runtime or create_github_runtime()
    router = APIRouter()

    @router.post("/webhooks/github")
    async def receive_github_webhook(request: Request) -> JSONResponse:
        # Preserve the exact bytes: GitHubWebhookHandler verifies this body
        # before decoding or parsing it.
        raw_body = await request.body()
        secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
        try:
            result = selected_runtime.webhook_handler.handle(
                raw_body,
                request.headers,
                secret,
                selected_runtime.agent,
            )
        except AgentError as error:
            return _error_response(error)
        except Exception:
            # Keep unexpected internals and payload contents out of responses.
            return JSONResponse(
                status_code=500,
                content={"detail": "Webhook processing failed."},
            )

        return JSONResponse(
            status_code=200,
            content={"status": result.status, "delivery_id": result.delivery_id},
        )

    return router
