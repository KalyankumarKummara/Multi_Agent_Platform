
"""Production ASGI entry point for the platform application."""

from dotenv import load_dotenv

# Load .env before importing modules that use environment variables.
load_dotenv()

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.github.activities import create_github_activity_router
from app.api.github.reads import create_github_read_router
from app.api.webhooks.github import create_github_webhook_router
from app.composition import create_github_runtime


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize the database-backed runtime during application startup."""
    application.state.github_runtime = create_github_runtime()
    yield


app = FastAPI(title="Multi-Agent Platform", lifespan=lifespan)

app.include_router(create_github_webhook_router())
app.include_router(create_github_activity_router())
app.include_router(create_github_read_router())
