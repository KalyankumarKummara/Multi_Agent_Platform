"""Production ASGI entry point for the platform application."""

from fastapi import FastAPI

from app.api.webhooks.github import create_github_webhook_router


app = FastAPI(title="Multi-Agent Platform")
app.include_router(create_github_webhook_router())
