"""GitHub specialist agent package."""

from .agent import GitHubAgent, GITHUB_AGENT_IDENTITY
from .activity import ActivityRecord, ActivityStore, InMemoryActivityStore
from .authentication import EnvironmentTokenProvider, GitHubAuthProvider
from .gateway import GitHubToolGateway
from .github_client import GitHubClient, GitHubRESTClient
from .webhook import (
    GitHubEventNormalizer,
    GitHubWebhookHandler,
    GitHubWebhookSignatureVerifier,
    InMemoryDeliveryStore,
    NormalizedGitHubEvent,
    WebhookResult,
)

__all__ = [
    "GitHubAgent",
    "ActivityRecord",
    "ActivityStore",
    "InMemoryActivityStore",
    "GITHUB_AGENT_IDENTITY",
    "EnvironmentTokenProvider",
    "GitHubAuthProvider",
    "GitHubToolGateway",
    "GitHubClient",
    "GitHubRESTClient",
    "GitHubEventNormalizer",
    "GitHubWebhookHandler",
    "GitHubWebhookSignatureVerifier",
    "InMemoryDeliveryStore",
    "NormalizedGitHubEvent",
    "WebhookResult",
]
