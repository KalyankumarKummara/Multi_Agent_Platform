"""GitHub specialist agent package."""

from .agent import GitHubAgent, GITHUB_AGENT_IDENTITY
from .authentication import EnvironmentTokenProvider, GitHubAuthProvider
from .gateway import GitHubToolGateway
from .github_client import GitHubClient, GitHubRESTClient

__all__ = [
    "GitHubAgent",
    "GITHUB_AGENT_IDENTITY",
    "EnvironmentTokenProvider",
    "GitHubAuthProvider",
    "GitHubToolGateway",
    "GitHubClient",
    "GitHubRESTClient",
]
