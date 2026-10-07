"""Authentication providers for GitHub clients.

Providers are injected, keeping credentials out of tools and the gateway.
"""

from abc import ABC, abstractmethod
import os

from shared.agent_core.errors import AgentError, ErrorCategory


class GitHubAuthProvider(ABC):
    """Supplies an optional GitHub token to an API client."""

    @abstractmethod
    def get_token(self) -> str | None:
        """Return a token, or None for unauthenticated public reads."""


class EnvironmentTokenProvider(GitHubAuthProvider):
    """Reads a token from a named environment variable on demand."""

    def __init__(self, variable_name: str = "GITHUB_TOKEN") -> None:
        if not isinstance(variable_name, str) or not variable_name.strip():
            raise AgentError(
                code="GITHUB_AUTH_CONFIGURATION_ERROR",
                message="GitHub token environment variable name is invalid.",
                category=ErrorCategory.CONFIGURATION,
                retryable=False,
            )
        self.variable_name = variable_name.strip()

    def get_token(self) -> str | None:
        token = os.environ.get(self.variable_name)
        if token is not None and not token.strip():
            raise AgentError(
                code="GITHUB_AUTH_CONFIGURATION_ERROR",
                message="GitHub token environment variable is empty.",
                category=ErrorCategory.CONFIGURATION,
                retryable=False,
                details={"variable": self.variable_name},
            )
        return token
