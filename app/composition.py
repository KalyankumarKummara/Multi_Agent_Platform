"""Build the in-process runtime for the GitHub specialist application."""

from dataclasses import dataclass

from github_agent import GITHUB_AGENT_IDENTITY, GitHubAgent, GitHubWebhookHandler
from shared.agent_core.audit import AuditService
from shared.agent_core.config import AgentConfig
from shared.agent_core.error_handler import ErrorHandler
from shared.agent_core.memory import InMemoryStore
from shared.agent_core.policy import PolicyEngine


@dataclass
class GitHubRuntime:
    """Common services and GitHub components owned by this application."""

    config: AgentConfig
    memory: InMemoryStore
    policy: PolicyEngine
    audit: AuditService
    error_handler: ErrorHandler
    agent: GitHubAgent
    webhook_handler: GitHubWebhookHandler


def create_github_runtime() -> GitHubRuntime:
    """Construct the GitHub agent and its webhook ingestion service."""
    config = AgentConfig(identity=GITHUB_AGENT_IDENTITY)
    memory = InMemoryStore()
    policy = PolicyEngine()
    audit = AuditService()
    error_handler = ErrorHandler()

    agent = GitHubAgent(
        config=config,
        memory=memory,
        policy=policy,
        audit=audit,
        error_handler=error_handler,
    )
    webhook_handler = GitHubWebhookHandler()

    return GitHubRuntime(
        config=config,
        memory=memory,
        policy=policy,
        audit=audit,
        error_handler=error_handler,
        agent=agent,
        webhook_handler=webhook_handler,
    )
