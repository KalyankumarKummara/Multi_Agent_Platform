"""Build the in-process runtime for the GitHub specialist application."""

from dataclasses import dataclass

from github_agent import (
    GITHUB_AGENT_IDENTITY,
    GitHubAgent,
    GitHubWebhookHandler,
    ActivityStore,
    InMemoryActivityStore,
)
from github_agent.activity_query import GitHubActivityQuery
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
    activity_store: ActivityStore
    activity_query: GitHubActivityQuery
    agent: GitHubAgent
    webhook_handler: GitHubWebhookHandler


def create_github_runtime(
    activity_store: ActivityStore | None = None,
) -> GitHubRuntime:
    """Build a runtime, defaulting to memory or accepting another store."""
    config = AgentConfig(identity=GITHUB_AGENT_IDENTITY)
    memory = InMemoryStore()
    policy = PolicyEngine()
    audit = AuditService()
    error_handler = ErrorHandler()
    selected_activity_store = (
        activity_store if activity_store is not None else InMemoryActivityStore()
    )
    activity_query = GitHubActivityQuery(selected_activity_store)

    agent = GitHubAgent(
        config=config,
        memory=memory,
        policy=policy,
        audit=audit,
        error_handler=error_handler,
        activity_store=selected_activity_store,
        activity_query=activity_query,
    )
    webhook_handler = GitHubWebhookHandler()

    return GitHubRuntime(
        config=config,
        memory=memory,
        policy=policy,
        audit=audit,
        error_handler=error_handler,
        activity_store=selected_activity_store,
        activity_query=activity_query,
        agent=agent,
        webhook_handler=webhook_handler,
    )
