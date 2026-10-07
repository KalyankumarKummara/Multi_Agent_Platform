"""GitHub specialist agent and its shared-tool-registry integration."""

from typing import Any

from shared.agent_core.audit import AuditService
from shared.agent_core.base_agent import BaseAgent
from shared.agent_core.capabilities import AgentCapability
from shared.agent_core.config import AgentConfig
from shared.agent_core.context import AgentContext
from shared.agent_core.error_handler import ErrorHandler
from shared.agent_core.identity import AgentIdentity
from shared.agent_core.memory import Memory
from shared.agent_core.policy import PolicyEngine

from .activity import ActivityRecord, ActivityStore, InMemoryActivityStore
from .authentication import EnvironmentTokenProvider, GitHubAuthProvider
from .gateway import GitHubToolGateway
from .github_client import GitHubClient, GitHubRESTClient
from .tools import create_github_tools
from .webhook import NormalizedGitHubEvent

GITHUB_AGENT_IDENTITY = AgentIdentity(
    agent_id="github-agent",
    name="GitHub Agent",
    agent_type="specialist",
    domain="github",
    version="1.0.0",
)


class GitHubAgent(BaseAgent):
    """Background specialist scaffold for GitHub domain activity."""

    def __init__(
        self,
        config: AgentConfig,
        memory: Memory,
        policy: PolicyEngine,
        audit: AuditService,
        error_handler: ErrorHandler,
        *,
        github_gateway: GitHubToolGateway | None = None,
        github_client: GitHubClient | None = None,
        auth_provider: GitHubAuthProvider | None = None,
        activity_store: ActivityStore | None = None,
    ) -> None:
        super().__init__(
            identity=GITHUB_AGENT_IDENTITY,
            config=config,
            memory=memory,
            policy=policy,
            audit=audit,
            error_handler=error_handler,
        )
        self.capabilities.extend(
            [
                AgentCapability(
                    capability_id="github.event_processing",
                    name="GitHub Event Processing",
                    description="Process normalized GitHub events.",
                    input_schema={"event": "object"},
                    output_schema={"status": "string"},
                    required_permissions=[],
                ),
                AgentCapability(
                    capability_id="github.repository_monitoring",
                    name="GitHub Repository Monitoring",
                    description="Monitor activity for configured repositories.",
                    input_schema={"repository": "string"},
                    output_schema={"status": "string"},
                    required_permissions=[],
                ),
                AgentCapability(
                    capability_id="github.activity_tracking",
                    name="GitHub Activity Tracking",
                    description="Track meaningful GitHub activity.",
                    input_schema={"activity": "object"},
                    output_schema={"status": "string"},
                    required_permissions=[],
                ),
            ]
        )
        if github_gateway is not None and github_client is not None:
            raise ValueError("Pass github_gateway or github_client, not both.")
        if github_gateway is None:
            client = github_client or GitHubRESTClient(
                auth_provider or EnvironmentTokenProvider()
            )
            github_gateway = GitHubToolGateway(client)
        # Tools live in BaseAgent's registry and are always invoked through
        # the shared ToolExecutor supplied by the application runtime.
        for tool in create_github_tools(github_gateway):
            self.tool_registry.register(tool)
        self.activity_store = (
            activity_store if activity_store is not None else InMemoryActivityStore()
        )

    def handle_event(self, event: Any, context: AgentContext) -> dict[str, Any]:
        """Record a normalized GitHub event as successful domain activity."""
        if not isinstance(event, NormalizedGitHubEvent):
            raise ValueError("GitHubAgent requires a normalized GitHub event.")

        activity = ActivityRecord.from_event(
            event,
            agent_id=self.identity.agent_id,
        )
        self.activity_store.save(activity)
        context.event = event
        return {"status": "event_processed", "activity_id": activity.activity_id}

    def execute(self, request: Any, context: AgentContext) -> dict[str, Any]:
        """Accept an agent request without performing GitHub operations."""
        context.request = request
        return {"status": "request_received", "request": request}
