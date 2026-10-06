from abc import ABC, abstractmethod
from typing import Any

from .audit import AuditService
from .config import AgentConfig
from .context import AgentContext
from .error_handler import ErrorHandler
from .identity import AgentIdentity
from .lifecycle import AgentLifecycle
from .memory import Memory
from .policy import PolicyEngine
from .tools.registry import ToolRegistry


class BaseAgent(ABC):

    def __init__(
        self,
        identity: AgentIdentity,
        config: AgentConfig,
        memory: Memory,
        policy: PolicyEngine,
        audit: AuditService,
        error_handler: ErrorHandler,
    ):
        self.identity = identity
        self.config = config

        self.lifecycle = AgentLifecycle()

        self.capabilities: list[Any] = []

        self.tool_registry = ToolRegistry()

        self.memory = memory
        self.policy = policy
        self.audit = audit
        self.error_handler = error_handler

    def initialize(self) -> None:
        """
        Initialize the agent and move it to READY state.
        """

        self.lifecycle.initialize()

        self.lifecycle.mark_ready()

    def start(self) -> None:
        """
        Start the agent.
        """

        self.lifecycle.start()

    def stop(self) -> None:
        """
        Stop the agent.
        """

        self.lifecycle.stop()
        self.lifecycle.stopped()

    def create_context(
        self,
        **kwargs: Any,
    ) -> AgentContext:
        """
        Create a common AgentContext for this agent.
        """

        return AgentContext(
            agent_identity=self.identity,
            memory=self.memory,
            available_tools=self.tool_registry.list_tools(),
            **kwargs,
        )

    @abstractmethod
    def handle_event(
        self,
        event: Any,
        context: AgentContext,
    ) -> Any:
        """
        Process an incoming event.

        Specialist agents must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        request: Any,
        context: AgentContext,
    ) -> Any:
        """
        Execute an agent-specific request.

        Specialist agents must implement this method.
        """
        raise NotImplementedError
