from shared.agent_core.audit import AuditService
from shared.agent_core.config import AgentConfig
from shared.agent_core.error_handler import ErrorHandler
from shared.agent_core.identity import AgentIdentity
from shared.agent_core.memory import InMemoryStore
from shared.agent_core.policy import PolicyEngine
from shared.agent_core.context import AgentContext
from shared.agent_core.base_agent import BaseAgent


class TestAgent(BaseAgent):

    def handle_event(
        self,
        event,
        context: AgentContext,
    ):
        return {
            "status": "event_handled",
            "event": event,
        }

    def execute(
        self,
        request,
        context: AgentContext,
    ):
        return {
            "status": "request_executed",
            "request": request,
        }


identity = AgentIdentity(
    agent_id="test-agent",
    name="Test Agent",
    agent_type="specialist",
    domain="testing",
    version="1.0.0",
)

config = AgentConfig(
    identity=identity,
)

memory = InMemoryStore()
policy = PolicyEngine()
audit = AuditService()
error_handler = ErrorHandler()

agent = TestAgent(
    identity=identity,
    config=config,
    memory=memory,
    policy=policy,
    audit=audit,
    error_handler=error_handler,
)

print("Initial status:", agent.lifecycle.status)

agent.initialize()

print("After initialize:", agent.lifecycle.status)

agent.start()

print("After start:", agent.lifecycle.status)

context = agent.create_context(
    event={"type": "test_event"},
)

print("Context agent:", context.agent_identity.name)
print("Context tools:", context.available_tools)

event_result = agent.handle_event(
    {"type": "test_event"},
    context,
)

print("Event result:", event_result)

request_result = agent.execute(
    {"action": "test_request"},
    context,
)

print("Request result:", request_result)

agent.stop()

print("Final status:", agent.lifecycle.status)

print()
print("BaseAgent integration test passed.")
