import unittest

from github_agent import GitHubAgent, GITHUB_AGENT_IDENTITY
from shared.agent_core.audit import AuditService
from shared.agent_core.base_agent import BaseAgent
from shared.agent_core.config import AgentConfig
from shared.agent_core.context import AgentContext
from shared.agent_core.error_handler import ErrorHandler
from shared.agent_core.memory import InMemoryStore
from shared.agent_core.policy import PolicyEngine
from shared.agent_core.state import AgentStatus
from github_agent.webhook import GitHubEventNormalizer


class TestGitHubAgent(unittest.TestCase):
    def setUp(self):
        self.agent = GitHubAgent(
            config=AgentConfig(identity=GITHUB_AGENT_IDENTITY),
            memory=InMemoryStore(),
            policy=PolicyEngine(),
            audit=AuditService(),
            error_handler=ErrorHandler(),
        )

    def test_identity_and_base_agent_inheritance(self):
        self.assertIsInstance(self.agent, BaseAgent)
        self.assertEqual(self.agent.identity, GITHUB_AGENT_IDENTITY)
        self.assertEqual(self.agent.identity.agent_id, "github-agent")
        self.assertEqual(self.agent.identity.agent_type, "specialist")
        self.assertEqual(self.agent.identity.domain, "github")

    def test_initialization_start_and_stop(self):
        self.agent.initialize()
        self.assertEqual(self.agent.lifecycle.status, AgentStatus.READY)

        self.agent.start()
        self.assertEqual(self.agent.lifecycle.status, AgentStatus.RUNNING)

        self.agent.stop()
        self.assertEqual(self.agent.lifecycle.status, AgentStatus.STOPPED)

    def test_github_capabilities_registered(self):
        self.assertEqual(
            {capability.capability_id for capability in self.agent.capabilities},
            {
                "github.event_processing",
                "github.repository_monitoring",
                "github.activity_tracking",
            },
        )

    def test_context_creation_uses_common_context_and_memory(self):
        context = self.agent.create_context(event={"action": "opened"})

        self.assertIsInstance(context, AgentContext)
        self.assertEqual(context.agent_identity, self.agent.identity)
        self.assertIs(context.memory, self.agent.memory)
        self.assertEqual(context.event, {"action": "opened"})
        self.assertEqual(context.available_tools, self.agent.tool_registry.list_tools())

    def test_handle_event_returns_minimal_result(self):
        context = self.agent.create_context()
        event = GitHubEventNormalizer.normalize(
            "delivery-test",
            "repository",
            {
                "action": "created",
                "repository": {"name": "demo", "full_name": "octo/demo"},
            },
        )

        result = self.agent.handle_event(event, context)

        self.assertEqual(result["status"], "event_processed")
        self.assertIs(context.event, event)
        activity = self.agent.activity_store.get(result["activity_id"])
        self.assertEqual(activity.event_id, "delivery-test")

    def test_execute_returns_minimal_result(self):
        context = self.agent.create_context()
        request = {"capability": "github.event_processing"}

        result = self.agent.execute(request, context)

        self.assertEqual(result, {"status": "request_received", "request": request})
        self.assertIs(context.request, request)


if __name__ == "__main__":
    unittest.main()
