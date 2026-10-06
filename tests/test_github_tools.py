import json
import os
import unittest
from unittest.mock import patch

from github_agent import (
    EnvironmentTokenProvider,
    GitHubAgent,
    GitHubRESTClient,
    GitHubToolGateway,
)
from github_agent.testing import MockGitHubAuthProvider, MockGitHubClient
from shared.agent_core.approval import ApprovalService
from shared.agent_core.audit import AuditService
from shared.agent_core.config import AgentConfig
from shared.agent_core.error_handler import ErrorHandler
from shared.agent_core.errors import AgentError
from shared.agent_core.memory import InMemoryStore
from shared.agent_core.policy import PolicyEngine
from shared.agent_core.security import AuthorizationService, Principal
from shared.agent_core.tools.execution import ToolExecutionContext
from shared.agent_core.tools.executor import ToolExecutor
from shared.agent_core.tools.tool import Tool
from shared.agent_core.tracing import create_trace_context
from shared.agent_core.validation import InputValidator
from github_agent import GITHUB_AGENT_IDENTITY


class TestGitHubTools(unittest.TestCase):
    def setUp(self):
        self.client = MockGitHubClient()
        self.gateway = GitHubToolGateway(self.client)
        self.audit = AuditService()
        self.agent = GitHubAgent(
            config=AgentConfig(identity=GITHUB_AGENT_IDENTITY),
            memory=InMemoryStore(),
            policy=PolicyEngine(),
            audit=self.audit,
            error_handler=ErrorHandler(),
            github_gateway=self.gateway,
        )
        self.executor = ToolExecutor(
            authorization_service=AuthorizationService(),
            policy_engine=PolicyEngine(),
            approval_service=ApprovalService(),
            audit_service=self.audit,
            input_validator=InputValidator(),
        )
        self.context = ToolExecutionContext(
            principal=Principal("test-agent", "agent", frozenset({"github.read"})),
            trace_context=create_trace_context(),
            permissions=["github.read"],
            metadata={},
        )

    def execute(self, tool_id, arguments):
        tool = self.agent.tool_registry.get(tool_id)
        self.assertIsNotNone(tool)
        return self.executor.execute(tool, arguments, self.context)

    def test_authentication_provider_is_injected_and_reads_env_on_demand(self):
        fake_provider = MockGitHubAuthProvider()
        self.assertEqual(fake_provider.get_token(), "test-token")
        with patch.dict(os.environ, {"GH_TEST_TOKEN": "injected-test-token"}):
            self.assertEqual(EnvironmentTokenProvider("GH_TEST_TOKEN").get_token(), "injected-test-token")

    def test_rest_client_constructs_read_request_with_injected_auth(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"name": "demo"}).encode()

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["auth"] = request.get_header("Authorization")
            captured["timeout"] = timeout
            return Response()

        client = GitHubRESTClient(MockGitHubAuthProvider("injected-test-token"), opener=opener)
        self.assertEqual(client.get_repository("octo", "demo"), {"name": "demo"})
        self.assertEqual(captured["url"], "https://api.github.com/repos/octo/demo")
        self.assertEqual(captured["method"], "GET")
        self.assertEqual(captured["auth"], "Bearer injected-test-token")

    def test_gateway_exposes_all_approved_read_operations(self):
        calls = [
            ("get_repository", ("octo", "demo"), {}),
            ("list_repositories", (), {}),
            ("get_repository_activity", ("octo", "demo"), {}),
            ("list_pull_requests", ("octo", "demo"), {}),
            ("get_pull_request", ("octo", "demo", 7), {}),
            ("get_pull_request_files", ("octo", "demo", 7), {}),
            ("get_pull_request_commits", ("octo", "demo", 7), {}),
            ("get_pull_request_reviews", ("octo", "demo", 7), {}),
            ("get_pull_request_review_comments", ("octo", "demo", 7), {}),
            ("list_issues", ("octo", "demo"), {}),
            ("get_issue", ("octo", "demo", 8), {}),
            ("get_issue_comments", ("octo", "demo", 8), {}),
            ("list_commits", ("octo", "demo"), {}),
            ("get_commit", ("octo", "demo", "abc123"), {}),
        ]
        for operation, args, kwargs in calls:
            with self.subTest(operation=operation):
                result = getattr(self.gateway, operation)(*args, **kwargs)
                self.assertIsNotNone(result)
        self.assertEqual(len(self.client.calls), 14)

    def test_gateway_sanitizes_sensitive_response_fields(self):
        client = MockGitHubClient({"get_repository": {"name": "demo", "token": "secret", "nested": {"password": "hidden"}}})
        result = GitHubToolGateway(client).get_repository("octo", "demo")
        self.assertEqual(result, {"name": "demo", "nested": {}})

    def test_tool_registration_and_lookup(self):
        self.assertEqual(len(self.agent.tool_registry.list_tools()), 14)
        tool = self.agent.tool_registry.get("get_repository")
        self.assertEqual(tool.permissions, ["github.read"])
        self.assertEqual(tool.risk_level, "low")
        self.assertIsNone(self.agent.tool_registry.get("call_github_api"))

    def test_valid_repository_pull_request_issue_and_commit_reads(self):
        self.assertEqual(self.execute("get_repository", {"owner": "octo", "repo": "demo"})["name"], "demo")
        self.assertEqual(self.execute("get_pull_request", {"owner": "octo", "repo": "demo", "pull_number": 7})["number"], 7)
        self.assertEqual(self.execute("get_issue", {"owner": "octo", "repo": "demo", "issue_number": 8})["number"], 8)
        self.assertEqual(self.execute("get_commit", {"owner": "octo", "repo": "demo", "commit_sha": "abc123"})["sha"], "abc123")

    def test_missing_invalid_and_unexpected_inputs_are_rejected(self):
        tool = self.agent.tool_registry.get("get_repository")
        for args, expected in [
            ({"owner": "octo"}, "MISSING_REQUIRED_FIELD"),
            ({"owner": "octo", "repo": 3}, "INVALID_FIELD_TYPE"),
            ({"owner": "octo", "repo": "demo", "extra": True}, "UNEXPECTED_FIELD"),
        ]:
            with self.subTest(expected=expected), self.assertRaises(AgentError) as caught:
                self.executor.execute(tool, args, self.context)
            self.assertEqual(caught.exception.code, expected)

    def test_authorization_denied_without_github_read_permission(self):
        denied_context = ToolExecutionContext(
            principal=Principal("no-read", "agent", frozenset()),
            trace_context=create_trace_context(), permissions=[], metadata={},
        )
        with self.assertRaises(AgentError) as caught:
            self.executor.execute(self.agent.tool_registry.get("get_repository"), {"owner": "octo", "repo": "demo"}, denied_context)
        self.assertEqual(caught.exception.code, "AUTHORIZATION_DENIED")
        self.assertFalse(self.client.calls)
        self.assertEqual(self.audit.list_events()[-1].result, "authorization_denied")

    def test_unknown_action_is_denied_by_policy(self):
        unknown = Tool("unknown_action", "Unknown", "Unknown", {}, {}, [], "low", lambda: None)
        with self.assertRaises(AgentError) as caught:
            self.executor.execute(unknown, {}, self.context)
        self.assertEqual(caught.exception.code, "POLICY_DENIED")

    def test_successful_tool_execution_is_audited(self):
        self.execute("get_repository", {"owner": "octo", "repo": "demo"})
        event = self.audit.list_events()[-1]
        self.assertEqual(event.action, "get_repository")
        self.assertEqual(event.result, "success")
        self.assertEqual(event.trace_id, self.context.trace_context.trace_id)

    def test_client_failure_is_sanitized_and_audited(self):
        failing_client = MockGitHubClient(fail_operations={"get_repository"})
        agent = GitHubAgent(
            config=AgentConfig(identity=GITHUB_AGENT_IDENTITY), memory=InMemoryStore(),
            policy=PolicyEngine(), audit=self.audit, error_handler=ErrorHandler(),
            github_client=failing_client,
        )
        with self.assertRaises(AgentError) as caught:
            self.executor.execute(agent.tool_registry.get("get_repository"), {"owner": "octo", "repo": "demo"}, self.context)
        self.assertEqual(caught.exception.code, "TOOL_EXECUTION_FAILED")
        self.assertNotIn("simulated", caught.exception.message)
        self.assertEqual(self.audit.list_events()[-1].result, "execution_failed")

    def test_agent_has_no_direct_api_client(self):
        self.assertFalse(hasattr(self.agent, "github_client"))
        self.assertIsNotNone(self.agent.tool_registry.get("get_repository"))


if __name__ == "__main__":
    unittest.main()
