"""Tests for the read-only GitHub demo API boundary."""

import asyncio
from datetime import datetime
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.github.activities import create_github_activity_router
from app.api.github.reads import create_github_read_router
from app.composition import create_github_runtime
from github_agent import InMemoryActivityStore
from github_agent.activity import ActivityRecord
from github_agent.webhook import GitHubEventNormalizer
from github_agent.testing import MockGitHubClient
from infrastructure.database.mssql.activity_store import ActivityPersistenceError
from shared.agent_core.errors import AgentError, ErrorCategory
from shared.agent_core.policy import PolicyDecision, PolicyResult


class TestGitHubDemoAPI(unittest.TestCase):
    def setUp(self):
        self.runtime = create_github_runtime(activity_store=InMemoryActivityStore())
        self.fake_client = MockGitHubClient(
            data={
                "get_repository": {"full_name": "octo/demo", "private": False},
                "list_repositories": [{"name": "demo", "full_name": "octo/demo"}],
                "list_commits": [{"sha": "abc123", "commit": {"message": "demo"}}],
                "list_branches": [{"name": "main", "protected": False}],
                "list_pull_requests": [{"number": 3, "title": "Example PR"}],
                "list_issues": [
                    {"number": 4, "title": "Example issue"},
                    {"number": 5, "title": "PR as issue", "pull_request": {"url": "safe"}},
                ],
            }
        )
        tool = self.runtime.agent.tool_registry.get("get_repository")
        tool.execute.__self__._client = self.fake_client
        for tool_id in ("list_commits", "list_pull_requests", "list_issues"):
            self.runtime.agent.tool_registry.get(tool_id).execute.__self__._client = self.fake_client

        self.app = FastAPI()
        self.app.include_router(create_github_read_router(self.runtime))
        self.app.include_router(create_github_activity_router(self.runtime))

    def request(self, method, path, **kwargs):
        async def send():
            async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def test_fixed_read_routes_use_named_agent_tools(self):
        cases = (
            ("/github/repository?owner=KalyankumarKummara&repo=github-agent-test", "get_repository"),
            ("/github/branches?owner=KalyankumarKummara&repo=github-agent-test", "list_branches"),
            ("/github/repositories?username=KalyankumarKummara", "list_repositories"),
            ("/github/commits?owner=KalyankumarKummara&repo=github-agent-test&per_page=5", "list_commits"),
            ("/github/pull-requests?owner=KalyankumarKummara&repo=github-agent-test", "list_pull_requests"),
            ("/github/issues?owner=KalyankumarKummara&repo=github-agent-test", "list_issues"),
        )
        for path, operation in cases:
            with self.subTest(operation=operation):
                response = self.request("GET", path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(self.fake_client.calls[-1][0], operation)
                if operation == "list_repositories":
                    self.assertEqual(self.fake_client.calls[-1][1][:2], ("KalyankumarKummara", None))
                else:
                    self.assertEqual(self.fake_client.calls[-1][1][:2], ("KalyankumarKummara", "github-agent-test"))

        issues = self.request("GET", "/github/issues?owner=KalyankumarKummara&repo=github-agent-test").json()
        self.assertEqual([issue["number"] for issue in issues], [4])

    def test_production_application_registers_demo_routes_without_startup(self):
        from app.main import app

        paths = app.openapi()["paths"]
        for path in (
            "/github/repository",
            "/github/repositories",
            "/github/branches",
            "/github/commits",
            "/github/pull-requests",
            "/github/issues",
            "/github/activities",
        ):
            with self.subTest(path=path):
                self.assertIn("get", paths[path])

    def test_tool_executor_audits_read_and_uses_server_owned_permission(self):
        response = self.request("GET", "/github/repository?owner=octo&repo=demo")
        self.assertEqual(response.status_code, 200)
        event = self.runtime.audit.list_events()[-1]
        self.assertEqual(event.action, "get_repository")
        self.assertEqual(event.result, "success")

    def test_authorization_denial_is_safe(self):
        with patch.object(self.runtime.authorization, "is_allowed", return_value=False):
            response = self.request("GET", "/github/repository?owner=octo&repo=demo")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "This GitHub read operation is not permitted.")

    def test_policy_denial_is_safe(self):
        with patch.object(
            self.runtime.policy,
            "evaluate",
            return_value=PolicyResult(PolicyDecision.DENY, "private policy detail"),
        ):
            response = self.request("GET", "/github/repository?owner=octo&repo=demo")
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("private policy detail", response.text)

    def test_invalid_arguments_and_tool_failures_are_safe(self):
        invalid = self.request("GET", "/github/commits?owner=octo&repo=demo&per_page=101")
        self.assertEqual(invalid.status_code, 422)

        self.fake_client.fail_operations.add("list_commits")
        failed = self.request("GET", "/github/commits?owner=octo&repo=demo")
        self.assertEqual(failed.status_code, 502)
        self.assertNotIn("simulated GitHub client failure", failed.text)

    def test_upstream_repository_permission_denial_is_explained_safely(self):
        class ForbiddenClient:
            def __getattr__(self, operation):
                def forbidden(*args, **kwargs):
                    raise AgentError(
                        code="GITHUB_API_ERROR",
                        message="GitHub API returned HTTP 403.",
                        category=ErrorCategory.EXTERNAL_SERVICE,
                        retryable=False,
                    )

                return forbidden

        self.runtime.agent.tool_registry.get("get_repository").execute.__self__._client = ForbiddenClient()
        response = self.request("GET", "/github/repository?owner=octo&repo=private")
        self.assertEqual(response.status_code, 502)
        self.assertIn("Check the configured token's repository access", response.json()["detail"])
        self.assertNotIn("GITHUB_API_ERROR", response.text)

    def test_upstream_not_found_is_reported_as_github_not_found(self):
        class NotFoundClient:
            def __getattr__(self, operation):
                def not_found(*args, **kwargs):
                    raise AgentError(
                        code="GITHUB_API_ERROR",
                        message="GitHub API returned HTTP 404.",
                        category=ErrorCategory.EXTERNAL_SERVICE,
                        retryable=False,
                        details={"status": 404},
                    )

                return not_found

        self.runtime.agent.tool_registry.get("get_repository").execute.__self__._client = NotFoundClient()
        response = self.request("GET", "/github/repository?owner=octo&repo=missing")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["detail"],
            "The GitHub repository or requested data was not found.",
        )

    def test_activity_storage_error_is_not_reported_as_empty_success(self):
        self.runtime.activity_query.list_recent_activity = lambda **kwargs: (_ for _ in ()).throw(
            ActivityPersistenceError("private database detail")
        )
        response = self.request("GET", "/github/activities")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Persisted GitHub activity storage is unavailable.")
        self.assertNotIn("private database detail", response.text)

    def test_activity_api_returns_processed_at_when_occurrence_is_unknown(self):
        event = GitHubEventNormalizer.normalize(
            "create-api-id",
            "create",
            {
                "repository": {"name": "demo", "full_name": "octo/demo"},
                "ref": "feature",
                "ref_type": "branch",
            },
        )
        saved = self.runtime.activity_store.save(
            ActivityRecord.from_event(event, agent_id="github-agent")
        )

        response = self.request("GET", "/github/activities?repository=octo/demo")
        self.assertEqual(response.status_code, 200)
        item = next(item for item in response.json() if item["event_id"] == "create-api-id")
        self.assertIsNone(item["occurred_at"])
        self.assertEqual(
            datetime.fromisoformat(item["processed_at"].replace("Z", "+00:00")),
            saved.processed_at,
        )

    def test_missing_registered_tool_returns_service_unavailable(self):
        self.runtime.agent.tool_registry.unregister("get_repository")
        response = self.request("GET", "/github/repository?owner=octo&repo=demo")
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
