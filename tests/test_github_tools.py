import json
import os
import unittest
from datetime import datetime, timezone
from email.message import Message
from io import BytesIO
from urllib.parse import parse_qs, urlsplit
from urllib.error import HTTPError
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
from shared.agent_core.errors import AgentError, ErrorCategory
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
        with patch.dict(os.environ, {"GH_TEST_TOKEN": "   "}):
            with self.assertRaises(AgentError) as caught:
                EnvironmentTokenProvider("GH_TEST_TOKEN").get_token()
        self.assertEqual(caught.exception.category, ErrorCategory.CONFIGURATION)
        self.assertFalse(caught.exception.retryable)

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

    def test_rest_client_lists_branches_using_fixed_read_endpoint(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'[{"name":"main"}]'

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            return Response()

        client = GitHubRESTClient(MockGitHubAuthProvider(), opener=opener)
        self.assertEqual(client.list_branches("octo", "demo", per_page=5), [{"name": "main"}])
        self.assertEqual(captured["url"], "https://api.github.com/repos/octo/demo/branches?per_page=5")
        self.assertEqual(captured["method"], "GET")

    def test_rest_client_lists_contributors_using_fixed_read_endpoint_and_pagination(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'[]'

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            return Response()

        client = GitHubRESTClient(MockGitHubAuthProvider(), opener=opener)
        self.assertEqual(client.list_contributors("octo", "demo", per_page=5, page=2), [])
        self.assertEqual(
            captured["url"],
            "https://api.github.com/repos/octo/demo/contributors?per_page=5&page=2",
        )
        self.assertEqual(captured["method"], "GET")

    def test_rest_client_searches_issues_using_fixed_search_endpoint(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"total_count":1,"incomplete_results":false,"items":[]}'

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            return Response()

        client = GitHubRESTClient(MockGitHubAuthProvider(), opener=opener)
        query = 'repo:octo/demo is:issue in:title,body is:open "login failure"'
        response = client.search_issues(query, per_page=5, page=2)
        parsed_url = urlsplit(captured["url"])
        self.assertEqual(parsed_url.path, "/search/issues")
        self.assertEqual(parse_qs(parsed_url.query), {"q": [query], "per_page": ["5"], "page": ["2"]})
        self.assertEqual(captured["method"], "GET")
        self.assertEqual(response["total_count"], 1)

    def test_rest_client_lists_pull_request_files_using_fixed_read_endpoint(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'[{"filename":"src/example.py","status":"modified"}]'

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            return Response()

        client = GitHubRESTClient(MockGitHubAuthProvider(), opener=opener)
        self.assertEqual(
            client.get_pull_request_files("octo", "demo", 6, per_page=100, page=2),
            [{"filename": "src/example.py", "status": "modified"}],
        )
        self.assertEqual(
            captured["url"],
            "https://api.github.com/repos/octo/demo/pulls/6/files?per_page=100&page=2",
        )
        self.assertEqual(captured["method"], "GET")

    def test_rest_client_lists_pull_request_commits_using_fixed_read_endpoint(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'[{"sha":"abc1234","commit":{"message":"Example"}}]'

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            return Response()

        client = GitHubRESTClient(MockGitHubAuthProvider(), opener=opener)
        self.assertEqual(
            client.get_pull_request_commits("octo", "demo", 6, per_page=100, page=2),
            [{"sha": "abc1234", "commit": {"message": "Example"}}],
        )
        self.assertEqual(
            captured["url"],
            "https://api.github.com/repos/octo/demo/pulls/6/commits?per_page=100&page=2",
        )
        self.assertEqual(captured["method"], "GET")

    def test_rest_client_retries_transient_get_once(self):
        calls = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"name":"demo"}'

        def opener(request, timeout):
            calls.append(request.get_method())
            if len(calls) == 1:
                raise HTTPError(request.full_url, 503, "private response details", Message(), BytesIO())
            return Response()

        client = GitHubRESTClient(
            MockGitHubAuthProvider(), opener=opener,
            retry_delay_seconds=0, sleeper=lambda _: None,
        )
        self.assertEqual(client.get_repository("octo", "demo"), {"name": "demo"})
        self.assertEqual(calls, ["GET", "GET"])

    def test_rest_client_does_not_retry_auth_failure_or_leak_token(self):
        calls = []
        token = "private-test-token"

        def opener(request, timeout):
            calls.append(request)
            raise HTTPError(request.full_url, 401, token, Message(), BytesIO(token.encode()))

        client = GitHubRESTClient(
            MockGitHubAuthProvider(token), opener=opener,
            retry_delay_seconds=0, sleeper=lambda _: None,
        )
        with self.assertRaises(AgentError) as caught:
            client.get_repository("octo", "demo")
        self.assertEqual(caught.exception.code, "GITHUB_AUTHENTICATION_FAILED")
        self.assertEqual(caught.exception.category, ErrorCategory.AUTHENTICATION)
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn(token, str(caught.exception))
        self.assertEqual(len(calls), 1)

    def test_rest_client_marks_rate_limit_retryable_without_immediate_retry(self):
        calls = []

        def opener(request, timeout):
            calls.append(request)
            headers = Message()
            headers["X-RateLimit-Remaining"] = "0"
            raise HTTPError(request.full_url, 403, "rate limited", headers, BytesIO())

        client = GitHubRESTClient(
            MockGitHubAuthProvider(), opener=opener,
            retry_delay_seconds=0, sleeper=lambda _: None,
        )
        with self.assertRaises(AgentError) as caught:
            client.get_repository("octo", "demo")
        self.assertEqual(caught.exception.code, "GITHUB_API_ERROR")
        self.assertEqual(caught.exception.category, ErrorCategory.EXTERNAL_SERVICE)
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(len(calls), 1)

    def test_gateway_exposes_all_approved_read_operations(self):
        calls = [
            ("get_repository", ("octo", "demo"), {}),
            ("list_repositories", (), {}),
            ("get_repository_activity", ("octo", "demo"), {}),
            ("list_branches", ("octo", "demo"), {}),
            ("list_contributors", ("octo", "demo"), {}),
            ("list_pull_requests", ("octo", "demo"), {}),
            ("get_pull_request", ("octo", "demo", 7), {}),
            ("get_pull_request_files", ("octo", "demo", 7), {}),
            ("get_pull_request_commits", ("octo", "demo", 7), {}),
            ("get_pull_request_reviews", ("octo", "demo", 7), {}),
            ("get_pull_request_review_comments", ("octo", "demo", 7), {}),
            ("list_issues", ("octo", "demo"), {}),
            ("get_issue", ("octo", "demo", 8), {}),
            ("search_issues", ("octo", "demo", "database"), {}),
            ("get_issue_comments", ("octo", "demo", 8), {}),
            ("list_commits", ("octo", "demo"), {}),
            ("get_commit", ("octo", "demo", "abc123"), {}),
        ]
        for operation, args, kwargs in calls:
            with self.subTest(operation=operation):
                result = getattr(self.gateway, operation)(*args, **kwargs)
                self.assertIsNotNone(result)
        self.assertEqual(len(self.client.calls), 17)

    def test_gateway_sorts_contributors_and_enforces_requested_limit(self):
        self.client.data["list_contributors"] = [
            {"login": "third", "contributions": 2},
            {"login": "first", "contributions": 20},
            {"login": "second", "contributions": 8},
        ]

        result = self.gateway.list_contributors("octo", "demo", limit=2, page=3)

        self.assertEqual([item["login"] for item in result], ["first", "second"])
        self.assertEqual(
            self.client.calls[-1],
            ("list_contributors", ("octo", "demo"), {"per_page": 2, "page": 3}),
        )
        for invalid in (0, 101, True):
            with self.subTest(limit=invalid), self.assertRaises(AgentError):
                self.gateway.list_contributors("octo", "demo", limit=invalid)

    def test_gateway_redacts_credential_like_contributor_fields(self):
        client = MockGitHubClient({
            "list_contributors": [
                {"login": "TOKEN=private-value", "contributions": 1, "html_url": "https://example.test"}
            ]
        })
        result = GitHubToolGateway(client).list_contributors("octo", "demo")
        self.assertEqual(result[0]["login"], "[redacted credential]")
        self.assertNotIn("private-value", repr(result))

    def test_gateway_scopes_issue_search_filters_pull_requests_and_projects_safe_fields(self):
        self.client.data["search_issues"] = {
            "total_count": 2,
            "incomplete_results": False,
            "items": [
                {
                    "number": 8, "title": "TOKEN=private-title-token", "state": "open",
                    "user": {"login": "octo", "email": "private@example.test"},
                    "labels": [{"name": "bug", "color": "ff0000"}], "body": "unneeded body",
                    "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-02T00:00:00Z",
                    "html_url": "https://github.com/octo/demo/issues/8", "token": "private metadata",
                },
                {"number": 9, "title": "A pull request result", "pull_request": {"url": "ignored"}},
            ],
        }

        result = self.gateway.search_issues("octo", "demo", "authentication", state="open", per_page=5, page=2)

        self.assertEqual(
            self.client.calls[-1],
            (
                "search_issues",
                ('repo:octo/demo is:issue in:title,body is:open "authentication"',),
                {"per_page": 5, "page": 2},
            ),
        )
        self.assertEqual(result["total_count"], 2)
        self.assertEqual(result["page"], 2)
        self.assertEqual(len(result["items"]), 1)
        issue = result["items"][0]
        self.assertEqual(issue["title"], "[redacted credential]")
        self.assertEqual(issue["author"], {"login": "octo"})
        self.assertEqual(issue["labels"], [{"name": "bug"}])
        self.assertNotIn("body", issue)
        self.assertNotIn("email", repr(issue))
        self.assertNotIn("private metadata", repr(result))

    def test_gateway_issue_search_supports_closed_and_all_states_and_escapes_keyword(self):
        for state, qualifier in (("closed", " is:closed"), ("all", "")):
            with self.subTest(state=state):
                self.gateway.search_issues("octo", "demo", 'login "failure"', state=state)
                operation, args, options = self.client.calls[-1]
                self.assertEqual(operation, "search_issues")
                self.assertIn(f"repo:octo/demo is:issue in:title,body{qualifier}", args[0])
                self.assertIn('"login \\"failure\\""', args[0])
                self.assertEqual(options, {"per_page": 30, "page": 1})

    def test_issue_statistics_counts_states_excludes_pull_requests_and_paginates(self):
        pages = {
            1: [
                {"number": 1, "state": "open"},
                {"number": 2, "state": "closed"},
                {"number": 3, "state": "open", "pull_request": {"url": "ignored"}},
            ] + [{"number": number, "state": "open"} for number in range(4, 101)],
            2: [{"number": 4, "state": "closed"}],
        }

        class PagedClient:
            def __init__(self):
                self.calls = []

            def list_issues(self, owner, repo, **options):
                self.calls.append((owner, repo, options))
                return pages[options["page"]]

        client = PagedClient()
        result = GitHubToolGateway(client).get_issue_statistics("octo", "demo")

        self.assertEqual(result["repository"], "octo/demo")
        self.assertEqual((result["open_count"], result["closed_count"], result["total_count"]), (98, 2, 100))
        self.assertEqual((result["open_percentage"], result["closed_percentage"]), (98.0, 2.0))
        self.assertTrue(result["complete"])
        self.assertEqual([call[2]["page"] for call in client.calls], [1, 2])
        self.assertTrue(all(call[2]["state"] == "all" and call[2]["per_page"] == 100 for call in client.calls))

    def test_issue_statistics_handles_empty_and_single_state_repositories(self):
        for records, expected in (
            ([], (0, 0, 0, 0.0, 0.0)),
            ([{"state": "open"}, {"state": "open"}], (2, 0, 2, 100.0, 0.0)),
            ([{"state": "closed"}], (0, 1, 1, 0.0, 100.0)),
        ):
            with self.subTest(records=records):
                client = MockGitHubClient({"list_issues": records})
                result = GitHubToolGateway(client).get_issue_statistics("octo", "demo")
                actual = (
                    result["open_count"], result["closed_count"], result["total_count"],
                    result["open_percentage"], result["closed_percentage"],
                )
                self.assertEqual(actual, expected)
                self.assertTrue(result["complete"])

    def test_issue_statistics_marks_failed_pages_incomplete_without_percentages(self):
        class FailingPageClient:
            def __init__(self):
                self.calls = []

            def list_issues(self, owner, repo, **options):
                self.calls.append(options["page"])
                if options["page"] == 1:
                    return [{"state": "open"} for _ in range(100)]
                raise AgentError(
                    "GITHUB_API_ERROR", "safe error", ErrorCategory.EXTERNAL_SERVICE, True,
                    details={"status": 503},
                )

        client = FailingPageClient()
        result = GitHubToolGateway(client).get_issue_statistics("octo", "demo", max_pages=3)
        self.assertFalse(result["complete"])
        self.assertEqual(result["open_count"], 100)
        self.assertIsNone(result["open_percentage"])
        self.assertIn("page 2", result["incomplete_reason"])
        self.assertEqual(client.calls, [1, 2])

    def test_issue_statistics_marks_safety_limit_and_validates_max_pages(self):
        class FullPageClient:
            def list_issues(self, owner, repo, **options):
                return [{"state": "open"} for _ in range(100)]

        result = GitHubToolGateway(FullPageClient()).get_issue_statistics("octo", "demo", max_pages=1)
        self.assertFalse(result["complete"])
        self.assertEqual(result["pages_retrieved"], 1)
        self.assertIn("1-page safety limit", result["incomplete_reason"])
        with self.assertRaises(AgentError):
            GitHubToolGateway(FullPageClient()).get_issue_statistics("octo", "demo", max_pages=101)

    def test_list_issues_preserves_filters_and_excludes_pull_requests(self):
        self.client.data["list_issues"] = [
            {"number": 1, "state": "open"},
            {"number": 2, "state": "open", "pull_request": {"url": "ignored"}},
        ]
        result = self.gateway.list_issues(
            "octo", "demo", state="all", labels="bug", assignee="octocat", creator="hubot",
            since="2026-01-01T00:00:00Z", sort="updated", direction="asc", per_page=5, page=2,
        )
        self.assertEqual([item["number"] for item in result], [1])
        self.assertEqual(
            self.client.calls[-1],
            ("list_issues", ("octo", "demo"), {
                "state": "all", "labels": "bug", "assignee": "octocat", "creator": "hubot",
                "since": "2026-01-01T00:00:00Z", "sort": "updated", "direction": "asc",
                "per_page": 5, "page": 2,
            }),
        )

    def test_list_issues_created_after_uses_search_date_qualifier_and_filters_pull_requests(self):
        self.client.data["search_issues"] = {
            "total_count": 2,
            "incomplete_results": False,
            "items": [
                {"number": 3, "state": "open", "created_at": "2026-02-03T00:00:00Z"},
                {"number": 4, "state": "open", "pull_request": {"url": "ignored"}},
            ],
        }
        result = self.gateway.list_issues(
            "octo", "demo", created_after="2026-02-01", state="open", assignee="octocat",
            creator="hubot", labels="bug,needs review", sort="created", direction="desc", page=2,
        )
        operation, args, options = self.client.calls[-1]
        self.assertEqual(operation, "search_issues")
        for qualifier in (
            "repo:octo/demo", "is:issue", "created:>=2026-02-01", "is:open",
            "assignee:octocat", "author:hubot", 'label:"bug"', 'label:"needs review"',
        ):
            self.assertIn(qualifier, args[0])
        self.assertEqual(options, {"per_page": 30, "page": 2, "sort": "created", "order": "desc"})
        self.assertEqual([issue["number"] for issue in result], [3])

    def test_list_issues_my_assigned_uses_authenticated_user_search_qualifier(self):
        self.client.data["search_issues"] = {"items": [{"number": 1}]}
        result = self.gateway.list_issues("octo", "demo", assignee="@me")
        operation, args, _ = self.client.calls[-1]
        self.assertEqual(operation, "search_issues")
        self.assertIn("assignee:@me", args[0])
        self.assertEqual(result, [{"number": 1}])

    def test_list_issues_validates_state_sort_and_creation_date(self):
        for arguments in (
            {"state": "merged"}, {"sort": "number"}, {"direction": "sideways"},
            {"created_after": "2026-02-30"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(AgentError):
                self.gateway.list_issues("octo", "demo", **arguments)

    def test_stale_issue_report_classifies_thresholds_and_uses_timezone_aware_dates(self):
        now = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
        records = [
            {"number": 1, "title": "Recent", "state": "open", "updated_at": "2026-10-29T12:00:00Z"},
            {"number": 2, "title": "Stale", "state": "open", "updated_at": "2026-10-28T12:00:00Z"},
            {"number": 3, "title": "Very stale", "state": "closed", "updated_at": "2026-10-21T13:00:00+01:00"},
            {"number": 4, "title": "Critical", "state": "closed", "updated_at": "2026-10-05T12:00:00Z"},
            {"number": 5, "title": "Missing", "state": "open"},
            {"number": 6, "title": "Invalid", "state": "open", "updated_at": "not-a-date"},
        ]
        self.client.data["list_issues"] = records
        with patch.object(GitHubToolGateway, "_now_utc", return_value=now):
            report = self.gateway.list_issues("octo", "demo", state="all", stale_days=7)
        by_number = {issue["number"]: issue for issue in report["items"]}
        self.assertEqual(set(by_number), {2, 3, 4})
        self.assertEqual(by_number[2]["stale_classification"], "Stale")
        self.assertEqual(by_number[3]["stale_classification"], "Very stale")
        self.assertEqual(by_number[4]["stale_classification"], "Critical stale")
        self.assertEqual(by_number[3]["inactive_days"], 14)
        self.assertEqual(report["unclassifiable_count"], 2)
        self.assertTrue(report["complete"])

    def test_stale_issue_report_applies_custom_cutoff_state_and_assignee_filters(self):
        self.client.data["list_issues"] = [
            {"number": 8, "title": "Ten day issue", "state": "closed", "updated_at": "2026-10-25T12:00:00Z", "assignees": [{"login": "dev"}]},
            {"number": 9, "title": "Seven day issue", "state": "open", "updated_at": "2026-10-28T12:00:00Z"},
        ]
        now = __import__("datetime").datetime(2026, 11, 4, 12, tzinfo=__import__("datetime").timezone.utc)
        with patch.object(GitHubToolGateway, "_now_utc", return_value=now):
            report = self.gateway.list_issues(
                "octo", "demo", state="closed", assignee="dev", stale_days=10,
            )
        self.assertEqual([issue["number"] for issue in report["items"]], [8])
        self.assertEqual(self.client.calls[0][2]["state"], "closed")
        self.assertEqual(self.client.calls[0][2]["assignee"], "dev")
        self.client.calls.clear()
        with patch.object(GitHubToolGateway, "_now_utc", return_value=now):
            self.gateway.list_issues("octo", "demo", state="open", stale_days=10)
        self.assertEqual(self.client.calls[0][2]["state"], "open")

    def test_stale_issue_report_marks_bounded_page_and_search_limits_incomplete(self):
        full_page = [{"number": index + 1, "state": "open", "updated_at": "2026-01-01T00:00:00Z"} for index in range(100)]
        self.client.data["list_issues"] = full_page
        with patch.object(
            GitHubToolGateway, "_now_utc",
            return_value=datetime(2026, 11, 4, tzinfo=timezone.utc),
        ):
            report = self.gateway.list_issues("octo", "demo", stale_days=7)
        self.assertFalse(report["complete"])
        self.assertIn("safety limit", report["incomplete_reason"])
        self.assertEqual(report["pages_retrieved"], 10)

        self.client.calls.clear()
        self.client.data["search_issues"] = {
            "total_count": 1200, "incomplete_results": False,
            "items": [{"number": 9, "state": "open", "updated_at": "2026-01-01T00:00:00Z"}],
        }
        with patch.object(
            GitHubToolGateway, "_now_utc",
            return_value=datetime(2026, 11, 4, tzinfo=timezone.utc),
        ):
            report = self.gateway.list_issues("octo", "demo", assignee="@me", stale_days=7)
        self.assertFalse(report["complete"])
        self.assertIn("1,000", report["incomplete_reason"])

        self.client.data["search_issues"] = {
            "total_count": 5, "incomplete_results": True,
            "items": [{"number": 10, "state": "open", "updated_at": "2026-01-01T00:00:00Z"}],
        }
        with patch.object(
            GitHubToolGateway, "_now_utc", return_value=datetime(2026, 11, 4, tzinfo=timezone.utc),
        ):
            report = self.gateway.list_issues("octo", "demo", assignee="@me", stale_days=7)
        self.assertFalse(report["complete"])
        self.assertIn("incomplete results", report["incomplete_reason"])

    def test_gateway_filters_merged_pull_requests_from_closed_github_results(self):
        pull_requests = [
            {"number": 5, "state": "closed", "merged_at": None},
            {"number": 4, "state": "closed", "merged_at": "2026-10-12T07:02:26Z"},
            {"number": 3, "state": "closed", "merged_at": None},
            {"number": 2, "state": "open", "merged_at": None},
        ]
        self.client.data["list_pull_requests"] = pull_requests

        result = self.gateway.list_pull_requests("octo", "demo", merged_only=True)

        self.assertEqual(result, [pull_requests[1]])
        self.assertEqual(
            self.client.calls[-1],
            ("list_pull_requests", ("octo", "demo"), {"state": "closed"}),
        )

    def test_gateway_filters_closed_unmerged_pull_requests(self):
        pull_requests = [
            {"number": 5, "state": "closed", "merged_at": None},
            {"number": 4, "state": "closed", "merged_at": "2026-10-12T07:02:26Z"},
            {"number": 3, "state": "closed", "merged_at": None},
            {"number": 2, "state": "open", "merged_at": None},
        ]
        self.client.data["list_pull_requests"] = pull_requests

        result = self.gateway.list_pull_requests("octo", "demo", unmerged_only=True)

        self.assertEqual(result, [pull_requests[0], pull_requests[2]])
        self.assertEqual(
            self.client.calls[-1],
            ("list_pull_requests", ("octo", "demo"), {"state": "closed"}),
        )

    def test_gateway_returns_empty_list_when_no_closed_unmerged_pull_requests_exist(self):
        self.client.data["list_pull_requests"] = [
            {"number": 4, "state": "closed", "merged_at": "2026-10-12T07:02:26Z"},
            {"number": 2, "state": "open", "merged_at": None},
        ]

        self.assertEqual(
            self.gateway.list_pull_requests("octo", "demo", unmerged_only=True),
            [],
        )

    def test_gateway_keeps_ordinary_pull_request_state_results_unchanged(self):
        closed_pull_requests = [
            {"number": 5, "state": "closed", "merged_at": None},
            {"number": 4, "state": "closed", "merged_at": "2026-10-12T07:02:26Z"},
            {"number": 3, "state": "closed", "merged_at": None},
        ]
        open_pull_requests = [{"number": 6, "state": "open", "merged_at": None}]
        all_pull_requests = closed_pull_requests + open_pull_requests

        for state, data in (
            ("closed", closed_pull_requests),
            ("open", open_pull_requests),
            ("all", all_pull_requests),
        ):
            with self.subTest(state=state):
                self.client.data["list_pull_requests"] = data
                result = self.gateway.list_pull_requests("octo", "demo", state=state)
                self.assertEqual(result, data)
                self.assertEqual(self.client.calls[-1][2], {"state": state})

    def test_gateway_requests_all_pull_request_states_when_state_is_unspecified(self):
        all_pull_requests = [
            {"number": 5, "state": "closed", "merged_at": None},
            {"number": 4, "state": "closed", "merged_at": "2026-10-12T07:02:26Z"},
            {"number": 6, "state": "open", "merged_at": None},
        ]
        self.client.data["list_pull_requests"] = all_pull_requests

        result = self.gateway.list_pull_requests("octo", "demo")

        self.assertEqual(result, all_pull_requests)
        self.assertEqual(
            self.client.calls[-1],
            ("list_pull_requests", ("octo", "demo"), {"state": "all"}),
        )

    def test_gateway_returns_empty_list_when_no_merged_pull_requests_exist(self):
        self.client.data["list_pull_requests"] = [
            {"number": 5, "state": "closed", "merged_at": None},
            {"number": 3, "state": "closed", "merged_at": None},
        ]

        self.assertEqual(
            self.gateway.list_pull_requests("octo", "demo", merged_only=True),
            [],
        )

    def test_gateway_sanitizes_sensitive_response_fields(self):
        client = MockGitHubClient({
            "get_repository": {
                "name": "demo",
                "token": "secret",
                "refresh_token": "hidden-token",
                "nested": {"password": "hidden", "api_secret": "hidden-secret"},
            }
        })
        result = GitHubToolGateway(client).get_repository("octo", "demo")
        self.assertEqual(result, {"name": "demo", "nested": {}})

    def test_gateway_redacts_credentials_inside_pull_request_patches(self):
        client = MockGitHubClient(
            {
                "get_pull_request_files": [
                    {
                        "filename": "src/example.py",
                        "patch": "+API_KEY=private-value\n+Bearer ghp_12345678901234567890",
                    }
                ]
            }
        )

        result = GitHubToolGateway(client).get_pull_request_files("octo", "demo", 6)

        self.assertIn("[redacted credential]", result[0]["patch"])
        self.assertIn("Bearer [redacted]", result[0]["patch"])
        self.assertNotIn("private-value", result[0]["patch"])
        self.assertNotIn("ghp_12345678901234567890", result[0]["patch"])

    def test_gateway_redacts_credentials_inside_pull_request_commit_messages(self):
        client = MockGitHubClient(
            {
                "get_pull_request_commits": [
                    {
                        "sha": "abc1234",
                        "commit": {
                            "message": "Set API_KEY=private-value and use Bearer ghp_12345678901234567890",
                            "author": {"name": "Example Author", "email": "author@example.test"},
                            "committer": {"name": "Example Committer", "email": "committer@example.test"},
                        },
                    }
                ]
            }
        )

        result = GitHubToolGateway(client).get_pull_request_commits("octo", "demo", 6)

        self.assertNotIn("private-value", result[0]["commit"]["message"])
        self.assertNotIn("ghp_12345678901234567890", result[0]["commit"]["message"])
        self.assertIn("[redacted credential]", result[0]["commit"]["message"])
        self.assertNotIn("email", result[0]["commit"]["author"])
        self.assertNotIn("email", result[0]["commit"]["committer"])

    def test_gateway_logs_only_safe_agent_error_fields(self):
        secret_detail = "Authorization: Bearer private-token api-key=private-key"

        class FailingClient:
            def get_repository(self, owner, repo):
                raise AgentError(
                    code="GITHUB_API_ERROR",
                    message=secret_detail,
                    category=ErrorCategory.EXTERNAL_SERVICE,
                    retryable=True,
                    details={"status": 404, "private_detail": secret_detail},
                )

        gateway = GitHubToolGateway(FailingClient())
        with self.assertLogs("github_agent.gateway", level="ERROR") as captured:
            with self.assertRaises(AgentError):
                gateway.get_repository("octo", "demo")

        logs = "\n".join(captured.output)
        self.assertIn("operation=get_repository", logs)
        self.assertIn("error_code=GITHUB_API_ERROR", logs)
        self.assertIn("category=external_service", logs)
        self.assertIn("retryable=True", logs)
        self.assertIn("http_status=404", logs)
        self.assertNotIn(secret_detail, logs)
        self.assertNotIn("private-token", logs)
        self.assertNotIn("private-key", logs)

    def test_gateway_logs_only_operation_and_exception_type_for_unexpected_errors(self):
        secret_detail = "private URL query and Authorization: Bearer private-token"

        class FailingClient:
            def get_repository(self, owner, repo):
                raise RuntimeError(secret_detail)

        gateway = GitHubToolGateway(FailingClient())

        with self.assertLogs("github_agent.gateway", level="ERROR") as captured:
            with self.assertRaises(AgentError):
                gateway.get_repository("octo", "demo")

        logs = "\n".join(captured.output)
        self.assertIn("operation=get_repository", logs)
        self.assertIn("exception_type=RuntimeError", logs)
        self.assertNotIn(secret_detail, logs)
        self.assertNotIn("private-token", logs)

    def test_tool_registration_and_lookup(self):
        self.assertEqual(len(self.agent.tool_registry.list_tools()), 18)
        tool = self.agent.tool_registry.get("get_repository")
        self.assertEqual(tool.permissions, ["github.read"])
        self.assertEqual(tool.risk_level, "low")
        self.assertIsNone(self.agent.tool_registry.get("call_github_api"))
        self.assertFalse(
            any("write" in tool.tool_id or "merge" in tool.tool_id
                for tool in self.agent.tool_registry.list_tools())
        )

    def test_valid_repository_pull_request_issue_and_commit_reads(self):
        self.assertEqual(self.execute("get_repository", {"owner": "octo", "repo": "demo"})["name"], "demo")
        self.assertEqual(
            self.execute("list_contributors", {"owner": "octo", "repo": "demo"})[0]["login"],
            "octocat",
        )
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

    def test_gateway_rejects_wrong_direct_input_type(self):
        with self.assertRaises(AgentError) as caught:
            self.gateway.get_repository(12, "demo")
        self.assertEqual(caught.exception.code, "INVALID_FIELD_TYPE")

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
