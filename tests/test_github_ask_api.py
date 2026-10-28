"""API tests for POST /github/ask using an injected, mocked Groq service."""

import asyncio
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.github.ask import create_github_ask_router
from app.api.github.reads import create_github_read_router
from app.composition import create_github_runtime
from app.services.github_request_service import GitHubRequestService
from github_agent import InMemoryActivityStore
from github_agent.testing import MockGitHubClient
from tests.test_github_request_service import StubGroqService, llm_response


class TestGitHubAskAPI(unittest.TestCase):
    def setUp(self):
        self.runtime = create_github_runtime(activity_store=InMemoryActivityStore())
        self.client = MockGitHubClient()
        self.runtime.agent.tool_registry.get("get_repository").execute.__self__._client = self.client
        self.groq = StubGroqService(
            response=llm_response("github_read", "get_repository", {"owner": "octo", "repo": "demo"})
        )
        self.runtime.github_request_service = GitHubRequestService(
            groq_service=self.groq,
            tool_registry=self.runtime.agent.tool_registry,
            tool_executor=self.runtime.tool_executor,
        )
        self.app = FastAPI()
        self.app.state.github_runtime = self.runtime
        self.app.include_router(create_github_ask_router())
        self.app.include_router(create_github_read_router())

    def request(self, method, path, **kwargs):
        async def send():
            async with AsyncClient(
                transport=ASGITransport(app=self.app),
                base_url="http://test",
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def test_success_uses_existing_runtime_tool_executor_and_audit(self):
        with patch("app.composition.create_github_runtime") as create_runtime:
            response = self.request(
                "POST",
                "/github/ask",
                json={"message": "Get repository octo/demo"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["tool_id"], "get_repository")
        self.assertEqual(body["answer"], self.groq.answer)
        self.assertEqual(body["result"]["full_name"], "octo/demo")
        self.assertEqual(self.client.calls[-1][0], "get_repository")
        self.assertEqual(self.runtime.audit.list_events()[-1].result, "success")
        create_runtime.assert_not_called()

    def test_answer_generation_failure_returns_safe_error_and_keeps_reads_only(self):
        self.groq.answer_error = TimeoutError("private model detail")

        response = self.request(
            "POST",
            "/github/ask",
            json={"message": "Get repository octo/demo"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "GitHub returned data, but a natural-language answer could not be generated.",
        )
        self.assertNotIn("private model detail", response.text)
        self.assertEqual(self.client.calls[-1][0], "get_repository")

    def test_unsupported_and_clarification_responses_are_safe(self):
        self.groq.response = llm_response("unsupported")
        unsupported = self.request("POST", "/github/ask", json={"message": "Tell me a joke"})
        self.assertEqual(unsupported.status_code, 200)
        self.assertEqual(unsupported.json()["status"], "unsupported")

        self.groq.response = llm_response(
            "clarification_required",
            clarification="Which repository should I check?",
        )
        clarification = self.request("POST", "/github/ask", json={"message": "List pull requests"})
        self.assertEqual(clarification.status_code, 200)
        self.assertEqual(clarification.json()["status"], "clarification_required")
        self.assertEqual(self.client.calls, [])

    def test_malformed_model_response_keeps_safe_http_error_and_skips_github(self):
        self.groq.response = "malformed private model output"

        response = self.request(
            "POST",
            "/github/ask",
            json={"message": "Get repository octo/demo"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "The GitHub request could not be interpreted safely.",
        )
        self.assertEqual(self.client.calls, [])

    def test_groq_request_exception_details_are_not_exposed_by_api(self):
        private_exception_detail = (
            "Invalid schema for Get repository octo/demo; "
            "Authorization: Bearer super-secret-token; API_KEY=private-key-value"
        )
        provider_error = RuntimeError(private_exception_detail)
        provider_error.body = {
            "error": {
                "code": "invalid_json_schema",
                "message": private_exception_detail,
            }
        }
        self.groq.error = provider_error

        with self.assertLogs("app.services.github_request_service", level="ERROR") as captured:
            response = self.request(
                "POST",
                "/github/ask",
                json={"message": "Get repository octo/demo"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "The GitHub request service is temporarily unavailable.",
        )
        self.assertNotIn(private_exception_detail, response.text)
        logs = "\n".join(captured.output)
        self.assertIn("Groq request failed exception_type=RuntimeError", logs)
        self.assertIn("provider_code=invalid_json_schema", logs)
        self.assertIn("Invalid schema for [redacted]", logs)
        self.assertNotIn(private_exception_detail, logs)
        self.assertNotIn("octo/demo", logs)
        self.assertNotIn("super-secret-token", logs)
        self.assertNotIn("private-key-value", logs)
        self.assertNotIn("Invalid schema for", response.text)
        self.assertEqual(self.client.calls, [])

    def test_existing_github_read_route_still_uses_same_runtime(self):
        response = self.request("GET", "/github/repository?owner=octo&repo=demo")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["full_name"], "octo/demo")
        self.assertEqual(self.client.calls[-1][0], "get_repository")

    def test_message_length_limit_is_enforced(self):
        response = self.request("POST", "/github/ask", json={"message": "x" * 4001})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.groq.prompts, [])
        self.assertEqual(self.client.calls, [])

    def test_blank_message_is_rejected(self):
        response = self.request("POST", "/github/ask", json={"message": "   "})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.groq.prompts, [])

    def test_route_does_not_construct_a_runtime_when_state_is_missing(self):
        app = FastAPI()
        app.include_router(create_github_ask_router())
        original_app = self.app
        self.app = app
        try:
            with patch("app.composition.create_github_runtime") as create_runtime:
                response = self.request("POST", "/github/ask", json={"message": "List branches"})
            self.assertEqual(response.status_code, 503)
            create_runtime.assert_not_called()
        finally:
            self.app = original_app

    def test_production_application_registers_ask_post_route(self):
        from app.main import app

        self.assertIn("post", app.openapi()["paths"]["/github/ask"])


if __name__ == "__main__":
    unittest.main()
