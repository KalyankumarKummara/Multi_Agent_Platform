import asyncio
import hashlib
import hmac
import json
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.webhooks.github import create_github_webhook_router
from app.composition import create_github_runtime
from github_agent import InMemoryActivityStore


WEBHOOK_SECRET = "unit-test-webhook-secret"


def valid_payload():
    return {
        "action": "created",
        "repository": {
            "id": 1,
            "name": "demo",
            "full_name": "octo/demo",
            "owner": {"login": "octo"},
        },
        "sender": {"id": 2, "login": "octocat"},
        "private_body": "must not appear in the response",
    }


class TestGitHubWebhookAPI(unittest.TestCase):
    def setUp(self):
        self.runtime = create_github_runtime(activity_store=InMemoryActivityStore())
        self.received_events = []
        self.runtime.agent.handle_event = (
            lambda event, context: self.received_events.append(event) or {"processed": True}
        )
        self.app = FastAPI()
        self.app.include_router(create_github_webhook_router(self.runtime))

    def post(self, body, *, event="repository", delivery="delivery-api-1", signature=None):
        if signature is None:
            signature = "sha256=" + hmac.new(
                WEBHOOK_SECRET.encode("utf-8"), body, hashlib.sha256
            ).hexdigest()
        headers = {
            "X-GitHub-Event": event,
            "X-GitHub-Delivery": delivery,
            "X-Hub-Signature-256": signature,
        }

        async def send_request():
            async with AsyncClient(
                transport=ASGITransport(app=self.app),
                base_url="http://testserver",
            ) as client:
                return await client.post("/webhooks/github", content=body, headers=headers)

        with patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": WEBHOOK_SECRET}):
            return asyncio.run(send_request())

    def test_valid_webhook_reaches_agent_and_returns_safe_response(self):
        body = json.dumps(valid_payload()).encode("utf-8")
        response = self.post(body)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "processed", "delivery_id": "delivery-api-1"})
        self.assertEqual(len(self.received_events), 1)
        self.assertEqual(self.received_events[0].event_type, "repository")

    def test_invalid_signature_is_rejected(self):
        body = json.dumps(valid_payload()).encode("utf-8")
        response = self.post(body, signature="sha256=" + "0" * 64)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Webhook signature was rejected."})
        self.assertEqual(self.received_events, [])

    def test_missing_webhook_secret_is_rejected_safely(self):
        body = json.dumps(valid_payload()).encode("utf-8")
        signature = "sha256=" + hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
        headers = {
            "X-GitHub-Event": "repository",
            "X-GitHub-Delivery": "no-secret",
            "X-Hub-Signature-256": signature,
        }

        async def send_request():
            async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://testserver") as client:
                return await client.post("/webhooks/github", content=body, headers=headers)

        with patch.dict(os.environ, {}, clear=True):
            response = asyncio.run(send_request())

        self.assertEqual(response.status_code, 503)
        self.assertNotIn(WEBHOOK_SECRET, response.text)
        self.assertNotIn(body.decode(), response.text)

    def test_missing_required_github_headers_are_rejected(self):
        body = json.dumps(valid_payload()).encode("utf-8")
        signature = "sha256=" + hmac.new(WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
        cases = [
            ({"X-GitHub-Delivery": "missing-event", "X-Hub-Signature-256": signature}, "WEBHOOK_EVENT_HEADER_MISSING"),
            ({"X-GitHub-Event": "repository", "X-Hub-Signature-256": signature}, "WEBHOOK_DELIVERY_ID_MISSING"),
        ]
        for headers, expected_message in cases:
            with self.subTest(expected_message=expected_message):
                async def send_request():
                    async with AsyncClient(transport=ASGITransport(app=self.app), base_url="http://testserver") as client:
                        return await client.post("/webhooks/github", content=body, headers=headers)

                with patch.dict(os.environ, {"GITHUB_WEBHOOK_SECRET": WEBHOOK_SECRET}):
                    response = asyncio.run(send_request())

                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.received_events, [])

    def test_duplicate_delivery_returns_duplicate_response(self):
        body = json.dumps(valid_payload()).encode("utf-8")
        first = self.post(body, delivery="duplicate-api")
        second = self.post(body, delivery="duplicate-api")

        self.assertEqual(first.json()["status"], "processed")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json(), {"status": "duplicate", "delivery_id": "duplicate-api"})
        self.assertEqual(len(self.received_events), 1)

    def test_malformed_json_is_rejected_without_echoing_body(self):
        body = b"{raw secret-like payload"
        response = self.post(body, delivery="malformed-json")

        self.assertEqual(response.status_code, 400)
        self.assertNotIn(body.decode(), response.text)
        self.assertNotIn(WEBHOOK_SECRET, response.text)
        self.assertEqual(self.received_events, [])

    def test_retryable_processing_failure_can_be_retried_safely(self):
        body = json.dumps(valid_payload()).encode("utf-8")
        original_handler = self.runtime.agent.handle_event
        attempt = 0

        def fail_once(event, context):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise RuntimeError(f"private token {WEBHOOK_SECRET}; payload {body.decode()}")
            return original_handler(event, context)

        self.runtime.agent.handle_event = fail_once
        first = self.post(body, delivery="retry-api-delivery")
        second = self.post(body, delivery="retry-api-delivery")

        self.assertEqual(first.status_code, 503)
        self.assertNotIn(WEBHOOK_SECRET, first.text)
        self.assertNotIn(body.decode(), first.text)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["status"], "processed")
        self.assertEqual(len(self.received_events), 1)


if __name__ == "__main__":
    unittest.main()
