import hashlib
import hmac
import json
import unittest

from github_agent import GITHUB_AGENT_IDENTITY, GitHubAgent
from github_agent.webhook import (
    GitHubEventNormalizer,
    GitHubWebhookHandler,
    GitHubWebhookSignatureVerifier,
    InMemoryDeliveryStore,
    NormalizedGitHubEvent,
)
from shared.agent_core.audit import AuditService
from shared.agent_core.config import AgentConfig
from shared.agent_core.error_handler import ErrorHandler
from shared.agent_core.errors import AgentError
from shared.agent_core.memory import InMemoryStore
from shared.agent_core.policy import PolicyEngine


SECRET = "local-test-webhook-secret"


def payload_for(event_type="repository"):
    base = {
        "repository": {
            "id": 42,
            "name": "demo",
            "full_name": "octo/demo",
            "owner": {"login": "octo"},
            "html_url": "https://github.com/octo/demo",
            "private": False,
        },
        "sender": {"id": 9, "login": "octocat", "type": "User"},
    }
    if event_type == "repository":
        return {**base, "action": "created"}
    if event_type == "push":
        return {
            **base,
            "ref": "refs/heads/main",
            "before": "oldsha",
            "after": "newsha",
            "commits": [{"id": "newsha", "message": "update", "timestamp": "2026-01-01T00:00:00Z", "added": ["secret-path"]}],
            "head_commit": {"timestamp": "2026-01-01T00:00:00Z", "author": {"email": "private@example.test"}},
        }
    if event_type in ("pull_request", "pull_request_review", "pull_request_review_comment"):
        result = {**base, "action": "opened", "pull_request": {"number": 7, "title": "Improve", "state": "open", "updated_at": "2026-01-01T00:00:00Z", "body": "unneeded", "base": {"ref": "main"}, "head": {"ref": "feature"}}}
        if event_type == "pull_request_review":
            result["action"] = "submitted"
            result["review"] = {"id": 11, "state": "approved", "body": "private review body", "submitted_at": "2026-01-01T00:00:00Z"}
        if event_type == "pull_request_review_comment":
            result["action"] = "created"
            result["comment"] = {"id": 12, "path": "app.py", "line": 4, "body": "private comment", "created_at": "2026-01-01T00:00:00Z"}
        return result
    if event_type in ("issues", "issue_comment"):
        result = {**base, "action": "opened", "issue": {"number": 8, "title": "Bug", "state": "open", "body": "private issue body", "updated_at": "2026-01-01T00:00:00Z"}}
        if event_type == "issue_comment":
            result["action"] = "created"
            result["comment"] = {"id": 13, "body": "private comment", "created_at": "2026-01-01T00:00:00Z"}
        return result
    if event_type == "create":
        return {**base, "ref": "feature", "ref_type": "branch", "master_branch": "main"}
    return base


class TestGitHubWebhook(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryDeliveryStore()
        self.handler = GitHubWebhookHandler(delivery_store=self.store)
        self.agent = GitHubAgent(
            config=AgentConfig(identity=GITHUB_AGENT_IDENTITY),
            memory=InMemoryStore(),
            policy=PolicyEngine(),
            audit=AuditService(),
            error_handler=ErrorHandler(),
        )
        self.received = []
        self.agent.handle_event = lambda event, context: self.received.append((event, context)) or {"handled": event.event_id}

    def request(self, event_type="repository", delivery_id="delivery-001", payload=None, secret=SECRET):
        raw_body = json.dumps(payload if payload is not None else payload_for(event_type), separators=(",", ":")).encode("utf-8")
        signature = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
        return raw_body, {
            "X-GitHub-Event": event_type,
            "X-GitHub-Delivery": delivery_id,
            "X-Hub-Signature-256": signature,
        }

    def handle(self, **kwargs):
        raw_body, headers = self.request(**kwargs)
        return self.handler.handle(raw_body, headers, kwargs.get("secret", SECRET), self.agent)

    def test_signature_valid_invalid_missing_malformed_and_empty_secret(self):
        raw_body, headers = self.request()
        verifier = GitHubWebhookSignatureVerifier()
        verifier.verify(raw_body, headers["X-Hub-Signature-256"], SECRET)
        for signature, secret, expected in [
            ("sha256=" + "0" * 64, SECRET, "WEBHOOK_SIGNATURE_INVALID"),
            (None, SECRET, "WEBHOOK_SIGNATURE_MISSING"),
            ("sha256=bad", SECRET, "WEBHOOK_SIGNATURE_MALFORMED"),
            (headers["X-Hub-Signature-256"], "  ", "WEBHOOK_SECRET_MISSING"),
        ]:
            with self.subTest(expected=expected), self.assertRaises(AgentError) as caught:
                verifier.verify(raw_body, signature, secret)
            self.assertEqual(caught.exception.code, expected)
            self.assertNotIn(SECRET, str(caught.exception))

    def test_signature_covers_exact_raw_body(self):
        raw_body, headers = self.request()
        altered = raw_body + b" "
        with self.assertRaises(AgentError) as caught:
            self.handler.handle(altered, headers, SECRET, self.agent)
        self.assertEqual(caught.exception.code, "WEBHOOK_SIGNATURE_INVALID")

    def test_signature_is_verified_before_json_parsing(self):
        raw_body = b"{malformed-private-body"
        headers = {
            "X-GitHub-Event": "repository",
            "X-GitHub-Delivery": "bad-signature-first",
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
        }
        with self.assertRaises(AgentError) as caught:
            self.handler.handle(raw_body, headers, SECRET, self.agent)
        self.assertEqual(caught.exception.code, "WEBHOOK_SIGNATURE_INVALID")
        self.assertNotIn(raw_body.decode(), str(caught.exception))

    def test_missing_event_and_delivery_headers(self):
        raw_body, headers = self.request()
        headers.pop("X-GitHub-Event")
        with self.assertRaises(AgentError) as caught:
            self.handler.handle(raw_body, headers, SECRET, self.agent)
        self.assertEqual(caught.exception.code, "WEBHOOK_EVENT_HEADER_MISSING")

        raw_body, headers = self.request()
        headers.pop("X-GitHub-Delivery")
        with self.assertRaises(AgentError) as caught:
            self.handler.handle(raw_body, headers, SECRET, self.agent)
        self.assertEqual(caught.exception.code, "WEBHOOK_DELIVERY_ID_MISSING")

    def test_supported_event_processed_and_unsupported_event_rejected(self):
        result = self.handle(event_type="repository")
        self.assertEqual(result.status, "processed")
        self.assertEqual(self.received[0][0].event_type, "repository")
        raw_body, headers = self.request(event_type="ping", delivery_id="unsupported")
        with self.assertRaises(AgentError) as caught:
            self.handler.handle(raw_body, headers, SECRET, self.agent)
        self.assertEqual(caught.exception.code, "WEBHOOK_EVENT_UNSUPPORTED")

    def test_malformed_json_and_required_event_fields(self):
        raw_body = b"{"
        signature = "sha256=" + hmac.new(SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
        headers = {"X-GitHub-Event": "repository", "X-GitHub-Delivery": "bad-json", "X-Hub-Signature-256": signature}
        with self.assertRaises(AgentError) as caught:
            self.handler.handle(raw_body, headers, SECRET, self.agent)
        self.assertEqual(caught.exception.code, "WEBHOOK_JSON_INVALID")
        self.assertNotIn(raw_body.decode(), str(caught.exception))

        raw_body, headers = self.request(event_type="push", payload={"repository": payload_for()["repository"]})
        with self.assertRaises(AgentError) as caught:
            self.handler.handle(raw_body, headers, SECRET, self.agent)
        self.assertEqual(caught.exception.code, "WEBHOOK_PAYLOAD_INVALID")

    def test_normalizes_repository_push_pull_request_issue_and_comment(self):
        cases = [
            ("repository", "repository_action"),
            ("push", "ref"),
            ("create", "ref"),
            ("pull_request", "pull_request"),
            ("pull_request_review", "review"),
            ("pull_request_review_comment", "comment"),
            ("issues", "issue"),
            ("issue_comment", "comment"),
        ]
        for index, (event_type, key) in enumerate(cases):
            with self.subTest(event_type=event_type):
                event = GitHubEventNormalizer.normalize(str(index), event_type, payload_for(event_type))
                self.assertIsInstance(event, NormalizedGitHubEvent)
                self.assertEqual(event.source, "github")
                self.assertIn(key, event.metadata)
                self.assertEqual(event.repository["full_name"], "octo/demo")
                self.assertEqual(event.actor["login"], "octocat")
                self.assertNotIn("body", repr(event.metadata))
                self.assertNotIn("private@example.test", repr(event.metadata))

    def test_idempotency_first_duplicate_and_reset(self):
        first = self.handle(delivery_id="same-delivery")
        duplicate = self.handle(delivery_id="same-delivery")
        self.assertEqual(first.status, "processed")
        self.assertEqual(duplicate.status, "duplicate")
        self.assertEqual(len(self.received), 1)
        self.store.clear()
        self.assertEqual(self.handle(delivery_id="same-delivery").status, "processed")
        self.assertEqual(len(self.received), 2)

    def test_normalized_event_and_common_context_reach_agent(self):
        result = self.handle(event_type="push")
        event, context = self.received[0]
        self.assertEqual(result.result, {"handled": "delivery-001"})
        self.assertIs(context.event, event)
        self.assertIsInstance(event, NormalizedGitHubEvent)
        self.assertEqual(event.event_type, "push")

    def test_agent_failure_is_sanitized_and_delivery_can_retry(self):
        self.agent.handle_event = lambda event, context: (_ for _ in ()).throw(RuntimeError(f"failed with {SECRET} and payload"))
        raw_body, headers = self.request(delivery_id="retry-me")
        with self.assertRaises(AgentError) as caught:
            self.handler.handle(raw_body, headers, SECRET, self.agent)
        self.assertEqual(caught.exception.code, "WEBHOOK_AGENT_PROCESSING_FAILED")
        self.assertTrue(caught.exception.retryable)
        self.assertNotIn(SECRET, str(caught.exception))
        self.assertNotIn("payload", str(caught.exception))

        self.agent.handle_event = lambda event, context: "ok"
        retried = self.handler.handle(raw_body, headers, SECRET, self.agent)
        self.assertEqual(retried.status, "processed")

    def test_empty_secret_error_does_not_expose_secret_or_payload(self):
        raw_body, headers = self.request()
        with self.assertRaises(AgentError) as caught:
            self.handler.handle(raw_body, headers, "", self.agent)
        self.assertEqual(caught.exception.code, "WEBHOOK_SECRET_MISSING")
        self.assertNotIn(SECRET, repr(caught.exception))
        self.assertNotIn(raw_body.decode(), str(caught.exception))


if __name__ == "__main__":
    unittest.main()
