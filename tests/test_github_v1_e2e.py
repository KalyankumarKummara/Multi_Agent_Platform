"""Deterministic signed-webhook to activity-query acceptance test."""

import hashlib
import hmac
import json
import unittest

from app.composition import create_github_runtime
from github_agent import InMemoryActivityStore


WEBHOOK_SECRET = "e2e-test-webhook-secret"
RAW_MARKER = "raw-payload-private-marker"


class TestGitHubV1EndToEnd(unittest.TestCase):
    def test_signed_event_is_processed_persisted_and_queryable_once(self):
        runtime = create_github_runtime(activity_store=InMemoryActivityStore())
        payload = {
            "action": "created",
            "repository": {
                "id": 73,
                "name": "demo",
                "full_name": "octo/demo",
                "owner": {"login": "octo", "token": RAW_MARKER},
            },
            "sender": {"id": 8, "login": "octocat"},
            "private_payload": RAW_MARKER,
        }
        raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        signature = "sha256=" + hmac.new(
            WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256
        ).hexdigest()
        headers = {
            "X-GitHub-Event": "repository",
            "X-GitHub-Delivery": "e2e-delivery-1",
            "X-Hub-Signature-256": signature,
        }

        first = runtime.webhook_handler.handle(raw_body, headers, WEBHOOK_SECRET, runtime.agent)
        duplicate = runtime.webhook_handler.handle(raw_body, headers, WEBHOOK_SECRET, runtime.agent)

        self.assertEqual(first.status, "processed")
        self.assertEqual(first.result["processing"]["classification"], "repository")
        self.assertEqual(first.result["processing"]["significance"], "medium")
        self.assertFalse(first.result["processing"]["requires_action"])
        self.assertIsNone(first.result["processing"]["recommended_action"])
        self.assertEqual(duplicate.status, "duplicate")

        activities = runtime.agent.query_activity(
            repository="octo/demo",
            event_type="repository",
            significance="medium",
            limit=10,
        )
        self.assertEqual(len(activities), 1)
        self.assertEqual(activities[0].event_id, "e2e-delivery-1")
        self.assertEqual(activities[0].significance, "medium")
        self.assertNotIn(RAW_MARKER, repr(activities[0]))
        self.assertNotIn(RAW_MARKER, repr(runtime.activity_store.list_activities()))


if __name__ == "__main__":
    unittest.main()
