import hashlib
import hmac
import json
import unittest

from github_agent import ActivityRecord, GITHUB_AGENT_IDENTITY, GitHubAgent, GitHubWebhookHandler, InMemoryActivityStore
from github_agent.webhook import GitHubEventNormalizer
from shared.agent_core.audit import AuditService
from shared.agent_core.config import AgentConfig
from shared.agent_core.error_handler import ErrorHandler
from shared.agent_core.errors import AgentError
from shared.agent_core.memory import InMemoryStore
from shared.agent_core.policy import PolicyEngine
from infrastructure.database.mssql.activity_store import ActivityPersistenceError


SECRET = "activity-tests-webhook-secret"


def repository_payload():
    return {
        "action": "created",
        "repository": {
            "id": 72,
            "name": "demo",
            "full_name": "octo/demo",
            "owner": {"login": "octo", "email": "private-owner@example.test"},
            "private": True,
        },
        "sender": {"id": 12, "login": "octocat", "email": "private-actor@example.test"},
        "body": "private raw webhook payload marker",
        "token": "private-token-marker",
    }


def push_payload():
    return {
        "repository": {"name": "demo", "full_name": "octo/demo", "owner": {"login": "octo"}},
        "sender": {"login": "octocat"},
        "ref": "refs/heads/main",
        "before": "old",
        "after": "new",
        "commits": [{"id": "new", "message": "safe commit summary", "added": ["private.txt"]}],
    }


def make_agent(store=None):
    activity_store = store if store is not None else InMemoryActivityStore()
    agent = GitHubAgent(
        config=AgentConfig(identity=GITHUB_AGENT_IDENTITY),
        memory=InMemoryStore(),
        policy=PolicyEngine(),
        audit=AuditService(),
        error_handler=ErrorHandler(),
        activity_store=activity_store,
    )
    return agent, activity_store


class FailingActivityStore(InMemoryActivityStore):
    def save(self, activity):
        raise RuntimeError("simulated activity storage failure")


class FailOnceDatabaseActivityStore(InMemoryActivityStore):
    def __init__(self):
        super().__init__()
        self.fail_next_save = True

    def save(self, activity):
        if self.fail_next_save:
            self.fail_next_save = False
            raise ActivityPersistenceError("GitHub activity database operation failed.")
        return super().save(activity)


class TestGitHubActivity(unittest.TestCase):
    def setUp(self):
        self.agent, self.store = make_agent()

    @staticmethod
    def normalize(event_id, event_type="repository", payload=None):
        data = payload if payload is not None else repository_payload()
        return GitHubEventNormalizer.normalize(event_id, event_type, data)

    @staticmethod
    def signed_request(delivery_id, event_type, payload):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        signature = "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        headers = {
            "X-GitHub-Event": event_type,
            "X-GitHub-Delivery": delivery_id,
            "X-Hub-Signature-256": signature,
        }
        return body, headers

    def deliver(self, handler, delivery_id, event_type, payload):
        body, headers = self.signed_request(delivery_id, event_type, payload)
        return handler.handle(body, headers, SECRET, self.agent)

    def test_activity_record_creation(self):
        event = self.normalize("delivery-record")
        record = ActivityRecord.from_event(event, agent_id="github-agent")

        self.assertTrue(record.activity_id.startswith("activity-"))
        self.assertEqual(record.event_id, event.event_id)
        self.assertEqual(record.source, event.source)
        self.assertEqual(record.agent, "github-agent")
        self.assertEqual(record.event_type, "repository")
        self.assertEqual(record.action, "created")
        self.assertEqual(record.repository, event.repository)
        self.assertEqual(record.actor, event.actor)
        self.assertEqual(record.target["type"], "repository")
        self.assertEqual(record.status, "processed")

    def test_store_save_get_list_and_filters(self):
        first = ActivityRecord.from_event(self.normalize("one"), agent_id="github-agent")
        second = ActivityRecord.from_event(
            self.normalize("two", "push", push_payload()),
            agent_id="github-agent",
        )
        self.store.save(first)
        self.store.save(second)

        self.assertEqual(self.store.get(first.activity_id).event_id, "one")
        self.assertEqual(len(self.store.list_activities()), 2)
        self.assertEqual(len(self.store.list_activities(repository="octo/demo")), 2)
        self.assertEqual(len(self.store.list_activities(event_type="push")), 1)
        self.assertEqual(len(self.store.list_activities(status="processed")), 2)
        self.assertIsNone(self.store.get("unknown-activity"))

    def test_successful_github_event_creates_activity(self):
        event = self.normalize("delivery-success")
        context = self.agent.create_context()

        result = self.agent.handle_event(event, context)

        self.assertEqual(result["status"], "event_processed")
        saved = self.store.get(result["activity_id"])
        self.assertEqual(saved.event_id, event.event_id)
        self.assertEqual(saved.status, "processed")
        self.assertEqual(saved.significance, "medium")

    def test_duplicate_webhook_does_not_create_duplicate_activity(self):
        handler = GitHubWebhookHandler()

        first = self.deliver(handler, "same-delivery", "repository", repository_payload())
        second = self.deliver(handler, "same-delivery", "repository", repository_payload())

        self.assertEqual(first.status, "processed")
        self.assertEqual(second.status, "duplicate")
        self.assertEqual(len(self.store.list_activities()), 1)

    def test_persisted_delivery_retry_is_idempotent_after_handler_restart(self):
        first_handler = GitHubWebhookHandler()
        retry_handler = GitHubWebhookHandler()
        payload = push_payload()

        self.deliver(first_handler, "restart-retry-id", "push", payload)
        retried = self.deliver(retry_handler, "restart-retry-id", "push", payload)

        self.assertEqual(retried.status, "processed")
        self.assertEqual(len(self.store.list_activities()), 1)
        self.assertEqual(self.store.list_activities()[0].event_id, "restart-retry-id")

    def test_identical_push_details_with_distinct_delivery_ids_are_preserved(self):
        handler = GitHubWebhookHandler()
        payload = {
            **push_payload(),
            "head_commit": {"timestamp": "2026-01-01T00:00:00Z"},
        }

        self.deliver(handler, "similar-push-one", "push", payload)
        self.deliver(handler, "similar-push-two", "push", payload)

        activities = self.store.list_activities()
        self.assertEqual(len(activities), 2)
        self.assertEqual({item.event_id for item in activities}, {"similar-push-one", "similar-push-two"})
        self.assertEqual({item.event_type for item in activities}, {"push"})
        self.assertEqual({item.occurred_at for item in activities}, {"2026-01-01T00:00:00Z"})
        self.assertEqual(activities[0].repository, activities[1].repository)
        self.assertEqual(activities[0].actor, activities[1].actor)

    def test_different_events_create_separate_activity_records(self):
        handler = GitHubWebhookHandler()
        self.deliver(handler, "repository-delivery", "repository", repository_payload())
        self.deliver(handler, "push-delivery", "push", push_payload())

        activities = self.store.list_activities()
        self.assertEqual(len(activities), 2)
        self.assertEqual(
            {activity.event_id for activity in activities},
            {"repository-delivery", "push-delivery"},
        )

    def test_create_and_push_for_same_branch_persist_separately_with_retry_idempotency(self):
        handler = GitHubWebhookHandler()
        create_payload = {"repository": push_payload()["repository"], "ref": "feature", "ref_type": "branch"}
        branch_push = {**push_payload(), "ref": "refs/heads/feature", "created": True, "deleted": False, "commits": []}
        self.deliver(handler, "branch-create-id", "create", create_payload)
        self.deliver(handler, "branch-push-id", "push", branch_push)
        self.deliver(handler, "branch-create-id", "create", create_payload)

        activities = self.store.list_activities()
        self.assertEqual(len(activities), 2)
        self.assertEqual({item.event_type for item in activities}, {"create", "push"})
        create = next(item for item in activities if item.event_type == "create")
        self.assertIsNone(create.occurred_at)
        self.assertEqual(create.metadata["ref_type"], "branch")
        self.assertIsNotNone(create.processed_at)

    def test_delete_branch_is_persisted(self):
        handler = GitHubWebhookHandler()
        payload = {"repository": push_payload()["repository"], "ref": "feature", "ref_type": "branch"}
        self.deliver(handler, "branch-delete-id", "delete", payload)
        activity = self.store.list_activities()[0]
        self.assertEqual(activity.event_type, "delete")
        self.assertEqual(activity.metadata, {"ref": "feature", "ref_type": "branch"})

    def test_failed_processing_does_not_record_success_and_can_retry(self):
        failing_agent, failing_store = make_agent(FailingActivityStore())
        body, headers = self.signed_request("failure-delivery", "repository", repository_payload())
        handler = GitHubWebhookHandler()

        for _ in range(2):
            with self.assertRaises(AgentError) as caught:
                handler.handle(body, headers, SECRET, failing_agent)
            self.assertEqual(caught.exception.code, "WEBHOOK_AGENT_PROCESSING_FAILED")
            self.assertEqual(failing_store.list_activities(), [])

    def test_database_failure_releases_delivery_for_retry(self):
        database_store = FailOnceDatabaseActivityStore()
        database_agent, _ = make_agent(database_store)
        handler = GitHubWebhookHandler()
        body, headers = self.signed_request(
            "database-retry-delivery", "repository", repository_payload()
        )

        with self.assertRaises(AgentError) as caught:
            handler.handle(body, headers, SECRET, database_agent)
        self.assertEqual(caught.exception.code, "WEBHOOK_AGENT_PROCESSING_FAILED")
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(database_store.list_activities(), [])

        result = handler.handle(body, headers, SECRET, database_agent)
        self.assertEqual(result.status, "processed")
        self.assertEqual(len(database_store.list_activities()), 1)

    def test_only_normalized_minimized_data_is_stored(self):
        event = self.normalize("safe-delivery", payload=repository_payload())
        self.agent.handle_event(event, self.agent.create_context())
        record_text = repr(self.store.list_activities()[0])

        self.assertIn("octo/demo", record_text)
        self.assertNotIn("private raw webhook payload marker", record_text)
        self.assertNotIn("private-token-marker", record_text)
        self.assertNotIn("private-owner@example.test", record_text)
        self.assertNotIn("private-actor@example.test", record_text)
        self.assertNotIn('"private": True', record_text)

    def test_agent_rejects_raw_payload_without_saving(self):
        with self.assertRaises(ValueError):
            self.agent.handle_event({"body": "raw"}, self.agent.create_context())
        self.assertEqual(self.store.list_activities(), [])


if __name__ == "__main__":
    unittest.main()
