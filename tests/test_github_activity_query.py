"""Focused tests for the validated GitHub activity query layer."""

import unittest
from datetime import datetime, timezone

from github_agent import ActivityRecord, GitHubAgent, GITHUB_AGENT_IDENTITY, InMemoryActivityStore
from github_agent.activity_query import ActivityQueryError, GitHubActivityQuery
from shared.agent_core.audit import AuditService
from shared.agent_core.config import AgentConfig
from shared.agent_core.error_handler import ErrorHandler
from shared.agent_core.memory import InMemoryStore
from shared.agent_core.policy import PolicyEngine


def record(
    activity_id: str,
    *,
    repository: str = "octo/demo",
    event_type: str = "repository",
    status: str = "processed",
    significance: str = "low",
    occurred_at: str | None = "2026-10-01T12:00:00+00:00",
    processed_hour: int = 12,
) -> ActivityRecord:
    return ActivityRecord(
        activity_id=activity_id,
        event_id=f"delivery-{activity_id}",
        source="github",
        agent="github-agent",
        event_type=event_type,
        action="created",
        repository={"full_name": repository},
        actor=None,
        target={"type": "repository"},
        occurred_at=occurred_at,
        processed_at=datetime(2026, 10, 1, processed_hour, tzinfo=timezone.utc),
        status=status,
        metadata={},
        significance=significance,
    )


class TestGitHubActivityQuery(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryActivityStore()
        self.query = GitHubActivityQuery(self.store)
        for item in (
            record("alpha", significance="high", processed_hour=12),
            record("beta", repository="octo/other", event_type="push", significance="medium", occurred_at="2026-10-02T12:00:00Z", processed_hour=13),
            record("gamma", status="failed", significance="low", occurred_at="2026-09-30T12:00:00Z", processed_hour=14),
            record("no-time", occurred_at=None, processed_hour=15),
        ):
            self.store.save(item)

    def test_repository_event_status_significance_and_time_filters(self):
        self.assertEqual([item.activity_id for item in self.query.list_recent_activity(repository="octo/demo")], ["no-time", "gamma", "alpha"])
        self.assertEqual([item.activity_id for item in self.query.list_recent_activity(event_type="push")], ["beta"])
        self.assertEqual([item.activity_id for item in self.query.list_recent_activity(status="failed")], ["gamma"])
        self.assertEqual([item.activity_id for item in self.query.list_recent_activity(significance="high")], ["alpha"])
        self.assertEqual([item.activity_id for item in self.query.list_recent_activity(occurred_at_from="2026-10-01T00:00:00Z")], ["beta", "alpha"])
        self.assertEqual([item.activity_id for item in self.query.list_recent_activity(occurred_at_to="2026-10-01T23:59:59Z")], ["gamma", "alpha"])

    def test_combined_filters_limit_and_deterministic_ordering(self):
        self.store.save(record("aardvark", processed_hour=12))
        found = self.query.list_recent_activity(
            repository="octo/demo",
            status="processed",
            occurred_at_from=datetime(2026, 10, 1, tzinfo=timezone.utc),
            limit=2,
        )
        self.assertEqual([item.activity_id for item in found], ["aardvark", "alpha"])
        self.assertTrue(all(isinstance(item, ActivityRecord) for item in found))

    def test_convenience_queries(self):
        self.assertEqual(self.query.list_repository_activity("octo/other")[0].activity_id, "beta")
        self.assertEqual(self.query.list_high_significance_activity()[0].activity_id, "alpha")
        self.assertEqual(self.query.list_activity_by_event_type("push")[0].activity_id, "beta")

    def test_invalid_filters_are_rejected(self):
        for filters in (
            {"limit": 0},
            {"limit": 101},
            {"limit": True},
            {"repository": "../demo"},
            {"significance": "urgent"},
            {"event_type": "arbitrary"},
            {"occurred_at_from": "not-a-time"},
            {"occurred_at_from": "2026-10-02T00:00:00Z", "occurred_at_to": "2026-10-01T00:00:00Z"},
        ):
            with self.subTest(filters=filters), self.assertRaises(ActivityQueryError):
                self.query.list_recent_activity(**filters)

    def test_agent_queries_via_injected_domain_service_and_returns_records(self):
        agent = GitHubAgent(
            config=AgentConfig(identity=GITHUB_AGENT_IDENTITY),
            memory=InMemoryStore(),
            policy=PolicyEngine(),
            audit=AuditService(),
            error_handler=ErrorHandler(),
            activity_store=self.store,
            activity_query=self.query,
        )
        result = agent.query_activity(repository="octo/other")
        self.assertEqual([item.activity_id for item in result], ["beta"])
        self.assertIs(agent.activity_query, self.query)
        with self.assertRaises(ActivityQueryError):
            agent.query_activity(limit=1000)

    def test_agent_module_has_no_mssql_or_sqlalchemy_dependency(self):
        import github_agent.agent as agent_module

        self.assertFalse(hasattr(agent_module, "MSSQLActivityStore"))
        self.assertFalse(hasattr(agent_module, "SQLAlchemy"))


if __name__ == "__main__":
    unittest.main()
