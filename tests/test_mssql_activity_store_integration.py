"""Opt-in integration test for the local MultiAgentPlatform SQL Server DB."""

import os
import unittest
from datetime import datetime, timezone
import uuid

from sqlalchemy import delete
from sqlalchemy.engine import make_url

from github_agent.activity import ActivityRecord
from infrastructure.database.mssql.activity_store import DuplicateActivityError, MSSQLActivityStore
from infrastructure.database.mssql.connection import (
    DatabaseSettings,
    create_database_engine,
    create_session_factory,
    initialize_activity_schema,
)
from infrastructure.database.mssql.models import GitHubActivityModel


ENABLED = os.environ.get("RUN_MSSQL_INTEGRATION_TESTS") == "1"


@unittest.skipUnless(ENABLED, "Set RUN_MSSQL_INTEGRATION_TESTS=1 to use local SQL Server.")
class TestMSSQLActivityStoreIntegration(unittest.TestCase):
    def test_persistence_filters_duplicate_guard_and_cleanup(self):
        settings = DatabaseSettings.from_environment()
        database_name = (
            make_url(settings.database_url).database
            if settings.database_url
            else settings.database
        )
        if database_name != "MultiAgentPlatform":
            self.skipTest("Integration tests are restricted to the MultiAgentPlatform database.")

        engine = create_database_engine(settings)
        activity_id = f"activity-integration-{uuid.uuid4()}"
        event_id = f"event-integration-{uuid.uuid4()}"
        store = MSSQLActivityStore(create_session_factory(engine))
        schema_initialized = False
        try:
            initialize_activity_schema(engine)
            schema_initialized = True
            activity = ActivityRecord(
                activity_id=activity_id,
                event_id=event_id,
                source="github",
                agent="github-agent",
                event_type="repository",
                action="created",
                repository={"full_name": "integration/test-repo", "name": "test-repo"},
                actor={"login": "integration-test"},
                target={"type": "repository", "name": "integration/test-repo"},
                occurred_at=None,
                processed_at=datetime.now(timezone.utc),
                status="processed",
                metadata={"test_record": True},
            )
            store.save(activity)
            self.assertEqual(store.get(activity_id).event_id, event_id)
            self.assertIn(
                activity_id,
                [item.activity_id for item in store.list_activities(repository="integration/test-repo")],
            )
            self.assertIn(
                activity_id,
                [item.activity_id for item in store.list_activities(event_type="repository")],
            )
            self.assertIn(
                activity_id,
                [item.activity_id for item in store.list_activities(status="processed")],
            )

            duplicate = ActivityRecord(
                **{**activity.__dict__, "activity_id": f"activity-integration-duplicate-{uuid.uuid4()}"}
            )
            with self.assertRaises(DuplicateActivityError):
                store.save(duplicate)
        finally:
            # Delete only the row uniquely identified by this test's activity ID.
            if schema_initialized:
                with engine.begin() as connection:
                    connection.execute(
                        delete(GitHubActivityModel.__table__).where(
                            GitHubActivityModel.activity_id == activity_id
                        )
                    )
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
