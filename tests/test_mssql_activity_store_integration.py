"""Opt-in integration test for the local MultiAgentPlatform SQL Server DB."""

import os
import unittest
from datetime import datetime, timezone
import uuid

from sqlalchemy import delete, inspect, text

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
        engine = create_database_engine(settings)
        activity_id = f"activity-integration-{uuid.uuid4()}"
        legacy_activity_id = f"activity-integration-legacy-{uuid.uuid4()}"
        test_repository = f"integration/test-repo-{uuid.uuid4().hex}"
        event_id = f"event-integration-{uuid.uuid4()}"
        legacy_event_id = f"event-integration-legacy-{uuid.uuid4()}"
        store = MSSQLActivityStore(create_session_factory(engine))
        schema_initialized = False
        test_time = datetime.now(timezone.utc)
        preexisting_ids: list[str] = []
        try:
            with engine.connect() as connection:
                actual_database = connection.scalar(text("SELECT DB_NAME()"))
                self.assertEqual(actual_database.casefold(), "multiagentplatform")
                schema_inspector = inspect(connection)
                if schema_inspector.has_table("github_activities"):
                    existing_columns = {
                        column["name"]
                        for column in schema_inspector.get_columns("github_activities")
                    }
                    if "significance" not in existing_columns:
                        preexisting_ids = list(
                            connection.execute(
                                text("SELECT [activity_id] FROM [github_activities]")
                            ).scalars()
                        )

            initialize_activity_schema(engine)
            schema_initialized = True
            initialize_activity_schema(engine)
            schema_inspector = inspect(engine)
            activity_columns = {
                column["name"]: column
                for column in schema_inspector.get_columns("github_activities")
            }
            self.assertIn("significance", activity_columns)
            self.assertFalse(activity_columns["significance"]["nullable"])
            self.assertIsNotNone(activity_columns["significance"]["default"])
            self.assertIn(
                "ix_github_activities_significance",
                {index["name"] for index in schema_inspector.get_indexes("github_activities")},
            )

            for existing_id in preexisting_ids:
                existing_record = store.get(existing_id)
                self.assertIsNotNone(existing_record)
                self.assertEqual(existing_record.significance, "low")

            activity = ActivityRecord(
                activity_id=activity_id,
                event_id=event_id,
                source="github",
                agent="github-agent",
                event_type="repository",
                action="created",
                repository={"full_name": test_repository, "name": test_repository.split("/")[-1]},
                actor={"login": "integration-test"},
                target={"type": "repository", "name": "integration/test-repo"},
                occurred_at=test_time.isoformat(),
                processed_at=test_time,
                status="processed",
                metadata={"test_record": True},
                significance="high",
            )
            store.save(activity)

            # Simulate an M5.1 writer that omits the M7 column. The database
            # server default must provide a value without changing old callers.
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO [github_activities] "
                        "([activity_id], [event_id], [source], [agent], [event_type], [action], "
                        "[repository], [repository_full_name], [actor], [target], [occurred_at], "
                        "[processed_at], [status], [metadata]) "
                        "VALUES (:activity_id, :event_id, :source, :agent, :event_type, :action, "
                        ":repository, :repository_full_name, :actor, :target, :occurred_at, "
                        ":processed_at, :status, :metadata)"
                    ),
                    {
                        "activity_id": legacy_activity_id,
                        "event_id": legacy_event_id,
                        "source": "github",
                        "agent": "github-agent",
                        "event_type": "repository",
                        "action": "created",
                        "repository": f'{{"full_name":"{test_repository}","name":"{test_repository.split("/")[-1]}"}}',
                        "repository_full_name": test_repository,
                        "actor": '{"login":"integration-test"}',
                        "target": '{"type":"repository","name":"integration/test-repo"}',
                        "occurred_at": test_time,
                        "processed_at": test_time,
                        "status": "processed",
                        "metadata": '{"test_record":true}',
                    },
                )

            self.assertEqual(store.get(activity_id).event_id, event_id)
            self.assertEqual(store.get(legacy_activity_id).significance, "low")
            self.assertIn(
                activity_id,
                [item.activity_id for item in store.list_activities(repository=test_repository)],
            )
            self.assertIn(
                activity_id,
                [item.activity_id for item in store.list_activities(event_type="repository")],
            )
            self.assertIn(
                activity_id,
                [item.activity_id for item in store.list_activities(status="processed")],
            )
            self.assertIn(
                activity_id,
                [item.activity_id for item in store.list_activities(significance="high")],
            )
            self.assertIn(
                legacy_activity_id,
                [item.activity_id for item in store.list_activities(significance="low")],
            )
            self.assertEqual(
                [item.activity_id for item in store.list_activities(
                    occurred_at_from=test_time.replace(year=2000),
                    occurred_at_to=test_time.replace(year=2100),
                    repository=test_repository,
                    limit=2,
                )],
                [activity_id, legacy_activity_id],
            )
            self.assertEqual(
                [item.activity_id for item in store.list_activities(
                    repository=test_repository,
                    event_type="repository",
                    limit=2,
                )],
                [activity_id, legacy_activity_id],
            )

            duplicate = ActivityRecord(
                **{**activity.__dict__, "activity_id": f"activity-integration-duplicate-{uuid.uuid4()}"}
            )
            with self.assertRaises(DuplicateActivityError):
                store.save(duplicate)
        finally:
            # Delete only rows uniquely identified by this test's activity IDs.
            if schema_initialized:
                with engine.begin() as connection:
                    connection.execute(
                        delete(GitHubActivityModel.__table__).where(
                            GitHubActivityModel.activity_id.in_([activity_id, legacy_activity_id])
                        )
                    )
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
