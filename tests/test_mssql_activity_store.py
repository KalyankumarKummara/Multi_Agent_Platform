import unittest
from datetime import datetime, timezone
import os
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

from github_agent.activity import ActivityRecord
from app.composition import create_github_runtime
from infrastructure.database.mssql.activity_store import (
    ActivityPersistenceError,
    DuplicateActivityError,
    MSSQLActivityStore,
)
from infrastructure.database.mssql.connection import (
    DatabaseSettings,
    create_session_factory,
    initialize_activity_schema,
)
from infrastructure.database.mssql.models import GitHubActivityModel


def make_record(
    activity_id="activity-1",
    event_id="delivery-1",
    *,
    repository="octo/demo",
    event_type="repository",
    status="processed",
    significance="medium",
    occurred_at="2026-10-07T10:00:00+00:00",
):
    return ActivityRecord(
        activity_id=activity_id,
        event_id=event_id,
        source="github",
        agent="github-agent",
        event_type=event_type,
        action="created",
        repository={"name": repository.split("/")[-1], "full_name": repository, "owner": {"login": "octo"}},
        actor={"login": "octocat"},
        target={"type": "repository", "nested": {"key": "value"}},
        occurred_at=occurred_at,
        processed_at=datetime(2026, 10, 7, 10, 1, tzinfo=timezone.utc),
        status=status,
        metadata={"labels": ["one", "two"], "nested": {"enabled": True}},
        significance=significance,
    )


class CommitFailureSession(Session):
    """Simulate a database failure after flushing but before commit."""

    def commit(self):
        self.flush()
        raise OperationalError("commit", {}, Exception("secret connection information"))


class TestMSSQLActivityStore(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        initialize_activity_schema(self.engine)
        self.sessions = create_session_factory(self.engine)
        self.store = MSSQLActivityStore(self.sessions)

    def tearDown(self):
        self.engine.dispose()

    def test_activity_record_round_trip_and_nested_json(self):
        significance_column = GitHubActivityModel.__table__.c.significance
        self.assertIsNotNone(significance_column.server_default)
        self.assertEqual(str(significance_column.server_default.arg), "'low'")

        record = make_record()
        model = self.store._to_model(record)
        restored = self.store._to_record(model)

        self.assertEqual(restored, record)
        self.assertEqual(restored.repository["owner"]["login"], "octo")
        self.assertEqual(restored.target["nested"]["key"], "value")
        self.assertEqual(restored.metadata["nested"]["enabled"], True)
        self.assertEqual(restored.significance, "medium")
        self.assertEqual(model.repository, '{"full_name":"octo/demo","name":"demo","owner":{"login":"octo"}}')

    def test_save_get_and_list(self):
        record = make_record()
        saved = self.store.save(record)

        self.assertEqual(saved, record)
        self.assertEqual(self.store.get(record.activity_id), record)
        self.assertEqual(self.store.list_activities(), [record])
        self.assertIsNone(self.store.get("unknown"))

    def test_repository_event_type_and_status_filters(self):
        records = [
            make_record("one", "event-one", repository="octo/demo", event_type="repository", status="processed", significance="high", occurred_at="2026-10-07T09:00:00+00:00"),
            make_record("two", "event-two", repository="octo/other", event_type="push", status="processed", significance="low", occurred_at="2026-10-07T10:00:00+00:00"),
            make_record("three", "event-three", repository="octo/demo", event_type="push", status="failed", significance="medium", occurred_at="2026-10-07T11:00:00+00:00"),
        ]
        for record in records:
            self.store.save(record)

        self.assertEqual(len(self.store.list_activities(repository="octo/demo")), 2)
        self.assertEqual(len(self.store.list_activities(event_type="push")), 2)
        self.assertEqual(len(self.store.list_activities(status="processed")), 2)
        self.assertEqual([item.activity_id for item in self.store.list_activities(significance="high")], ["one"])
        self.assertEqual([item.activity_id for item in self.store.list_activities(occurred_at_from=datetime(2026, 10, 7, 10, tzinfo=timezone.utc))], ["three", "two"])
        self.assertEqual([item.activity_id for item in self.store.list_activities(occurred_at_to=datetime(2026, 10, 7, 10, tzinfo=timezone.utc))], ["one", "two"])
        self.assertEqual([item.activity_id for item in self.store.list_activities(repository="octo/demo", significance="medium", limit=1)], ["three"])
        self.assertEqual([item.activity_id for item in self.store.list_activities(limit=2)], ["one", "three"])
        self.assertEqual(
            [(item.activity_id, item.event_type) for item in self.store.list_activities()],
            [("one", "repository"), ("three", "push"), ("two", "push")],
        )

    def test_duplicate_event_id_is_rejected_without_overwriting(self):
        original = make_record()
        self.store.save(original)
        duplicate = make_record("activity-2", original.event_id)

        with self.assertRaises(DuplicateActivityError):
            self.store.save(duplicate)

        activities = self.store.list_activities()
        self.assertEqual(len(activities), 1)
        self.assertEqual(activities[0].activity_id, original.activity_id)

    def test_duplicate_activity_id_is_rejected(self):
        self.store.save(make_record())
        duplicate = make_record(event_id="another-event")
        with self.assertRaises(DuplicateActivityError):
            self.store.save(duplicate)

    def test_database_error_rolls_back_and_hides_connection_details(self):
        failing_sessions = sessionmaker(
            bind=self.engine,
            class_=CommitFailureSession,
            expire_on_commit=False,
        )
        failing_store = MSSQLActivityStore(failing_sessions)

        with self.assertRaises(ActivityPersistenceError) as caught:
            failing_store.save(make_record())

        self.assertNotIn("secret connection information", str(caught.exception))
        with self.sessions() as session:
            rows = session.scalars(select(GitHubActivityModel)).all()
        self.assertEqual(rows, [])

    def test_settings_use_safe_development_defaults(self):
        settings = DatabaseSettings()
        url = settings.to_sqlalchemy_url()
        connection = url.query["odbc_connect"]

        self.assertEqual(settings.database, "MultiAgentPlatform")
        self.assertIn("DRIVER={ODBC Driver 18 for SQL Server}", connection)
        self.assertIn("SERVER={localhost}", connection)
        self.assertIn("DATABASE={MultiAgentPlatform}", connection)
        self.assertIn("Trusted_Connection=yes", connection)
        self.assertIn("TrustServerCertificate=yes", connection)

    def test_managerai_database_is_rejected(self):
        with self.assertRaises(ValueError):
            DatabaseSettings(database="ManagerAI").to_sqlalchemy_url()

    def test_database_target_must_be_multi_agent_platform(self):
        with self.assertRaises(ValueError):
            DatabaseSettings(database="OtherDatabase").to_sqlalchemy_url()

    def test_conflicting_url_and_odbc_database_names_are_rejected(self):
        conflicting_url = URL.create(
            "mssql+pyodbc",
            database="MultiAgentPlatform",
            query={"odbc_connect": "DRIVER={ODBC Driver 18 for SQL Server};DATABASE={ManagerAI}"},
        ).render_as_string(hide_password=False)
        with self.assertRaises(ValueError):
            DatabaseSettings(database_url=conflicting_url).to_sqlalchemy_url()

    def test_custom_url_must_explicitly_target_multi_agent_platform(self):
        manager_ai_url = URL.create(
            "mssql+pyodbc",
            query={"odbc_connect": "DRIVER={ODBC Driver 18 for SQL Server};DATABASE={ManagerAI}"},
        ).render_as_string(hide_password=False)
        with self.assertRaises(ValueError):
            DatabaseSettings(database_url=manager_ai_url).to_sqlalchemy_url()

    def test_environment_configuration_and_runtime_store_injection(self):
        with patch.dict(
            os.environ,
            {
                "GITHUB_ACTIVITY_DB_SERVER": "sql-dev",
                "GITHUB_ACTIVITY_DB_NAME": "MultiAgentPlatform",
                "GITHUB_ACTIVITY_DB_TRUST_SERVER_CERTIFICATE": "no",
            },
            clear=True,
        ):
            settings = DatabaseSettings.from_environment()
        self.assertEqual(settings.server, "sql-dev")
        self.assertFalse(settings.trust_server_certificate)

        injected_store = MSSQLActivityStore(self.sessions)
        runtime = create_github_runtime(activity_store=injected_store)
        self.assertIs(runtime.activity_store, injected_store)
        self.assertIs(runtime.agent.activity_store, injected_store)


if __name__ == "__main__":
    unittest.main()
