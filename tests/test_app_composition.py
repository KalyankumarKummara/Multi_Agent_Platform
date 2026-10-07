"""Tests for application runtime storage wiring."""

import unittest
from unittest.mock import Mock, patch

from app.composition import create_github_runtime
from github_agent import InMemoryActivityStore


class TestGitHubRuntimeComposition(unittest.TestCase):
    def test_explicit_in_memory_store_is_used_unchanged(self) -> None:
        activity_store = InMemoryActivityStore()

        with (
            patch("app.composition.DatabaseSettings.from_environment") as settings,
            patch("app.composition.create_database_engine") as create_engine,
            patch("app.composition.initialize_activity_schema") as initialize_schema,
            patch("app.composition.create_session_factory") as create_sessions,
            patch("app.composition.MSSQLActivityStore") as create_mssql_store,
        ):
            runtime = create_github_runtime(activity_store=activity_store)

        self.assertIs(runtime.activity_store, activity_store)
        settings.assert_not_called()
        create_engine.assert_not_called()
        initialize_schema.assert_not_called()
        create_sessions.assert_not_called()
        create_mssql_store.assert_not_called()

    def test_default_runtime_initializes_and_uses_mssql_store(self) -> None:
        settings = Mock(name="database_settings")
        engine = Mock(name="database_engine")
        session_factory = Mock(name="session_factory")
        activity_store = Mock(name="mssql_activity_store")

        with (
            patch(
                "app.composition.DatabaseSettings.from_environment",
                return_value=settings,
            ) as load_settings,
            patch("app.composition.create_database_engine", return_value=engine) as create_engine,
            patch("app.composition.initialize_activity_schema") as initialize_schema,
            patch(
                "app.composition.create_session_factory",
                return_value=session_factory,
            ) as create_sessions,
            patch(
                "app.composition.MSSQLActivityStore",
                return_value=activity_store,
            ) as create_mssql_store,
        ):
            runtime = create_github_runtime()

        self.assertIs(runtime.activity_store, activity_store)
        load_settings.assert_called_once_with()
        create_engine.assert_called_once_with(settings)
        initialize_schema.assert_called_once_with(engine)
        create_sessions.assert_called_once_with(engine)
        create_mssql_store.assert_called_once_with(session_factory)


if __name__ == "__main__":
    unittest.main()
