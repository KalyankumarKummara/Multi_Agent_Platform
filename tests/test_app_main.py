"""Tests for the production ASGI application entry point."""

import importlib
import unittest
from unittest.mock import Mock, patch


class TestProductionApp(unittest.TestCase):
    def test_app_exposes_github_webhook_post(self) -> None:
        # Keep this route-registration test independent of SQL Server setup.
        with patch(
            "app.composition.create_github_runtime",
            return_value=Mock(),
        ) as create_runtime:
            production_app = importlib.import_module("app.main").app
            create_runtime.assert_not_called()

        operations = production_app.openapi()["paths"]["/webhooks/github"]

        self.assertIn("post", operations)


if __name__ == "__main__":
    unittest.main()
