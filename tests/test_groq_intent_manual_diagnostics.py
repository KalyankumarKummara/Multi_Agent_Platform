"""Tests for safe diagnostics in the one-case manual Groq runner."""

import unittest

from httpx import Request
from groq import APIConnectionError
from groq_intent_diagnostics import underlying_exception_type


class TestUnderlyingExceptionType(unittest.TestCase):
    def test_returns_only_the_chained_cause_class_name(self):
        request = Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        cause = ConnectionError("secret URL and credential must not be recorded")
        wrapped = APIConnectionError(request=request)
        wrapped.__cause__ = cause

        self.assertEqual(underlying_exception_type(wrapped), "ConnectionError")

    def test_returns_none_when_no_chained_cause_exists(self):
        request = Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        wrapped = APIConnectionError(request=request)

        self.assertIsNone(underlying_exception_type(wrapped))


if __name__ == "__main__":
    unittest.main()
