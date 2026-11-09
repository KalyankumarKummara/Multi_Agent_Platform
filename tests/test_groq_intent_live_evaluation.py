"""Offline tests for the explicitly opt-in, classification-only harness."""

from __future__ import annotations

import importlib
import io
import json
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import groq_intent_live_evaluation as evaluation
from app.composition import create_github_runtime
from github_agent import InMemoryActivityStore
from shared.agent_core.errors import AgentError


def response(intent: str, tool_id=None, arguments=None, clarification=None) -> str:
    return json.dumps({
        "intent": intent,
        "tool_id": tool_id,
        "arguments": json.dumps(arguments or {}),
        "clarification": clarification,
    })


class FakeProvider:
    def __init__(self, responses=None, failures=None):
        self.responses = list(responses or [])
        self.failures = list(failures or [])
        self.calls = []

    def ask(self, prompt, *, response_format):
        self.calls.append((prompt, response_format))
        if self.failures:
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
        return self.responses.pop(0)


def provider_error(status=429, code="rate_limit_exceeded", retry_after="4"):
    error = RuntimeError("private provider detail must not be reported")
    error.status_code = status
    error.body = {"error": {"code": code, "message": "private body"}}
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    headers["authorization"] = "Bearer private-header-secret"
    error.response = SimpleNamespace(headers=headers)
    return error


class TestLiveEvaluationOptIn(unittest.TestCase):
    def test_default_mode_is_offline_and_does_not_call_a_provider(self):
        provider = Mock()
        result = evaluation.run_evaluation(live=False, provider=provider)
        provider.ask.assert_not_called()
        self.assertEqual(result["mode"], "offline")
        self.assertEqual(result["status"], "not_run")
        self.assertEqual(result["model_submitted"], 0)
        self.assertEqual(result["total_cases"], 20)

    def test_cli_live_mode_requires_explicit_flag(self):
        self.assertFalse(evaluation.build_argument_parser().parse_args([]).live)
        self.assertTrue(evaluation.build_argument_parser().parse_args(["--live"]).live)
        with self.assertRaisesRegex(ValueError, "explicit service and provider"):
            evaluation.run_evaluation(live=True)

    def test_cli_parses_case_and_provider_controls_without_enabling_live_mode(self):
        args = evaluation.build_argument_parser().parse_args([
            "--case-id", "get_commit", "--case-id", "missing_pr_number",
            "--request-spacing-seconds", "1.25", "--max-provider-retries", "2",
            "--retry-base-wait-seconds", "0.5", "--max-retry-wait-seconds", "8",
        ])
        self.assertFalse(args.live)
        self.assertEqual(args.case_ids, ["get_commit", "missing_pr_number"])
        self.assertEqual(args.request_spacing_seconds, 1.25)
        self.assertEqual(args.max_provider_retries, 2)

    def test_case_selection_controls_require_live_opt_in(self):
        with patch.object(evaluation, "_create_live_dependencies") as create_dependencies:
            for argv in (
                ["--case-id", "get_commit"],
                ["--request-spacing-seconds", "1"],
                ["--max-provider-retries", "1"],
            ):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    evaluation.main(argv)
            create_dependencies.assert_not_called()

    def test_cli_offline_mode_does_not_construct_live_dependencies(self):
        with patch.object(evaluation, "_create_live_dependencies") as create_dependencies:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(evaluation.main([]), 0)
        create_dependencies.assert_not_called()

    def test_limit_without_live_is_rejected_before_live_initialization(self):
        with patch.object(evaluation, "_create_live_dependencies") as create_dependencies:
            with self.assertRaises(SystemExit) as raised:
                evaluation.main(["--limit", "1"])
        self.assertEqual(raised.exception.code, 2)
        create_dependencies.assert_not_called()

    def test_cli_rejects_invalid_and_overlarge_limits_before_initialization(self):
        for argv in (
            ["--live", "--limit", "0"],
            ["--live", "--limit", "-2"],
            ["--live", "--limit", "not-an-int"],
            ["--live", "--limit", str(len(evaluation.LIVE_CASES) + 1)],
        ):
            with self.subTest(argv=argv):
                with patch.object(evaluation, "_create_live_dependencies") as create_dependencies:
                    with self.assertRaises(SystemExit) as raised:
                        evaluation.main(argv)
                self.assertEqual(raised.exception.code, 2)
                create_dependencies.assert_not_called()

    def test_production_path_option_requires_live_mode(self):
        with patch.object(evaluation, "_create_live_dependencies") as create_dependencies:
            with self.assertRaises(SystemExit) as raised:
                evaluation.main(["--production-path-only"])
        self.assertEqual(raised.exception.code, 2)
        create_dependencies.assert_not_called()

    def test_production_path_limit_is_validated_before_initialization(self):
        with patch.object(evaluation, "_create_live_dependencies") as create_dependencies:
            with self.assertRaises(SystemExit) as raised:
                evaluation.main([
                    "--live", "--production-path-only", "--limit",
                    str(len(evaluation.LIVE_CASES)),
                ])
        self.assertEqual(raised.exception.code, 2)
        create_dependencies.assert_not_called()

    def test_importing_harness_does_not_initialize_external_clients(self):
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network")):
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                reloaded = importlib.reload(evaluation)
        self.assertEqual(reloaded.offline_result()["model_submitted"], 0)


class TestClassificationOnlyEvaluation(unittest.TestCase):
    def setUp(self):
        self.safe_client, self.transport = evaluation._evaluation_github_client()
        runtime = create_github_runtime(
            activity_store=InMemoryActivityStore(), github_client=self.safe_client
        )
        self.service = runtime.github_request_service
        self.runtime = runtime
        self.service.default_owner = evaluation.SYNTHETIC_DEFAULT_OWNER
        self.service.default_repo = evaluation.SYNTHETIC_DEFAULT_REPO
        self.executor = Mock(side_effect=AssertionError("executor must never be called"))
        self.service.tool_executor.execute = self.executor

    def test_evaluation_github_transport_rejects_simulated_request(self):
        tool = self.runtime.agent.tool_registry.get("list_branches")
        with self.assertRaises(AgentError):
            tool.execute(owner="acme", repo="atlas")
        self.assertEqual(self.transport.attempts, 1)
        self.assertEqual(self.transport.rejected, 1)
        interceptor = evaluation.EvaluationExecutorInterceptor()
        report = evaluation.build_report(
            [], [], execution_interceptor=interceptor, github_transport=self.transport
        )
        self.assertEqual(report["executor_interceptor_attempts"], 0)
        self.assertEqual(report["executor_interceptor_rejections"], 0)
        self.assertEqual(report["github_transport_request_attempts"], 1)
        self.assertEqual(report["github_transport_request_rejections"], 1)
        self.assertIsNone(report["production_executor_invocations"])
        self.executor.assert_not_called()

    def test_executor_interceptor_records_and_rejects_without_delegating(self):
        interceptor = evaluation.EvaluationExecutorInterceptor()
        tool = self.runtime.agent.tool_registry.get("list_branches")
        with self.assertRaises(evaluation.EvaluationNetworkBlocked):
            interceptor.execute(tool, {"owner": "acme", "repo": "atlas"}, None)
        self.assertEqual(interceptor.attempt_count, 1)
        self.assertEqual(interceptor.rejected, 1)
        self.executor.assert_not_called()
        self.assertEqual(self.transport.attempts, 0)

    def test_report_execution_counters_come_from_interceptor(self):
        interceptor = evaluation.EvaluationExecutorInterceptor()
        tool = self.runtime.agent.tool_registry.get("list_branches")
        with self.assertRaises(evaluation.EvaluationNetworkBlocked):
            interceptor.execute(tool, {"owner": "acme", "repo": "atlas"}, None)
        report = evaluation.build_report(
            [], [], execution_interceptor=interceptor, github_transport=self.transport
        )
        self.assertEqual(report["executor_interceptor_attempts"], 1)
        self.assertEqual(report["executor_interceptor_rejections"], 1)
        self.assertEqual(report["github_transport_request_attempts"], 0)
        self.assertEqual(report["github_transport_request_rejections"], 0)
        self.assertIsNone(report["production_executor_invocations"])
        self.executor.assert_not_called()

    def test_missing_or_inconsistent_transport_counters_are_rejected(self):
        interceptor = evaluation.EvaluationExecutorInterceptor()
        with self.assertRaisesRegex(ValueError, "transport is required"):
            evaluation.build_report([], [], execution_interceptor=interceptor)

        del self.transport.rejected
        with self.assertRaisesRegex(ValueError, "transport counters are missing"):
            evaluation.build_report(
                [], [], execution_interceptor=interceptor, github_transport=self.transport
            )

    def test_inconsistent_transport_attempt_and_rejection_counts_are_rejected(self):
        self.transport.attempts = 1
        with self.assertRaisesRegex(ValueError, "transport counters are missing or inconsistent"):
            evaluation.build_report(
                [], [], execution_interceptor=evaluation.EvaluationExecutorInterceptor(),
                github_transport=self.transport,
            )

    def test_non_integer_or_negative_observed_counters_are_rejected(self):
        for invalid in (-1, True, 1.5):
            with self.subTest(counter_value=type(invalid).__name__):
                self.transport.attempts = invalid
                self.transport.rejected = invalid
                with self.assertRaisesRegex(ValueError, "transport counters are missing or inconsistent"):
                    evaluation.build_report(
                        [], [], execution_interceptor=evaluation.EvaluationExecutorInterceptor(),
                        github_transport=self.transport,
                    )

        self.transport.attempts = 0
        self.transport.rejected = 0
        interceptor = evaluation.EvaluationExecutorInterceptor()
        interceptor.rejected = True
        with self.assertRaisesRegex(ValueError, "interceptor counters are missing or inconsistent"):
            evaluation.build_report(
                [], [], execution_interceptor=interceptor, github_transport=self.transport
            )

    def test_run_evaluation_rejects_unsafe_service_before_provider_call(self):
        unsafe_runtime = create_github_runtime(activity_store=InMemoryActivityStore())
        provider = Mock()
        with self.assertRaisesRegex(ValueError, "fail-closed evaluation client"):
            evaluation.run_evaluation(
                live=True, service=unsafe_runtime.github_request_service, provider=provider
            )
        provider.ask.assert_not_called()

    def test_run_evaluation_rejects_unverified_tool_binding_before_provider_call(self):
        original = self.service.tool_registry.get("list_branches")
        self.service.tool_registry._tools["list_branches"] = replace(
            original, execute=lambda **_kwargs: None
        )
        provider = Mock()
        with self.assertRaisesRegex(ValueError, "fail-closed evaluation client"):
            evaluation.run_evaluation(live=True, service=self.service, provider=provider)
        provider.ask.assert_not_called()

    def test_correctly_constructed_evaluation_runtime_passes_boundary_check(self):
        self.assertIs(
            evaluation._verify_evaluation_github_bindings(self.service), self.transport
        )

    def test_every_live_case_is_classified_without_executor_invocation(self):
        outputs = []
        for case in evaluation.LIVE_CASES:
            if case.expected_intent == "github_read":
                expected = evaluation._expected_for(self.service, case)
                outputs.append(response(
                    case.expected_intent, case.expected_tool_id, expected["arguments"]
                ))
            elif case.expected_intent == "clarification_required":
                outputs.append(response(
                    "clarification_required", clarification="Please clarify the request."
                ))
            else:
                outputs.append(response("unsupported"))
        provider = FakeProvider(outputs)

        report = evaluation.run_evaluation(
            live=True, service=self.service, provider=provider
        )

        self.assertEqual(report["total_cases"], 20)
        self.assertEqual(report["live_eligible_cases"], 15)
        self.assertEqual(report["model_submitted"], 15)
        self.assertEqual(len(provider.calls), 15)
        self.executor.assert_not_called()
        self.assertEqual(report["executor_interceptor_attempts"], 0)
        self.assertEqual(report["executor_interceptor_rejections"], 0)
        self.assertIsNone(report["production_executor_invocations"])
        self.assertEqual(report["github_transport_request_attempts"], 0)
        self.assertEqual(report["github_transport_request_rejections"], 0)
        self.assertEqual(report["intent_accuracy"], {
            "numerator": 11, "denominator": 11, "rate": 1.0,
        })
        self.assertEqual(report["production_path_metrics"]["sample_size"], 11)
        self.assertEqual(report["standalone_probe_metrics"]["sample_size"], 4)
        self.assertEqual(report["standalone_probe_metrics"]["intent_accuracy"], {
            "numerator": 4, "denominator": 4, "rate": 1.0,
        })

    def test_limit_one_submits_exactly_the_first_case_and_reports_counts(self):
        provider = FakeProvider([response("unsupported")])
        report = evaluation.run_evaluation(
            live=True, service=self.service, provider=provider, limit=1
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(report["requested_case_count"], 1)
        self.assertEqual(report["actual_case_count"], 1)
        self.assertEqual(report["case_limit"], 1)
        self.assertEqual(report["selected_case_ids"], [evaluation.LIVE_CASES[0].case_id])

    def test_no_limit_preserves_all_eligible_cases_and_reports_counts(self):
        provider = FakeProvider([response("unsupported") for _ in evaluation.LIVE_CASES])
        report = evaluation.run_evaluation(
            live=True, service=self.service, provider=provider
        )
        self.assertEqual(len(provider.calls), len(evaluation.LIVE_CASES))
        self.assertEqual(report["requested_case_count"], len(evaluation.LIVE_CASES))
        self.assertEqual(report["actual_case_count"], len(evaluation.LIVE_CASES))
        self.assertIsNone(report["case_limit"])

    def test_run_evaluation_rejects_invalid_limits_before_provider_call(self):
        for limit in (0, -1, len(evaluation.LIVE_CASES) + 1, True, 1.5):
            with self.subTest(limit=limit):
                provider = Mock()
                with self.assertRaises(ValueError):
                    evaluation.run_evaluation(
                        live=True, service=self.service, provider=provider, limit=limit
                    )
                provider.ask.assert_not_called()

    def test_production_path_selection_uses_actual_preprocessing_and_preserves_order(self):
        selected = evaluation.select_evaluation_cases(production_path_only=True)
        self.assertEqual([case.case_id for case in selected], [
            "list_repositories", "list_pull_requests", "get_pull_request", "get_issue",
            "get_commit", "missing_pr_number", "ambiguous_request", "explicit_repository",
            "configured_default_repository", "unsupported_request", "prompt_injection",
        ])
        self.assertTrue(all(evaluation._preprocessing_outcome(case.message) is None for case in selected))
        excluded = {case.case_id for case in evaluation.LIVE_CASES} - {case.case_id for case in selected}
        self.assertEqual(excluded, {
            "greeting", "conflicting_repositories", "create_issue", "merge_pull_request",
        })

    def test_production_path_filter_runs_before_limit_and_reports_eligible_count(self):
        provider = FakeProvider([response("unsupported")])
        report = evaluation.run_evaluation(
            live=True, service=self.service, provider=provider,
            production_path_only=True, limit=1,
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(report["eligible_case_count"], 11)
        self.assertEqual(report["requested_case_count"], 1)
        self.assertEqual(report["actual_case_count"], 1)
        self.assertTrue(report["production_path_only"])
        self.assertEqual(report["selected_case_ids"], ["list_repositories"])

    def test_empty_production_path_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No production-path evaluation cases"):
            evaluation.select_evaluation_cases(
                production_path_only=True, cases=evaluation.SYNTHETIC_FAULT_CASES
            )

    def test_limit_is_checked_against_filtered_production_path_pool(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed 11 selected eligible cases"):
            evaluation.select_evaluation_cases(production_path_only=True, limit=12)

    def test_case_ids_validate_filter_order_and_apply_limit_after_filter(self):
        selected = evaluation.select_evaluation_cases(
            case_ids=["prompt_injection", "get_commit", "get_commit"], limit=1,
            production_path_only=True,
        )
        self.assertEqual([case.case_id for case in selected], ["get_commit"])
        with self.assertRaisesRegex(ValueError, "Unknown evaluation case ID.*missing_case"):
            evaluation.select_evaluation_cases(case_ids=["missing_case"])
        with self.assertRaisesRegex(ValueError, "excluded by production-path filter.*greeting"):
            evaluation.select_evaluation_cases(
                production_path_only=True, case_ids=["greeting"]
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed 1 selected eligible cases"):
            evaluation.select_evaluation_cases(case_ids=["get_commit"], limit=2)

    def test_case_id_run_reports_selected_case_and_match_count(self):
        case = next(case for case in evaluation.LIVE_CASES if case.case_id == "get_commit")
        provider = FakeProvider([response(
            "github_read", case.expected_tool_id, case.expected_arguments,
        )])
        provider.model = "openai/gpt-oss-20b"
        report = evaluation.run_evaluation(
            live=True, service=self.service, provider=provider,
            production_path_only=True, case_ids=["get_commit"], limit=1,
        )
        self.assertEqual(report["selected_case_ids"], ["get_commit"])
        self.assertEqual(report["eligible_case_count"], 11)
        self.assertEqual(report["case_filter_match_count"], 1)
        self.assertEqual(report["requested_case_count"], 1)
        self.assertEqual(report["resolved_model_name"], "openai/gpt-oss-20b")

    def test_prompt_injection_proposing_unregistered_tool_is_rejected_without_execution(self):
        case = next(case for case in evaluation.LIVE_CASES if case.case_id == "prompt_injection")
        provider = FakeProvider([response(
            "github_read", "call_github_api", {"url": "https://invalid.example/"},
        )])

        report = evaluation.run_evaluation(
            live=True, service=self.service, provider=provider,
            cases=[case], case_ids=[case.case_id],
        )

        result = report["results"][0]
        self.assertEqual(result["classifier_output"]["parse_status"], "valid")
        self.assertEqual(result["pre_execution_outcome"], "invalid_tool")
        self.assertEqual(result["validation_error_code"], "UNAPPROVED_TOOL")
        self.assertEqual(report["executor_interceptor_attempts"], 0)
        self.assertEqual(report["github_transport_request_attempts"], 0)
        self.assertEqual(self.transport.attempts, 0)
        self.executor.assert_not_called()

    def test_explicit_repository_is_effective_even_if_model_proposes_other_target(self):
        case = next(case for case in evaluation.LIVE_CASES if case.case_id == "explicit_repository")
        provider = FakeProvider([response(
            "github_read", "list_branches", {"owner": "wrong", "repo": "target"}
        )])
        result = evaluation.evaluate_case(self.service, provider, case)
        self.assertEqual(result["classifier_output"]["arguments"], {
            "owner": "wrong", "repo": "target",
        })
        self.assertEqual(result["effective_arguments"], {"owner": "acme", "repo": "atlas"})
        self.assertEqual(result["pre_execution_outcome"], "ready_for_pre_execution_boundary")
        self.assertTrue(result["ready_for_pre_execution_boundary"])
        self.executor.assert_not_called()

    def test_malformed_output_is_recorded_without_crashing(self):
        case = next(case for case in evaluation.SYNTHETIC_FAULT_CASES
                    if case.synthetic_fault == "malformed_output")
        result = evaluation.evaluate_case(self.service, FakeProvider(["not JSON"]), case)
        self.assertEqual(result["classifier_output"]["parse_status"], "invalid")
        self.assertEqual(result["pre_execution_outcome"], "invalid_output")

    def test_unexpected_tool_id_is_rejected_before_execution(self):
        case = next(case for case in evaluation.SYNTHETIC_FAULT_CASES
                    if case.synthetic_fault == "unexpected_tool_id")
        provider = FakeProvider([response("github_read", "call_github_api", {"url": "https://invalid"})])
        result = evaluation.evaluate_case(self.service, provider, case)
        self.assertEqual(result["classifier_output"]["parse_status"], "valid")
        self.assertEqual(result["pre_execution_outcome"], "invalid_tool")
        self.assertEqual(result["validation_error_code"], "UNAPPROVED_TOOL")
        self.executor.assert_not_called()

    def test_missing_required_arguments_become_clarification(self):
        case = next(case for case in evaluation.SYNTHETIC_FAULT_CASES
                    if case.synthetic_fault == "missing_required_arguments")
        provider = FakeProvider([response(
            "github_read", "get_issue", {"owner": "acme", "repo": "atlas"}
        )])
        result = evaluation.evaluate_case(self.service, provider, case)
        self.assertEqual(result["pre_execution_outcome"], "clarification_required")
        self.assertFalse(result["ready_for_pre_execution_boundary"])

    def test_extra_and_wrong_type_arguments_are_rejected(self):
        case = next(case for case in evaluation.SYNTHETIC_FAULT_CASES
                    if case.synthetic_fault == "extra_wrong_type_arguments")
        provider = FakeProvider([response(
            "github_read", "list_branches",
            {"owner": "acme", "repo": "atlas", "page": "first", "extra": True},
        )])
        result = evaluation.evaluate_case(self.service, provider, case)
        self.assertEqual(result["pre_execution_outcome"], "invalid_arguments")
        self.assertEqual(result["validation_error_code"], "INVALID_TOOL_ARGUMENTS")
        self.executor.assert_not_called()

    def test_provider_timeout_is_safe_and_later_cases_can_continue(self):
        first, second = evaluation.LIVE_CASES[1:3]
        provider = FakeProvider(
            [response("github_read", second.expected_tool_id, second.expected_arguments)],
            failures=[TimeoutError("synthetic private provider detail"), None],
        )
        results = evaluation.run_evaluation(
            live=True, service=self.service, provider=provider, cases=[first, second]
        )
        self.assertEqual(results["provider_failures_and_timeouts"], {
            "numerator": 1, "denominator": 2, "rate": 0.5,
        })
        self.assertEqual(results["results"][0]["classifier_output"]["parse_status"],
                         "provider_failure")
        self.assertEqual(
            results["results"][0]["classifier_output"]["provider_error"]["category"],
            "provider_timeout",
        )
        self.assertNotIn("synthetic private provider detail", repr(results))

    def test_prompt_builder_failure_is_harness_error_not_submission(self):
        case = evaluation.LIVE_CASES[1]
        self.service._build_prompt = Mock(side_effect=ValueError("synthetic prompt error"))
        provider = FakeProvider([response("github_read", case.expected_tool_id, case.expected_arguments)])
        result = evaluation.evaluate_case(self.service, provider, case)
        self.assertFalse(result["model_submitted"])
        self.assertEqual(result["classifier_output"]["parse_status"], "harness_error")
        self.assertEqual(provider.calls, [])

    def test_provider_exception_is_not_a_harness_error(self):
        case = evaluation.LIVE_CASES[1]
        provider = FakeProvider(failures=[RuntimeError("synthetic provider error")])
        result = evaluation.evaluate_case(self.service, provider, case)
        self.assertTrue(result["model_submitted"])
        self.assertEqual(result["classifier_output"]["parse_status"], "provider_failure")
        self.assertEqual(result["classifier_output"]["provider_error"]["category"], "provider_error")

    def test_api_connection_diagnostics_include_only_allowlisted_cause_types(self):
        from groq import APIConnectionError
        from httpx import Request

        try:
            try:
                raise ConnectionResetError("private socket detail")
            except ConnectionResetError as root_cause:
                raise OSError("private wrapped detail") from root_cause
        except OSError as cause:
            error = APIConnectionError(
                request=Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            )
            error.__cause__ = cause

        diagnostics = evaluation._safe_provider_error(self.service, error)
        self.assertEqual(diagnostics["category"], "connection_failure")
        self.assertEqual(diagnostics["transport_cause_types"], ["OSError", "ConnectionResetError"])
        self.assertNotIn("private socket detail", repr(diagnostics))
        self.assertNotIn("private wrapped detail", repr(diagnostics))
        self.assertNotIn("api.groq.com", repr(diagnostics))

    def test_request_spacing_is_sequential_and_reported_separately(self):
        cases = evaluation.LIVE_CASES[1:3]
        provider = FakeProvider([
            response("github_read", case.expected_tool_id, case.expected_arguments)
            for case in cases
        ])
        sleeper = Mock()
        report = evaluation.run_evaluation(
            live=True, service=self.service, provider=provider, cases=cases,
            request_spacing_seconds=2.5, sleeper=sleeper,
        )
        self.assertEqual(len(provider.calls), 2)
        sleeper.assert_called_once_with(2.5)
        self.assertEqual(report["provider_request_attempts"], 2)
        self.assertEqual(report["provider_retry_attempts"], 0)
        self.assertEqual(report["request_spacing_waits"], [
            {"case_id": cases[1].case_id, "wait_seconds": 2.5},
        ])

    def test_retry_after_is_respected_and_retry_attempts_are_reported(self):
        case = evaluation.LIVE_CASES[1]
        provider = FakeProvider(
            [response("github_read", case.expected_tool_id, case.expected_arguments)],
            failures=[provider_error(retry_after="4.5")],
        )
        sleeper = Mock()
        report = evaluation.run_evaluation(
            live=True, service=self.service, provider=provider, cases=[case],
            max_provider_retries=1, max_retry_wait_seconds=5, sleeper=sleeper,
        )
        self.assertEqual(len(provider.calls), 2)
        sleeper.assert_called_once_with(4.5)
        self.assertEqual(report["provider_request_attempts"], 2)
        self.assertEqual(report["provider_retry_attempts"], 1)
        self.assertEqual(report["provider_retry_waits"], [
            {"case_id": case.case_id, "wait_seconds": 4.5},
        ])
        self.assertEqual(report["provider_failures_and_timeouts"]["numerator"], 0)
        self.assertEqual(report["results"][0]["provider_retry_events"][0]["provider_error"]["http_status"], 429)
        self.assertNotIn("private-header-secret", repr(report))
        self.assertNotIn("private provider detail", repr(report))

    def test_retry_after_above_maximum_skips_retry(self):
        case = evaluation.LIVE_CASES[1]
        provider = FakeProvider(failures=[provider_error(retry_after="6")])
        sleeper = Mock()
        result = evaluation.evaluate_case(
            self.service, provider, case, max_provider_retries=2,
            max_retry_wait_seconds=5, sleeper=sleeper,
        )
        self.assertEqual(len(provider.calls), 1)
        sleeper.assert_not_called()
        self.assertEqual(result["provider_request_attempts"], 1)
        self.assertEqual(result["provider_retry_waits_seconds"], [])
        self.assertEqual(result["provider_retry_events"][0]["skip_reason"], "retry_after_exceeds_max_wait")

        malformed = FakeProvider(failures=[provider_error(retry_after="tomorrow")])
        result = evaluation.evaluate_case(
            self.service, malformed, case, max_provider_retries=2,
            max_retry_wait_seconds=30, sleeper=sleeper,
        )
        self.assertEqual(len(malformed.calls), 1)
        self.assertEqual(result["provider_retry_events"][0]["skip_reason"], "retry_after_unparseable")

    def test_retry_uses_configured_bounded_backoff_when_retry_after_is_absent(self):
        case = evaluation.LIVE_CASES[1]
        provider = FakeProvider(
            [response("github_read", case.expected_tool_id, case.expected_arguments)],
            failures=[provider_error(status=503, code="service_unavailable", retry_after=None)],
        )
        sleeper = Mock()
        result = evaluation.evaluate_case(
            self.service, provider, case, max_provider_retries=1,
            retry_base_wait_seconds=2, max_retry_wait_seconds=3, sleeper=sleeper,
        )
        self.assertEqual(len(provider.calls), 2)
        sleeper.assert_called_once_with(2)
        self.assertEqual(result["provider_retry_waits_seconds"], [2])

    def test_retry_limit_is_bounded_and_permanent_errors_are_not_retried(self):
        case = evaluation.LIVE_CASES[1]
        rate_limited = FakeProvider(failures=[
            provider_error(retry_after="1"), provider_error(retry_after="1"),
        ])
        sleeper = Mock()
        result = evaluation.evaluate_case(
            self.service, rate_limited, case, max_provider_retries=1,
            max_retry_wait_seconds=2, sleeper=sleeper,
        )
        self.assertEqual(len(rate_limited.calls), 2)
        self.assertEqual(result["provider_request_attempts"], 2)
        self.assertEqual(result["provider_retry_waits_seconds"], [1.0])
        self.assertEqual(result["provider_retry_events"][-1]["skip_reason"], "retry_limit_reached")

        permanent = FakeProvider(failures=[provider_error(
            status=401, code="invalid_api_key", retry_after="1",
        )])
        result = evaluation.evaluate_case(
            self.service, permanent, case, max_provider_retries=3,
            max_retry_wait_seconds=2, sleeper=sleeper,
        )
        self.assertEqual(len(permanent.calls), 1)
        self.assertEqual(result["provider_retry_events"][0]["skip_reason"], "permanent_provider_error")

    def test_retry_after_http_date_is_parsed_without_exposing_header(self):
        from datetime import datetime, timedelta, timezone
        from email.utils import format_datetime

        error = provider_error(retry_after=None)
        error.response.headers["retry-after"] = format_datetime(
            datetime.now(timezone.utc) + timedelta(seconds=8), usegmt=True
        )
        wait, reason = evaluation._provider_retry_plan(
            error, retry_index=0, base_wait_seconds=1, max_wait_seconds=10,
        )
        self.assertIsNone(reason)
        self.assertGreater(wait, 0)
        self.assertLessEqual(wait, 8)

    def test_retry_controls_reject_unbounded_or_nonfinite_values_before_provider(self):
        provider = Mock()
        for options in (
            {"max_provider_retries": 6},
            {"request_spacing_seconds": float("nan")},
            {"max_retry_wait_seconds": -1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                evaluation.run_evaluation(
                    live=True, service=self.service, provider=provider, **options,
                )
        provider.ask.assert_not_called()

    def test_group_metrics_separate_production_cases_from_standalone_probes(self):
        production_case = evaluation.LIVE_CASES[1]
        standalone_case = evaluation.LIVE_CASES[0]
        production_failure = evaluation.evaluate_case(
            self.service, FakeProvider(failures=[TimeoutError("synthetic timeout")]), production_case
        )
        standalone_invalid = evaluation.evaluate_case(
            self.service, FakeProvider(["not JSON"]), standalone_case
        )
        report = evaluation.build_report(
            [production_failure, standalone_invalid],
            [production_case, standalone_case],
            execution_interceptor=evaluation.EvaluationExecutorInterceptor(),
            github_transport=self.transport,
        )
        self.assertEqual(report["production_path_metrics"]["provider_failures"], {
            "numerator": 1, "denominator": 1, "rate": 1.0,
        })
        self.assertEqual(report["standalone_probe_metrics"]["parser_invalid_outputs"], {
            "numerator": 1, "denominator": 1, "rate": 1.0,
        })
        self.assertEqual(report["production_preprocessed_cases_excluded"], 1)

    def test_group_intent_accuracy_is_separate_from_argument_and_full_accuracy(self):
        case = next(case for case in evaluation.LIVE_CASES if case.case_id == "explicit_repository")
        result = evaluation.evaluate_case(
            self.service,
            FakeProvider([response("github_read", "list_branches", {
                "owner": "acme", "repo": "atlas", "page": "1",
            })]),
            case,
        )
        report = evaluation.build_report(
            [result], [case], execution_interceptor=evaluation.EvaluationExecutorInterceptor(),
            github_transport=self.transport,
        )
        metrics = report["production_path_metrics"]
        self.assertEqual(metrics["intent_accuracy"]["numerator"], 1)
        self.assertEqual(metrics["intent_accuracy"]["denominator"], 1)
        self.assertEqual(metrics["tool_selection_accuracy"]["numerator"], 1)
        self.assertEqual(metrics["exact_argument_set_accuracy"]["numerator"], 0)
        self.assertEqual(metrics["full_classification_accuracy"]["numerator"], 0)

    def test_group_metrics_give_intent_credit_but_not_tool_credit_for_wrong_tool(self):
        case = next(case for case in evaluation.LIVE_CASES if case.case_id == "explicit_repository")
        result = evaluation.evaluate_case(
            self.service,
            FakeProvider([response("github_read", "get_repository", {
                "owner": "acme", "repo": "atlas",
            })]),
            case,
        )
        report = evaluation.build_report(
            [result], [case], execution_interceptor=evaluation.EvaluationExecutorInterceptor(),
            github_transport=self.transport,
        )
        metrics = report["production_path_metrics"]
        self.assertEqual(metrics["intent_accuracy"]["numerator"], 1)
        self.assertEqual(metrics["tool_selection_accuracy"]["numerator"], 0)
        self.assertEqual(metrics["exact_argument_set_accuracy"]["numerator"], 0)
        self.assertEqual(metrics["full_classification_accuracy"]["numerator"], 0)

    def test_nullable_optional_placeholders_match_production_omitted_arguments(self):
        case = next(case for case in evaluation.LIVE_CASES if case.case_id == "list_pull_requests")
        raw_arguments = {
            "owner": "acme", "repo": "atlas", "state": "open",
            "per_page": None, "page": None, "base": None,
            "head": None, "sort": None, "direction": None,
        }
        result = evaluation.evaluate_case(
            self.service,
            FakeProvider([response("github_read", "list_pull_requests", raw_arguments)]),
            case,
        )

        # Keep the exact provider output visible, but score the same omission
        # semantics that GitHubRequestService applies before validation.
        self.assertEqual(result["classifier_output"]["arguments"], raw_arguments)
        self.assertEqual(result["arguments_for_exact_comparison"], {
            "owner": "acme", "repo": "atlas", "state": "open",
        })
        self.assertEqual(
            evaluation._arguments_for_exact_comparison(
                self.service, "get_pull_request", {"pull_number": None}
            ),
            {"pull_number": None},
        )
        self.assertEqual(
            evaluation._arguments_for_exact_comparison(
                self.service, "list_pull_requests", {"unexpected": None}
            ),
            {"unexpected": None},
        )

        report = evaluation.build_report(
            [result], [case],
            execution_interceptor=evaluation.EvaluationExecutorInterceptor(),
            github_transport=self.transport,
        )
        self.assertEqual(report["production_path_metrics"]["exact_argument_set_accuracy"], {
            "numerator": 1, "denominator": 1, "rate": 1.0,
        })
        self.assertEqual(report["production_path_metrics"]["full_classification_accuracy"], {
            "numerator": 1, "denominator": 1, "rate": 1.0,
        })

    def test_group_metrics_count_wrong_intent_invalid_output_and_provider_failure(self):
        case = next(case for case in evaluation.LIVE_CASES if case.case_id == "explicit_repository")
        results = [
            evaluation.evaluate_case(self.service, FakeProvider([response("unsupported")]), case),
            evaluation.evaluate_case(self.service, FakeProvider(["not JSON"]), case),
            evaluation.evaluate_case(self.service, FakeProvider(failures=[TimeoutError("synthetic")]), case),
        ]
        report = evaluation.build_report(
            results, [case], execution_interceptor=evaluation.EvaluationExecutorInterceptor(),
            github_transport=self.transport,
        )
        metrics = report["production_path_metrics"]
        self.assertEqual(metrics["intent_accuracy"], {
            "numerator": 0, "denominator": 3, "rate": 0.0,
        })
        self.assertEqual(metrics["parser_invalid_outputs"]["numerator"], 1)
        self.assertEqual(metrics["provider_failures"]["numerator"], 1)

    def test_build_report_requires_an_observed_interceptor(self):
        with self.assertRaisesRegex(ValueError, "interceptor is required"):
            evaluation.build_report([], [])

    def test_build_report_rejects_inconsistent_interceptor_counters(self):
        interceptor = evaluation.EvaluationExecutorInterceptor()
        tool = self.runtime.agent.tool_registry.get("list_branches")
        with self.assertRaises(evaluation.EvaluationNetworkBlocked):
            interceptor.execute(tool, {"owner": "a", "repo": "b"}, None)
        interceptor.rejected = 0
        with self.assertRaisesRegex(ValueError, "counters are missing or inconsistent"):
            evaluation.build_report(
                [], [], execution_interceptor=interceptor, github_transport=self.transport
            )

    def test_build_report_rejects_missing_interceptor_counters(self):
        interceptor = evaluation.EvaluationExecutorInterceptor()
        del interceptor.attempts
        with self.assertRaisesRegex(ValueError, "interceptor counters are missing"):
            evaluation.build_report(
                [], [], execution_interceptor=interceptor, github_transport=self.transport
            )

    def test_secret_like_values_are_redacted_from_reports(self):
        report = evaluation.sanitize_report({
            "arguments": {"api_key": "fake-private-value", "owner": "acme"},
            "text": "Authorization: Bearer gsk_abcdefgh12345678",
        })
        encoded = json.dumps(report)
        self.assertNotIn("fake-private-value", encoded)
        self.assertNotIn("gsk_abcdefgh12345678", encoded)
        self.assertEqual(report["arguments"]["api_key"], "[redacted]")

    def test_new_result_is_unique_and_existing_result_files_are_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            existing = {
                "groq_intent_evaluation_expanded_results.json": "existing expanded result",
                "groq_intent_single_case_result.json": "existing single result",
            }
            for name, content in existing.items():
                (output_dir / name).write_text(content, encoding="utf-8")
            path = evaluation.write_live_report({"mode": "test"}, output_dir)
            self.assertTrue(path.name.startswith("groq_intent_live_results_"))
            self.assertNotIn(path.name, existing)
            for name, content in existing.items():
                self.assertEqual((output_dir / name).read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
