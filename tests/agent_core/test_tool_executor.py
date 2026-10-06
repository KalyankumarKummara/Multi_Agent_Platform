from shared.agent_core.approval import ApprovalService, ApprovalStatus
from shared.agent_core.audit import AuditService
from shared.agent_core.policy import PolicyEngine
from shared.agent_core.security import (
    AuthorizationService,
    Principal,
)
from shared.agent_core.tracing import create_trace_context
from shared.agent_core.tools.execution import ToolExecutionContext
from shared.agent_core.tools.executor import ToolExecutor
from shared.agent_core.tools.tool import Tool
from shared.agent_core.validation import InputValidator
from shared.agent_core.errors import AgentError


# ---------------------------------------------------------
# Test tool
# ---------------------------------------------------------

def read_repository(repository: str):
    return {
        "repository": repository,
        "status": "ok",
    }


def merge_pull_request(
    repository: str,
    pull_request_number: int,
):
    return {
        "repository": repository,
        "pull_request_number": pull_request_number,
        "status": "merged",
    }


read_repository_tool = Tool(
    tool_id="read_repository",
    name="Read Repository",
    description="Read repository information.",
    input_schema={
        "repository": {
            "type": "string",
            "required": True,
        },
    },
    output_schema={
        "repository": "string",
        "status": "string",
    },
    permissions=[
        "github.read",
    ],
    risk_level="low",
    execute=read_repository,
)


merge_pull_request_tool = Tool(
    tool_id="merge_pull_request",
    name="Merge Pull Request",
    description="Merge a pull request.",
    input_schema={
        "repository": {
            "type": "string",
            "required": True,
        },
        "pull_request_number": {
            "type": "integer",
            "required": True,
        },
    },
    output_schema={
        "repository": "string",
        "pull_request_number": "integer",
        "status": "string",
    },
    permissions=[
        "github.write",
    ],
    risk_level="high",
    execute=merge_pull_request,
)


# ---------------------------------------------------------
# Common services
# ---------------------------------------------------------

authorization_service = AuthorizationService()
policy_engine = PolicyEngine()
approval_service = ApprovalService()
audit_service = AuditService()
input_validator = InputValidator()

executor = ToolExecutor(
    authorization_service=authorization_service,
    policy_engine=policy_engine,
    approval_service=approval_service,
    audit_service=audit_service,
    input_validator=input_validator,
)


# ---------------------------------------------------------
# Principal and execution context
# ---------------------------------------------------------

principal = Principal(
    principal_id="test-agent",
    principal_type="agent",
    permissions=frozenset(
        {
            "github.read",
            "github.write",
        }
    ),
)

trace_context = create_trace_context()

context = ToolExecutionContext(
    principal=principal,
    trace_context=trace_context,
    permissions=[
        "github.read",
        "github.write",
    ],
    metadata={},
)


# ---------------------------------------------------------
# Test 1: Valid input
# ---------------------------------------------------------

result = executor.execute(
    read_repository_tool,
    {
        "repository": "payment-service",
    },
    context,
)

assert result == {
    "repository": "payment-service",
    "status": "ok",
}

print("Test 1 passed: valid input executed")


# ---------------------------------------------------------
# Test 2: Missing required field
# ---------------------------------------------------------

try:
    executor.execute(
        read_repository_tool,
        {},
        context,
    )

    raise RuntimeError(
        "Test 2 failed: missing required field was accepted"
    )

except AgentError as error:

    assert error.code == "MISSING_REQUIRED_FIELD"

    print(
        "Test 2 passed:",
        error.code,
    )


# ---------------------------------------------------------
# Test 3: Invalid field type
# ---------------------------------------------------------

try:
    executor.execute(
        merge_pull_request_tool,
        {
            "repository": "payment-service",
            "pull_request_number": "42",
        },
        context,
    )

    raise RuntimeError(
        "Test 3 failed: invalid field type was accepted"
    )

except AgentError as error:

    assert error.code == "INVALID_FIELD_TYPE"

    print(
        "Test 3 passed:",
        error.code,
    )


# ---------------------------------------------------------
# Test 4: Unexpected field
# ---------------------------------------------------------

try:
    executor.execute(
        read_repository_tool,
        {
            "repository": "payment-service",
            "password": "secret",
        },
        context,
    )

    raise RuntimeError(
        "Test 4 failed: unexpected field was accepted"
    )

except AgentError as error:

    assert error.code == "UNEXPECTED_FIELD"

    print(
        "Test 4 passed:",
        error.code,
    )


# ---------------------------------------------------------
# Test 5: Approval flow
# ---------------------------------------------------------

approval_request = executor.execute(
    merge_pull_request_tool,
    {
        "repository": "payment-service",
        "pull_request_number": 42,
    },
    context,
)

assert approval_request.status == ApprovalStatus.PENDING

print(
    "Test 5 passed: approval required"
)


# ---------------------------------------------------------
# Test 6: Approve and execute
# ---------------------------------------------------------

approval_request = approval_service.approve(
    approval_request.request_id
)

assert approval_request is not None
assert approval_request.status == ApprovalStatus.APPROVED

result = executor.execute_approved(
    approval_request,
    merge_pull_request_tool,
    context,
)

assert result == {
    "repository": "payment-service",
    "pull_request_number": 42,
    "status": "merged",
}

print(
    "Test 6 passed: approved execution completed"
)


# ---------------------------------------------------------
# Test 7: Verify audit events
# ---------------------------------------------------------

audit_events = audit_service.list_events()

assert len(audit_events) > 0

print(
    "Test 7 passed: audit events recorded"
)


print()
print("ToolExecutor integration tests passed.")
