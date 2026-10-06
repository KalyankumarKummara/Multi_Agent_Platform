from shared.agent_core.audit import AuditService
from shared.agent_core.approval import ApprovalService
from shared.agent_core.error_handler import ErrorHandler
from shared.agent_core.policy import PolicyEngine
from shared.agent_core.security import AuthorizationService, Principal
from shared.agent_core.tools.execution import ToolExecutionContext
from shared.agent_core.tools.executor import ToolExecutor
from shared.agent_core.tools.tool import Tool
from shared.agent_core.tracing import TraceContext
from shared.agent_core.validation import InputValidator


# ---------------------------------------------------------
# 1. Create a tool that intentionally fails
# ---------------------------------------------------------

def failing_tool(repository: str):
    raise RuntimeError(
        "GitHub service temporarily unavailable"
    )


tool = Tool(
    tool_id="read_repository",
    name="Read Repository",
    description="Reads repository information",
    input_schema={
        "repository": {
            "type": "string",
            "required": True,
        },
    },
    output_schema={
        "repository": "string",
    },
    permissions=["github.read"],
    risk_level="low",
    execute=failing_tool,
)


# ---------------------------------------------------------
# 2. Create foundation services
# ---------------------------------------------------------

authorization_service = AuthorizationService()
policy_engine = PolicyEngine()
approval_service = ApprovalService()
audit_service = AuditService()
error_handler = ErrorHandler()
input_validator = InputValidator()


# ---------------------------------------------------------
# 3. Create an authorized agent principal
# ---------------------------------------------------------

principal = Principal(
    principal_id="github-agent",
    principal_type="agent",
    permissions=frozenset({"github.read"}),
)


# ---------------------------------------------------------
# 4. Create trace context
# ---------------------------------------------------------

trace_context = TraceContext(
    trace_id="trace-error-integration-001",
    span_id="span-error-integration-001",
)


# ---------------------------------------------------------
# 5. Create tool execution context
# ---------------------------------------------------------

context = ToolExecutionContext(
    principal=principal,
    trace_context=trace_context,
    permissions=["github.read"],
    metadata={},
)


# ---------------------------------------------------------
# 6. Create ToolExecutor
# ---------------------------------------------------------

executor = ToolExecutor(
    authorization_service=authorization_service,
    policy_engine=policy_engine,
    approval_service=approval_service,
    audit_service=audit_service,
    input_validator=input_validator,
)


# ---------------------------------------------------------
# 7. Execute the failing tool
# ---------------------------------------------------------

try:

    executor.execute(
        tool=tool,
        arguments={
            "repository": "test-repository",
        },
        context=context,
    )

except Exception as error:

    print("Original error:")
    print(error)

    # Send the error through centralized ErrorHandler
    handled_error = error_handler.handle(error)

    print("\nHandled error:")
    print(handled_error)

    print("\nError code:")
    print(handled_error.code)

    print("\nError category:")
    print(handled_error.category)

    print("\nRetryable:")
    print(handled_error.retryable)

    print("\nSerialized error:")
    print(error_handler.to_dict(handled_error))


# ---------------------------------------------------------
# 8. Verify audit was still recorded
# ---------------------------------------------------------

print("\nAudit events:")

for event in audit_service.list_events():
    print(event)


print("\nError handler integration test passed.")
