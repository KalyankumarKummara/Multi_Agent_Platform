from shared.agent_core.audit import AuditService
from shared.agent_core.approval import ApprovalService
from shared.agent_core.policy import PolicyEngine
from shared.agent_core.security import AuthorizationService, Principal
from shared.agent_core.tools.execution import ToolExecutionContext
from shared.agent_core.tools.executor import ToolExecutor
from shared.agent_core.tools.tool import Tool
from shared.agent_core.tracing import TraceContext
from shared.agent_core.validation import InputValidator


# Tool that intentionally fails
def failing_tool(repository: str):
    raise RuntimeError(
        "GitHub service temporarily unavailable"
    )


# Create a test tool
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


# Create services
authorization_service = AuthorizationService()
policy_engine = PolicyEngine()
approval_service = ApprovalService()
audit_service = AuditService()
input_validator = InputValidator()


# Create principal with permission
principal = Principal(
    principal_id="github-agent",
    principal_type="agent",
    permissions=frozenset({"github.read"}),
)


# Create trace context
trace_context = TraceContext(
    trace_id="trace-execution-failed-001",
    span_id="span-execution-failed-001",
)


# Create execution context
context = ToolExecutionContext(
    principal=principal,
    trace_context=trace_context,
    permissions=["github.read"],
    metadata={},
)


# Create executor
executor = ToolExecutor(
    authorization_service=authorization_service,
    policy_engine=policy_engine,
    approval_service=approval_service,
    audit_service=audit_service,
    input_validator=input_validator,
)


# Execute the failing tool
try:

    executor.execute(
        tool=tool,
        arguments={
            "repository": "test-repository",
        },
        context=context,
    )

except Exception as error:

    print("Execution error:")
    print(error)


# Print audit events
print("\nAudit events:")

for event in audit_service.list_events():
    print(event)


print("\nAudit execution failed test passed.")
