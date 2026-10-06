from shared.agent_core.audit import AuditService
from shared.agent_core.approval import ApprovalService
from shared.agent_core.policy import PolicyEngine
from shared.agent_core.security import AuthorizationService, Principal
from shared.agent_core.tools.execution import ToolExecutionContext
from shared.agent_core.tools.executor import ToolExecutor
from shared.agent_core.tools.tool import Tool
from shared.agent_core.tracing import create_trace_context
from shared.agent_core.validation import InputValidator


def read_repository(repository: str):
    return {
        "repository": repository,
        "status": "ok",
    }


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
    execute=read_repository,
)


authorization_service = AuthorizationService()
policy_engine = PolicyEngine()
approval_service = ApprovalService()
audit_service = AuditService()
input_validator = InputValidator()


principal = Principal(
    principal_id="github-agent",
    principal_type="agent",
    permissions=frozenset({"github.read"}),
)


trace_context = create_trace_context()


context = ToolExecutionContext(
    principal=principal,
    trace_context=trace_context,
    permissions=["github.read"],
    metadata={},
)


executor = ToolExecutor(
    authorization_service=authorization_service,
    policy_engine=policy_engine,
    approval_service=approval_service,
    audit_service=audit_service,
    input_validator=input_validator,
)


result = executor.execute(
    tool=tool,
    arguments={
        "repository": "test-repository",
    },
    context=context,
)


print("Tool result:")
print(result)

print("\nTrace context:")
print("Trace ID:", trace_context.trace_id)
print("Span ID:", trace_context.span_id)

print("\nAudit events:")

for event in audit_service.list_events():

    print(event)

    assert event.trace_id == trace_context.trace_id


print("\nTracing + audit integration test passed.")
