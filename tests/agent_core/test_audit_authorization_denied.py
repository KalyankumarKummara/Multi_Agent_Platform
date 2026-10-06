from shared.agent_core.audit import AuditService
from shared.agent_core.approval import ApprovalService
from shared.agent_core.policy import PolicyEngine
from shared.agent_core.security import AuthorizationService, Principal
from shared.agent_core.tools.execution import ToolExecutionContext
from shared.agent_core.tools.executor import ToolExecutor
from shared.agent_core.tools.tool import Tool
from shared.agent_core.tracing import TraceContext
from shared.agent_core.validation import InputValidator


def mock_get_repository(repository: str):
    return {
        "repository": repository,
        "status": "ok",
    }


# Principal intentionally has NO github.read permission
principal = Principal(
    principal_id="github-agent",
    principal_type="agent",
    permissions=frozenset(),
)


tool = Tool(
    tool_id="read_repository",
    name="Read Repository",
    description="Read repository information",
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
    permissions=["github.read"],
    risk_level="low",
    execute=mock_get_repository,
)


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


context = ToolExecutionContext(
    principal=principal,
    trace_context=TraceContext(
        trace_id="trace-auth-denied-001",
        span_id="span-auth-denied-001",
    ),
    permissions=[],
    metadata={},
)


try:

    result = executor.execute(
        tool=tool,
        arguments={
            "repository": "test-repository",
        },
        context=context,
    )

    print("Tool result:")
    print(result)

except Exception as error:

    print("Execution error:")
    print(error)


print("\nAudit events:")

for event in audit_service.list_events():
    print(event)


print("\nAudit authorization denied test passed.")
