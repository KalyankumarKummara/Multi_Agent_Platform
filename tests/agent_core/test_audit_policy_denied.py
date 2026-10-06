from shared.agent_core.audit import AuditService
from shared.agent_core.approval import ApprovalService
from shared.agent_core.policy import PolicyEngine
from shared.agent_core.security import AuthorizationService, Principal
from shared.agent_core.tools.execution import ToolExecutionContext
from shared.agent_core.tools.executor import ToolExecutor
from shared.agent_core.tools.tool import Tool
from shared.agent_core.tracing import TraceContext
from shared.agent_core.validation import InputValidator


def mock_delete_repository(repository: str):
    return {
        "repository": repository,
        "status": "deleted",
    }


principal = Principal(
    principal_id="github-agent",
    principal_type="agent",
    permissions=frozenset({"github.delete"}),
)


tool = Tool(
    tool_id="delete_repository",
    name="Delete Repository",
    description="Delete a repository",
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
    permissions=["github.delete"],
    risk_level="critical",
    execute=mock_delete_repository,
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
        trace_id="trace-policy-denied-001",
        span_id="span-policy-denied-001",
    ),
    permissions=["github.delete"],
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


print("\nAudit policy denied test passed.")
