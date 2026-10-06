from shared.agent_core.audit import AuditService
from shared.agent_core.approval import ApprovalService
from shared.agent_core.policy import PolicyEngine
from shared.agent_core.security import AuthorizationService, Principal
from shared.agent_core.tools.execution import ToolExecutionContext
from shared.agent_core.tools.executor import ToolExecutor
from shared.agent_core.tools.tool import Tool
from shared.agent_core.tracing import TraceContext
from shared.agent_core.validation import InputValidator


def mock_merge_pull_request(
    repository: str,
    pull_request: int,
):
    return {
        "repository": repository,
        "pull_request": pull_request,
        "status": "merged",
    }


principal = Principal(
    principal_id="github-agent",
    principal_type="agent",
    permissions=frozenset({"github.write"}),
)


tool = Tool(
    tool_id="merge_pull_request",
    name="Merge Pull Request",
    description="Merge a pull request",
    input_schema={
        "repository": {
            "type": "string",
            "required": True,
        },
        "pull_request": {
            "type": "integer",
            "required": True,
        },
    },
    output_schema={
        "repository": "string",
        "pull_request": "integer",
        "status": "string",
    },
    permissions=["github.write"],
    risk_level="high",
    execute=mock_merge_pull_request,
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
        trace_id="trace-approval-001",
        span_id="span-approval-001",
    ),
    permissions=["github.write"],
    metadata={},
)


# --------------------------------------------------
# Step 1: Request approval
# --------------------------------------------------

approval_request = executor.execute(
    tool=tool,
    arguments={
        "repository": "test-repository",
        "pull_request": 42,
    },
    context=context,
)


print("Approval request:")
print(approval_request)


print("\nApproval status before approval:")

print(
    approval_service.get_status(
        approval_request.request_id
    )
)


# --------------------------------------------------
# Step 2: Approve the request
# --------------------------------------------------

approved_request = approval_service.approve(
    approval_request.request_id
)


print("\nApproval request after approval:")
print(approved_request)


print("\nApproval status after approval:")

print(
    approval_service.get_status(
        approval_request.request_id
    )
)


# --------------------------------------------------
# Step 3: Execute approved action
# --------------------------------------------------

result = executor.execute_approved(
    approval_request=approved_request,
    tool=tool,
    context=context,
)


print("\nApproved execution result:")
print(result)


# --------------------------------------------------
# Step 4: Display audit events
# --------------------------------------------------

print("\nAudit events:")

for event in audit_service.list_events():
    print(event)


print("\nAudit approval test passed.")
