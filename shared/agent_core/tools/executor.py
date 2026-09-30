from typing import Any

from ..approval import ApprovalRequest,ApprovalStatus,ApprovalService,generate_approval_id
from ..errors import AgentError, ErrorCategory
from ..policy import PolicyDecision, PolicyEngine
from ..security import AuthorizationService
from .execution import ToolExecutionContext
from .tool import Tool


class ToolExecutor:

    def __init__(
        self,
        authorization_service: AuthorizationService,
        policy_engine: PolicyEngine,
        approval_service: ApprovalService
    ):
        self.authorization_service = authorization_service
        self.policy_engine = policy_engine
        self.approval_service = approval_service

    def execute(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        context: ToolExecutionContext
    ) -> Any:

        # Step 1: Authorization
        for permission in tool.permissions:

            allowed = self.authorization_service.is_allowed(
                context.principal,
                permission
            )

            if not allowed:
                raise AgentError(
                    code="AUTHORIZATION_DENIED",
                    message=f"Permission denied: {permission}",
                    category=ErrorCategory.AUTHORIZATION,
                    retryable=False,
                    details={
                        "permission": permission,
                        "principal_id": context.principal.principal_id
                    }
                )

        # Step 2: Policy
        policy_result = self.policy_engine.evaluate(
            tool.tool_id,
            context
        )

        if policy_result.decision == PolicyDecision.DENY:
            raise AgentError(
                code="POLICY_DENIED",
                message=f"Policy denied action: {policy_result.reason}",
                category=ErrorCategory.POLICY,
                retryable=False,
                details={
                    "action": tool.tool_id,
                    "reason": policy_result.reason
                }
            )

        if policy_result.decision == PolicyDecision.REQUIRE_APPROVAL:

            approval_request = ApprovalRequest(
                request_id=generate_approval_id(),
                action=tool.tool_id,
                agent_id=context.principal.principal_id,
                reason=policy_result.reason,
                target=arguments,
                requested_by=context.principal.principal_id,
                risk_level=tool.risk_level
            )

            return self.approval_service.request_approval(
                approval_request
            )

        # Step 3: Execute
        return tool.execute(**arguments)

    def execute_approved(
        self,
        approval_request: ApprovalRequest,
        tool: Tool,
        context: ToolExecutionContext
    ) -> Any:

        # Step 1: Verify approval status
        if approval_request.status != ApprovalStatus.APPROVED:
            raise AgentError(
                code="APPROVAL_REQUIRED",
                message="Approval request is not approved.",
                category=ErrorCategory.APPROVAL,
                retryable=False,
                details={
                    "approval_id": approval_request.request_id
                }
            )

        # Step 2: Verify the approved action matches the tool
        if approval_request.action != tool.tool_id:
            raise AgentError(
                code="APPROVAL_ACTION_MISMATCH",
                message="Approved action does not match the tool.",
                category=ErrorCategory.APPROVAL,
                retryable=False,
                details={
                    "approval_action": approval_request.action,
                    "tool_id": tool.tool_id
                }
            )

        # Step 3: Authorization check AGAIN
        for permission in tool.permissions:

            allowed = self.authorization_service.is_allowed(
                context.principal,
                permission
            )

            if not allowed:
                raise AgentError(
                    code="AUTHORIZATION_DENIED",
                    message=f"Permission denied: {permission}",
                    category=ErrorCategory.AUTHORIZATION,
                    retryable=False,
                    details={
                        "permission": permission,
                        "principal_id": context.principal.principal_id
                    }
                )

        # Step 4: Policy check AGAIN
        policy_result = self.policy_engine.evaluate(
            tool.tool_id,
            context
        )

        if policy_result.decision == PolicyDecision.DENY:
            raise AgentError(
                code="POLICY_DENIED",
                message=f"Policy denied action: {policy_result.reason}",
                category=ErrorCategory.POLICY,
                retryable=False,
                details={
                    "action": tool.tool_id,
                    "reason": policy_result.reason
                }
            )

        # Step 5: Execute using the approved target
        return tool.execute(**approval_request.target)