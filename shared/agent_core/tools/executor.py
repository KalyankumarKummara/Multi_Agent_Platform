from datetime import datetime
from typing import Any

from ..approval import (
    ApprovalRequest,
    ApprovalStatus,
    ApprovalService,
    generate_approval_id,
)
from ..audit import (
    AuditService,
    AuditEvent,
    generate_audit_id,
)
from ..errors import AgentError, ErrorCategory
from ..policy import PolicyDecision, PolicyEngine
from ..security import AuthorizationService
from ..validation import InputValidator
from .execution import ToolExecutionContext
from .tool import Tool


class ToolExecutor:

    def __init__(
        self,
        authorization_service: AuthorizationService,
        policy_engine: PolicyEngine,
        approval_service: ApprovalService,
        audit_service: AuditService,
        input_validator: InputValidator,
    ):
        self.authorization_service = authorization_service
        self.policy_engine = policy_engine
        self.approval_service = approval_service
        self.audit_service = audit_service
        self.input_validator = input_validator

    def execute(
        self,
        tool: Tool,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> Any:

        # Step 1: Input validation
        self.input_validator.validate(
            arguments,
            tool.input_schema,
        )

        # Step 2: Authorization
        for permission in tool.permissions:

            allowed = self.authorization_service.is_allowed(
                context.principal,
                permission,
            )

            if not allowed:

                # Record authorization failure in audit
                self.audit_service.record(
                    AuditEvent(
                        audit_id=generate_audit_id(),
                        timestamp=datetime.utcnow(),
                        agent_id=context.principal.principal_id,
                        action=tool.tool_id,
                        target=arguments,
                        actor=context.principal.principal_id,
                        result="authorization_denied",
                        trace_id=context.trace_context.trace_id,
                    )
                )

                raise AgentError(
                    code="AUTHORIZATION_DENIED",
                    message=f"Permission denied: {permission}",
                    category=ErrorCategory.AUTHORIZATION,
                    retryable=False,
                    details={
                        "permission": permission,
                        "principal_id": context.principal.principal_id,
                    },
                )

        # Step 3: Policy
        policy_result = self.policy_engine.evaluate(
            tool.tool_id,
            context,
        )

        if policy_result.decision == PolicyDecision.DENY:

            # Record policy denial in audit
            self.audit_service.record(
                AuditEvent(
                    audit_id=generate_audit_id(),
                    timestamp=datetime.utcnow(),
                    agent_id=context.principal.principal_id,
                    action=tool.tool_id,
                    target=arguments,
                    actor=context.principal.principal_id,
                    result="policy_denied",
                    trace_id=context.trace_context.trace_id,
                )
            )

            raise AgentError(
                code="POLICY_DENIED",
                message=f"Policy denied action: {policy_result.reason}",
                category=ErrorCategory.POLICY,
                retryable=False,
                details={
                    "action": tool.tool_id,
                    "reason": policy_result.reason,
                },
            )

        # Step 4: Human approval if required
        if policy_result.decision == PolicyDecision.REQUIRE_APPROVAL:

            approval_request = ApprovalRequest(
                request_id=generate_approval_id(),
                action=tool.tool_id,
                agent_id=context.principal.principal_id,
                reason=policy_result.reason,
                target=arguments,
                requested_by=context.principal.principal_id,
                risk_level=tool.risk_level,
            )

            approval_request = self.approval_service.request_approval(
                approval_request
            )

            # Record that approval was requested
            self.audit_service.record(
                AuditEvent(
                    audit_id=generate_audit_id(),
                    timestamp=datetime.utcnow(),
                    agent_id=context.principal.principal_id,
                    action=tool.tool_id,
                    target=arguments,
                    actor=context.principal.principal_id,
                    result="approval_required",
                    approval_id=approval_request.request_id,
                    trace_id=context.trace_context.trace_id,
                )
            )

            return approval_request

        # Step 5: Execute
        try:

            result = tool.execute(**arguments)

        except Exception as error:

            # Record tool execution failure in audit
            self.audit_service.record(
                AuditEvent(
                    audit_id=generate_audit_id(),
                    timestamp=datetime.utcnow(),
                    agent_id=context.principal.principal_id,
                    action=tool.tool_id,
                    target=arguments,
                    actor=context.principal.principal_id,
                    result="execution_failed",
                    trace_id=context.trace_context.trace_id,
                )
            )

            raise AgentError(
                code="TOOL_EXECUTION_FAILED",
                message=f"Tool execution failed: {error}",
                category=ErrorCategory.TOOL,
                retryable=True,
                details={
                    "action": tool.tool_id,
                    "error": str(error),
                },
            )

        # Step 6: Record successful execution in audit
        self.audit_service.record(
            AuditEvent(
                audit_id=generate_audit_id(),
                timestamp=datetime.utcnow(),
                agent_id=context.principal.principal_id,
                action=tool.tool_id,
                target=arguments,
                actor=context.principal.principal_id,
                result="success",
                trace_id=context.trace_context.trace_id,
            )
        )

        return result

    def execute_approved(
        self,
        approval_request: ApprovalRequest,
        tool: Tool,
        context: ToolExecutionContext,
    ) -> Any:

        # Step 1: Verify approval status
        if approval_request.status != ApprovalStatus.APPROVED:

            raise AgentError(
                code="APPROVAL_REQUIRED",
                message="Approval request is not approved.",
                category=ErrorCategory.APPROVAL,
                retryable=False,
                details={
                    "approval_id": approval_request.request_id,
                },
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
                    "tool_id": tool.tool_id,
                },
            )

        # Step 3: Validate the approved target
        self.input_validator.validate(
            approval_request.target,
            tool.input_schema,
        )

        # Step 4: Authorization check AGAIN
        for permission in tool.permissions:

            allowed = self.authorization_service.is_allowed(
                context.principal,
                permission,
            )

            if not allowed:

                # Record authorization failure in audit
                self.audit_service.record(
                    AuditEvent(
                        audit_id=generate_audit_id(),
                        timestamp=datetime.utcnow(),
                        agent_id=context.principal.principal_id,
                        action=tool.tool_id,
                        target=approval_request.target,
                        actor=context.principal.principal_id,
                        result="authorization_denied",
                        approval_id=approval_request.request_id,
                        trace_id=context.trace_context.trace_id,
                    )
                )

                raise AgentError(
                    code="AUTHORIZATION_DENIED",
                    message=f"Permission denied: {permission}",
                    category=ErrorCategory.AUTHORIZATION,
                    retryable=False,
                    details={
                        "permission": permission,
                        "principal_id": context.principal.principal_id,
                    },
                )

        # Step 5: Policy check AGAIN
        policy_result = self.policy_engine.evaluate(
            tool.tool_id,
            context,
        )

        if policy_result.decision == PolicyDecision.DENY:

            # Record policy denial in audit
            self.audit_service.record(
                AuditEvent(
                    audit_id=generate_audit_id(),
                    timestamp=datetime.utcnow(),
                    agent_id=context.principal.principal_id,
                    action=tool.tool_id,
                    target=approval_request.target,
                    actor=context.principal.principal_id,
                    result="policy_denied",
                    approval_id=approval_request.request_id,
                    trace_id=context.trace_context.trace_id,
                )
            )

            raise AgentError(
                code="POLICY_DENIED",
                message=f"Policy denied action: {policy_result.reason}",
                category=ErrorCategory.POLICY,
                retryable=False,
                details={
                    "action": tool.tool_id,
                    "reason": policy_result.reason,
                },
            )

        # Step 6: Execute using the approved target
        try:

            result = tool.execute(
                **approval_request.target
            )

        except Exception as error:

            # Record approved tool execution failure in audit
            self.audit_service.record(
                AuditEvent(
                    audit_id=generate_audit_id(),
                    timestamp=datetime.utcnow(),
                    agent_id=context.principal.principal_id,
                    action=tool.tool_id,
                    target=approval_request.target,
                    actor=context.principal.principal_id,
                    result="execution_failed",
                    approval_id=approval_request.request_id,
                    trace_id=context.trace_context.trace_id,
                )
            )

            raise AgentError(
                code="TOOL_EXECUTION_FAILED",
                message=f"Tool execution failed: {error}",
                category=ErrorCategory.TOOL,
                retryable=True,
                details={
                    "action": tool.tool_id,
                    "error": str(error),
                    "approval_id": approval_request.request_id,
                },
            )

        # Step 7: Record successful approved execution in audit
        self.audit_service.record(
            AuditEvent(
                audit_id=generate_audit_id(),
                timestamp=datetime.utcnow(),
                agent_id=context.principal.principal_id,
                action=tool.tool_id,
                target=approval_request.target,
                actor=context.principal.principal_id,
                result="success",
                approval_id=approval_request.request_id,
                trace_id=context.trace_context.trace_id,
            )
        )

        return result
