from typing import Any

from .errors import AgentError, ErrorCategory


class ErrorHandler:

    def handle(
        self,
        error: Exception,
        *,
        default_category: ErrorCategory = ErrorCategory.INTERNAL,
        default_retryable: bool = False,
        details: Any = None,
    ) -> AgentError:

        # If the error is already an AgentError,
        # preserve its existing structured information.
        if isinstance(error, AgentError):
            return error

        # Convert an unexpected exception into
        # our standard AgentError structure.
        return AgentError(
            code="UNEXPECTED_ERROR",
            message=str(error),
            category=default_category,
            retryable=default_retryable,
            details=details,
        )

    def to_dict(
        self,
        error: AgentError,
    ) -> dict[str, Any]:

        return {
            "code": error.code,
            "message": error.message,
            "category": error.category.value,
            "retryable": error.retryable,
            "details": error.details,
        }
