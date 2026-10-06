from shared.agent_core.error_handler import ErrorHandler
from shared.agent_core.errors import AgentError, ErrorCategory


handler = ErrorHandler()


# Test 1: Convert a normal Python exception
try:

    raise RuntimeError("GitHub service unavailable")

except Exception as error:

    handled_error = handler.handle(
        error,
        default_category=ErrorCategory.EXTERNAL_SERVICE,
        default_retryable=True,
    )

    print("Test 1:")
    print(handled_error)
    print("Code:", handled_error.code)
    print("Category:", handled_error.category)
    print("Retryable:", handled_error.retryable)

    print("Serialized error:")
    print(handler.to_dict(handled_error))


# Test 2: Preserve an existing AgentError
existing_error = AgentError(
    code="AUTHORIZATION_DENIED",
    message="Permission denied",
    category=ErrorCategory.AUTHORIZATION,
    retryable=False,
)

handled_existing_error = handler.handle(existing_error)

print("\nTest 2:")
print(handled_existing_error)
print("Code:", handled_existing_error.code)
print("Category:", handled_existing_error.category)
print("Retryable:", handled_existing_error.retryable)

print("Serialized error:")
print(handler.to_dict(handled_existing_error))
