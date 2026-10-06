from shared.agent_core.errors import AgentError
from shared.agent_core.validation import InputValidator


validator = InputValidator()


schema = {
    "repository": {
        "type": "string",
        "required": True,
    },
    "pull_request_number": {
        "type": "integer",
        "required": True,
    },
}


# Test 1: Valid input
validator.validate(
    {
        "repository": "payment-service",
        "pull_request_number": 42,
    },
    schema,
)

print("Test 1 passed: valid input")


# Test 2: Missing required field
try:
    validator.validate(
        {
            "repository": "payment-service",
        },
        schema,
    )

    raise RuntimeError(
        "Test 2 failed: missing field was accepted"
    )

except AgentError as error:
    print(
        "Test 2 passed:",
        error.code,
    )


# Test 3: Wrong type
try:
    validator.validate(
        {
            "repository": "payment-service",
            "pull_request_number": "42",
        },
        schema,
    )

    raise RuntimeError(
        "Test 3 failed: wrong type was accepted"
    )

except AgentError as error:
    print(
        "Test 3 passed:",
        error.code,
    )


# Test 4: Unexpected field
try:
    validator.validate(
        {
            "repository": "payment-service",
            "pull_request_number": 42,
            "password": "secret",
        },
        schema,
    )

    raise RuntimeError(
        "Test 4 failed: unexpected field was accepted"
    )

except AgentError as error:
    print(
        "Test 4 passed:",
        error.code,
    )


# Test 5: Boolean must not be accepted as integer
try:
    validator.validate(
        {
            "repository": "payment-service",
            "pull_request_number": True,
        },
        schema,
    )

    raise RuntimeError(
        "Test 5 failed: boolean was accepted as integer"
    )

except AgentError as error:
    print(
        "Test 5 passed:",
        error.code,
    )


print()
print("Input validation tests passed.")
