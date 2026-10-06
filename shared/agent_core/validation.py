from typing import Any

from .errors import AgentError, ErrorCategory


class InputValidator:

    SUPPORTED_TYPES = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    def validate(
        self,
        arguments: dict[str, Any],
        schema: dict[str, Any],
    ) -> None:

        if not isinstance(arguments, dict):
            raise AgentError(
                code="INVALID_INPUT",
                message="Tool arguments must be a dictionary.",
                category=ErrorCategory.VALIDATION,
                retryable=False,
            )

        # Check required fields.
        for field_name, field_schema in schema.items():

            required = field_schema.get("required", False)

            if required and field_name not in arguments:
                raise AgentError(
                    code="MISSING_REQUIRED_FIELD",
                    message=f"Required field is missing: {field_name}",
                    category=ErrorCategory.VALIDATION,
                    retryable=False,
                    details={
                        "field": field_name,
                    },
                )

        # Check for unexpected fields.
        for field_name in arguments:

            if field_name not in schema:
                raise AgentError(
                    code="UNEXPECTED_FIELD",
                    message=f"Unexpected field: {field_name}",
                    category=ErrorCategory.VALIDATION,
                    retryable=False,
                    details={
                        "field": field_name,
                    },
                )

        # Check field types.
        for field_name, value in arguments.items():

            field_schema = schema[field_name]
            expected_type_name = field_schema.get("type")

            expected_type = self.SUPPORTED_TYPES.get(
                expected_type_name
            )

            if expected_type is None:
                raise AgentError(
                    code="UNSUPPORTED_FIELD_TYPE",
                    message=(
                        f"Unsupported field type: "
                        f"{expected_type_name}"
                    ),
                    category=ErrorCategory.VALIDATION,
                    retryable=False,
                    details={
                        "field": field_name,
                        "type": expected_type_name,
                    },
                )

            # bool is technically a subclass of int in Python.
            # Handle integer validation explicitly.
            if expected_type_name == "integer":
                valid = (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                )
            else:
                valid = isinstance(value, expected_type)

            if not valid:
                raise AgentError(
                    code="INVALID_FIELD_TYPE",
                    message=(
                        f"Invalid type for field '{field_name}'. "
                        f"Expected {expected_type_name}."
                    ),
                    category=ErrorCategory.VALIDATION,
                    retryable=False,
                    details={
                        "field": field_name,
                        "expected_type": expected_type_name,
                        "actual_type": type(value).__name__,
                    },
                )
