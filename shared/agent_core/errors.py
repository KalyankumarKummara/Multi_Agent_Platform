from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    TOOL = "tool"
    EXTERNAL_SERVICE = "external_service"
    TIMEOUT = "timeout"
    POLICY = "policy"
    APPROVAL = "approval"
    INTERNAL = "internal"


@dataclass
class AgentError(Exception):
    code: str
    message: str
    category: ErrorCategory
    retryable: bool
    details: Any = None

    def __post_init__(self):
        super().__init__(self.message)