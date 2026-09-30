from dataclasses import dataclass
from enum import Enum
from typing import Any
import uuid


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ApprovalRequest:
    request_id: str
    action: str
    agent_id: str
    reason: str
    target: Any
    requested_by: str
    risk_level: str
    status: ApprovalStatus = ApprovalStatus.PENDING

def generate_approval_id() -> str:
    return f"approval-{uuid.uuid4()}"

class ApprovalService:

    def __init__(self):
        self._requests: dict[str, ApprovalRequest] = {}

    def request_approval(
        self,
        request: ApprovalRequest
    ) -> ApprovalRequest:

        self._requests[request.request_id] = request

        return request

    def get_status(
        self,
        request_id: str
    ) -> ApprovalStatus | None:

        request = self._requests.get(request_id)

        if request is None:
            return None

        return request.status

    def approve(
        self,
        request_id: str
    ) -> ApprovalRequest | None:

        request = self._requests.get(request_id)

        if request is None:
            return None

        if request.status != ApprovalStatus.PENDING:
            return request

        request.status = ApprovalStatus.APPROVED

        return request

    def reject(
        self,
        request_id: str
    ) -> ApprovalRequest | None:

        request = self._requests.get(request_id)

        if request is None:
            return None

        if request.status != ApprovalStatus.PENDING:
            return request

        request.status = ApprovalStatus.REJECTED

        return request