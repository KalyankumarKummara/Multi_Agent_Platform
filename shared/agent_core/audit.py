import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class AuditEvent:
    audit_id: str
    timestamp: datetime
    agent_id: str
    action: str
    target: Any
    actor: str
    result: str
    approval_id: str | None = None
    trace_id: str | None = None

def generate_audit_id() -> str:
    return f"audit-{uuid.uuid4()}"

class AuditService:

    def __init__(self):
        self._events: list[AuditEvent] = []

    def record(
        self,
        event: AuditEvent
    ) -> AuditEvent:

        self._events.append(event)

        return event

    def list_events(self) -> list[AuditEvent]:

        return list(self._events)