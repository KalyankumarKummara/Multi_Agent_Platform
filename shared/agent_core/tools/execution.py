from dataclasses import dataclass
from typing import Any

from ..security import Principal
from ..tracing import TraceContext


@dataclass
class ToolExecutionContext:
    principal: Principal
    trace_context: TraceContext
    permissions: list[str]
    metadata: dict[str, Any]