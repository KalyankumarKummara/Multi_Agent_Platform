import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None


def generate_trace_id() -> str:
    """
    Generate a unique ID for a complete operation trace.
    """

    return f"trace-{uuid.uuid4()}"


def generate_span_id() -> str:
    """
    Generate a unique ID for one operation/span
    inside a trace.
    """

    return f"span-{uuid.uuid4()}"


def create_trace_context(
    parent_span_id: Optional[str] = None,
) -> TraceContext:
    """
    Create a new trace context.

    If parent_span_id is provided, the new span becomes
    a child of that span.
    """

    return TraceContext(
        trace_id=generate_trace_id(),
        span_id=generate_span_id(),
        parent_span_id=parent_span_id,
    )


def create_child_context(
    context: TraceContext,
) -> TraceContext:
    """
    Create a child span while keeping the same trace ID.
    """

    return TraceContext(
        trace_id=context.trace_id,
        span_id=generate_span_id(),
        parent_span_id=context.span_id,
    )