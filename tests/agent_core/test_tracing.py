from shared.agent_core.tracing import (
    create_child_context,
    create_trace_context,
)


# Create the root trace
root_context = create_trace_context()

print("Root trace:")
print("Trace ID:", root_context.trace_id)
print("Span ID:", root_context.span_id)
print("Parent Span ID:", root_context.parent_span_id)


# Create a child span
child_context = create_child_context(root_context)

print("\nChild span:")
print("Trace ID:", child_context.trace_id)
print("Span ID:", child_context.span_id)
print("Parent Span ID:", child_context.parent_span_id)


# Verify the child belongs to the same trace
assert child_context.trace_id == root_context.trace_id

# Verify the child has a different span
assert child_context.span_id != root_context.span_id

# Verify parent relationship
assert child_context.parent_span_id == root_context.span_id


print("\nTracing test passed.")
