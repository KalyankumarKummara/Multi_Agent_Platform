from shared.agent_core.lifecycle import AgentLifecycle
from shared.agent_core.state import AgentStatus


lifecycle = AgentLifecycle()


print("Initial status:")
print(lifecycle.status)


# CREATED -> INITIALIZING
lifecycle.initialize()

print("\nAfter initialize:")
print(lifecycle.status)


# INITIALIZING -> READY
lifecycle.mark_ready()

print("\nAfter mark_ready:")
print(lifecycle.status)


# READY -> RUNNING
lifecycle.start()

print("\nAfter start:")
print(lifecycle.status)


# RUNNING -> WAITING
lifecycle.wait()

print("\nAfter wait:")
print(lifecycle.status)


# WAITING -> RUNNING
lifecycle.start()

print("\nAfter restart from waiting:")
print(lifecycle.status)


# RUNNING -> STOPPING
lifecycle.stop()

print("\nAfter stop:")
print(lifecycle.status)


# STOPPING -> STOPPED
lifecycle.stopped()

print("\nAfter stopped:")
print(lifecycle.status)


# Verify that an invalid transition is rejected
try:

    lifecycle.start()

    print("\nERROR: Invalid transition was allowed.")

except RuntimeError as error:

    print("\nInvalid transition test:")
    print(error)


# Verify that the lifecycle state did not change
print("\nFinal status:")
print(lifecycle.status)

assert lifecycle.status == AgentStatus.STOPPED

print("\nLifecycle test passed.")
