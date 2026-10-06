from shared.agent_core.context import AgentContext
from shared.agent_core.identity import AgentIdentity
from shared.agent_core.tools.tool import Tool


# Create agent identity
identity = AgentIdentity(
    agent_id="github-agent",
    name="GitHub Agent",
    agent_type="specialist",
    domain="github",
    version="1.0.0",
)


# Create a test tool
def read_repository(repository: str):
    return {
        "repository": repository,
        "status": "ok",
    }


tool = Tool(
    tool_id="read_repository",
    name="Read Repository",
    description="Reads repository information",
    input_schema={
    "repository": {
        "type": "string",
        "required": True,
    },
},
    output_schema={
        "repository": "string",
    },
    permissions=["github.read"],
    risk_level="low",
    execute=read_repository,
)


# Create context
context = AgentContext(
    agent_identity=identity,
)


print("Agent:")
print(context.agent_identity)

print("\nInitial permissions:")
print(context.permissions)


# Add permission
context.add_permission("github.read")

print("\nPermissions after adding github.read:")
print(context.permissions)

print("\nHas github.read permission:")
print(context.has_permission("github.read"))

print("\nHas github.write permission:")
print(context.has_permission("github.write"))


# Try adding the same permission again
context.add_permission("github.read")

print("\nPermissions after adding github.read again:")
print(context.permissions)


# Add history
context.add_history({
    "event": "repository_read",
    "repository": "test-repository",
})

print("\nRelevant history:")
print(context.relevant_history)


# Add tool
context.add_tool(tool)

print("\nAvailable tools:")
print(context.available_tools)

print("\nHas read_repository tool:")
print(context.has_tool("read_repository"))

print("\nHas merge_pull_request tool:")
print(context.has_tool("merge_pull_request"))
