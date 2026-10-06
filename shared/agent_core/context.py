from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    agent_identity: Any

    event: Any = None
    request: Any = None
    user: Any = None
    tenant: Any = None
    project: Any = None

    permissions: list[str] = field(default_factory=list)
    relevant_history: list[Any] = field(default_factory=list)

    memory: Any = None
    available_tools: list[Any] = field(default_factory=list)

    policy_context: Any = None
    trace_context: Any = None

    def has_permission(self, permission: str) -> bool:
        """
        Check whether the current context contains
        the requested permission.
        """

        return permission in self.permissions

    def add_permission(self, permission: str) -> None:
        """
        Add a permission if it is not already present.
        """

        if permission not in self.permissions:
            self.permissions.append(permission)

    def add_history(self, item: Any) -> None:
        """
        Add an item to the relevant context history.
        """

        self.relevant_history.append(item)

    def add_tool(self, tool: Any) -> None:
        """
        Add an available tool to the context.
        """

        self.available_tools.append(tool)

    def has_tool(self, tool_id: str) -> bool:
        """
        Check whether a tool with the given ID
        is available in the current context.
        """

        for tool in self.available_tools:

            if getattr(tool, "tool_id", None) == tool_id:
                return True

        return False