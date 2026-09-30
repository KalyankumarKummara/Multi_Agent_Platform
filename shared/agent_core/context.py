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