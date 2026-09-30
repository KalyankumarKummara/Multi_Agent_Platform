from dataclasses import dataclass


@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str
    name: str
    agent_type: str
    domain: str
    version: str