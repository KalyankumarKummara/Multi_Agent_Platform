from dataclasses import dataclass


@dataclass(frozen=True)
class AgentCapability:
    capability_id: str
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    required_permissions: list[str]