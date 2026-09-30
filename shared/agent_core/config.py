from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetryPolicy:
    max_retries: int = 3
    backoff_seconds: float = 1.0


@dataclass
class AgentConfig:
    identity: Any
    enabled: bool = True
    environment: str = "development"
    timeout_seconds: int = 30
    retry_policy: RetryPolicy = field(
        default_factory=RetryPolicy
    )
    llm_config: dict[str, Any] = field(default_factory=dict)
    memory_config: dict[str, Any] = field(default_factory=dict)
    policy_config: dict[str, Any] = field(default_factory=dict)
    tool_config: dict[str, Any] = field(default_factory=dict)